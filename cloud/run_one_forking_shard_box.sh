#!/usr/bin/env bash
#
# run_one_forking_shard_box.sh — ONE shard box of the SHARDED forking-paths fleet.
#
# Stands up EXACTLY ONE Vast.ai GPU box, pushes the LOCAL selected_item.json +
# base_path.json (produced by the phase-1 base box) UP to the box's out dir, then
# forks ONLY this box's slice of base-path positions (positions[SHARD_INDEX::NUM_SHARDS])
# via collect_forking_shard.py, and syncs the resulting forking_shard_<k>_of_<N>.json
# back into a DISJOINT local quarantine (sync/forkshards/shard_<k>/...), then DESTROYS
# the box. Destroy is on an EXIT trap so the box can never be left billing.
#
# SHARD_INDEX (k) and NUM_SHARDS (N) are REQUIRED env. Lifecycle mirrors
# run_one_forking_box_32b.sh; the long fork decode runs DETACHED + polled so an SSH
# drop mid-decode never loses the run. Env knobs mirror the 32b script.
#
# Required env:  SHARD_INDEX  NUM_SHARDS
# Optional env overrides:
#   GPU_NAME (H100_SXM) NUM_GPUS (1) MAX_PRICE (3.0) DISK (160) MIN_RELIABILITY (0.985)
#   N_PRIOR (50) MAX_NEW_TOKENS (768) TEMPERATURE (1.0) HF_GEN_MICRO_BATCH (24)

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="/root/apart"

MODEL="${MODEL:-Qwen/Qwen3-14B}"
BARE_MODEL="${MODEL##*/}"
SHARD_INDEX="${SHARD_INDEX:?SHARD_INDEX env required}"
NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS env required}"
GPU_NAME="${GPU_NAME:-H100_SXM}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_PRICE="${MAX_PRICE:-3.0}"
DISK="${DISK:-160}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.985}"

N_PRIOR="${N_PRIOR:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
TEMPERATURE="${TEMPERATURE:-1.0}"
HF_GEN_MICRO_BATCH="${HF_GEN_MICRO_BATCH:-24}"

OUT_DIR_REL="out/sesgo/forking/$BARE_MODEL"
LOCAL_ARTIFACTS="$REPO_ROOT/sync/forkbase/sesgo/forking/$BARE_MODEL"
IID_FILE="$HERE/.forkshard_${SHARD_INDEX}.iid"
INSTANCE=""

log() { echo "[forkshard$SHARD_INDEX $(date +%H:%M:%S)] $*"; }

destroy_box() {
  local rc=$?
  if [ -n "$INSTANCE" ]; then
    log "DESTROY instance $INSTANCE (exit rc=$rc)"
    INSTANCE="$INSTANCE" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure 2>&1 | sed "s/^/[forkshard$SHARD_INDEX destroy] /"
    sleep 5
    printf 'y\n' | vastai destroy instance "$INSTANCE" >/dev/null 2>&1 || true
    rm -f "$IID_FILE"
  fi
  return $rc
}
trap destroy_box EXIT

run_on_box() {
  local cmd="$1"
  . "$HERE/_ssh_target.sh"
  _resolve_ssh_target || return 1
  ssh -F /dev/null -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=240 \
      -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" \
      "cd $REMOTE_ROOT 2>/dev/null || true; $cmd"
}

# run a LONG command DETACHED on the box, then poll for completion over reconnecting
# SSH so a dropped session never loses the fork decode. Returns the remote exit code.
run_detached_and_wait() {
  local tag="$1" cmd="$2"
  local rlog="/root/apart/.${tag}.out" rdone="/root/apart/.${tag}.done"
  run_on_box "rm -f $rlog $rdone; \
    setsid bash -c 'cd $REMOTE_ROOT; { $cmd; } > $rlog 2>&1; echo \$? > $rdone' \
    </dev/null >/dev/null 2>&1 &" || { log "FATAL: could not launch $tag detached"; return 99; }
  log "launched '$tag' DETACHED; polling $rdone (log: $rlog)"
  local poll=20 drops=0 max_drops=30 seen=0
  while true; do
    local done_val tail_out
    done_val="$(run_on_box "cat $rdone 2>/dev/null" 2>/dev/null)"
    if [ -n "$done_val" ]; then
      run_on_box "tail -n 25 $rlog 2>/dev/null" 2>/dev/null | sed "s/^/[forkshard$SHARD_INDEX $tag] /"
      done_val="$(printf '%s' "$done_val" | tr -dc '0-9')"
      log "'$tag' finished with exit code ${done_val:-?}"
      return "${done_val:-1}"
    fi
    tail_out="$(run_on_box "wc -l < $rlog 2>/dev/null" 2>/dev/null | tr -dc '0-9')"
    if [ -n "$tail_out" ]; then
      drops=0
      if [ "$tail_out" -gt "$seen" ]; then
        run_on_box "sed -n '$((seen+1)),\$p' $rlog 2>/dev/null" 2>/dev/null | sed "s/^/[forkshard$SHARD_INDEX $tag] /"
        seen="$tail_out"
      fi
    else
      drops=$((drops+1)); log "'$tag' poll: no response ($drops/$max_drops)"
      [ "$drops" -ge "$max_drops" ] && { log "FATAL: box unreachable -- aborting"; return 98; }
    fi
    sleep "$poll"
  done
}

# ── 0. Require the local phase-1 artifacts BEFORE spending money on a box ──
[ -f "$LOCAL_ARTIFACTS/selected_item.json" ] || { log "FATAL: missing $LOCAL_ARTIFACTS/selected_item.json (run the base box first)"; exit 1; }
[ -f "$LOCAL_ARTIFACTS/base_path.json" ] || { log "FATAL: missing $LOCAL_ARTIFACTS/base_path.json (run the base box first)"; exit 1; }

# ── 1. SEARCH offer ──
QUERY="gpu_name=${GPU_NAME} num_gpus=${NUM_GPUS} verified=true rentable=true direct_port_count>=1 gpu_ram>=79 disk_space>=${DISK} dph_total<=${MAX_PRICE}"
log "search offers: $QUERY"
OFFERS_JSON="$(vastai search offers "$QUERY" -o 'dph_total+' --raw 2>/dev/null)"
OFFER_ID="$(printf '%s' "$OFFERS_JSON" | MIN_REL="$MIN_RELIABILITY" python3 -c '
import sys, json, os
floor=float(os.environ["MIN_REL"]); o=json.load(sys.stdin)
flt=[r for r in o if r.get("reliability2",0)>=floor]
print(flt[0]["id"] if flt else "")')"
[ -n "$OFFER_ID" ] || { log "FATAL: no matching offers for $GPU_NAME"; exit 2; }

# ── 2. CREATE ──
PORTAL_ENV='-p 1111:1111 -e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1" -e JUPYTER_DIR="/" -e DATA_DIRECTORY="/workspace/"'
CREATE_JSON="$(vastai create instance "$OFFER_ID" \
  --image 'vastai/pytorch:@vastai-automatic-tag' \
  --env "$PORTAL_ENV" --onstart-cmd 'entrypoint.sh' --disk "$DISK" --ssh --direct --raw)"
INSTANCE="$(printf '%s' "$CREATE_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("new_contract",""))')"
[ -n "$INSTANCE" ] || { log "FATAL: create failed: $CREATE_JSON"; exit 3; }
echo "$INSTANCE" > "$IID_FILE"
log "created instance $INSTANCE; waiting for 'running'"

# ── 3. POLL until running ──
instance_status() {
  INSTANCE_ID="$INSTANCE" python3 -c '
import json, os, subprocess, sys
iid=int(os.environ["INSTANCE_ID"]); token=None
for _ in range(40):
    cmd=["vastai","show","instances-v1","--raw"]
    if token: cmd+=["--next-token",token]
    try: d=json.loads(subprocess.run(cmd,capture_output=True,text=True).stdout)
    except Exception: break
    rows=d if isinstance(d,list) else d.get("instances",d.get("results",[]))
    for r in rows:
        if isinstance(r,dict) and r.get("id")==iid:
            print(r.get("actual_status") or r.get("cur_state") or "unknown"); sys.exit()
    token=d.get("next_token") if isinstance(d,dict) else None
    if not token or not rows: break
print("missing")'
}
for i in $(seq 1 120); do
  ST="$(instance_status)"; log "  [$i] status: $ST"
  [ "$ST" = "running" ] && break
  sleep 15
done
[ "$ST" = "running" ] || { log "FATAL: instance never reached running"; exit 4; }

# ── 4. Wait for sshd ──
log "waiting for sshd"
for i in $(seq 1 50); do
  if run_on_box "echo ssh_ok" 2>/dev/null | grep -q ssh_ok; then log "sshd up"; break; fi
  sleep 10
done

# ── 5. SYNC code UP + at_setup ──
log "sync_up (code + SESGO prompts)"
INSTANCE="$INSTANCE" bash "$HERE/sync_up.sh" 2>&1 | sed "s/^/[forkshard$SHARD_INDEX up] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_up failed"; exit 5; }
log "at_setup"
run_on_box "bash cloud/at_setup.sh" 2>&1 | sed "s/^/[forkshard$SHARD_INDEX setup] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: at_setup failed"; exit 6; }

# ── 6. PUSH the LOCAL selected_item.json + base_path.json UP to the box out dir ──
log "push selected_item.json + base_path.json -> box:$OUT_DIR_REL/"
. "$HERE/_ssh_target.sh"; _resolve_ssh_target || { log "FATAL: cannot resolve ssh"; exit 7; }
run_on_box "mkdir -p $OUT_DIR_REL"
RSYNC_E="ssh -F /dev/null -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i $SSH_KEY -p $SSH_PORT"
rsync -ah -e "$RSYNC_E" \
  "$LOCAL_ARTIFACTS/selected_item.json" "$LOCAL_ARTIFACTS/base_path.json" \
  "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/$OUT_DIR_REL/" 2>&1 | sed "s/^/[forkshard$SHARD_INDEX push] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: artifact push failed"; exit 7; }

# ── 7. FORK this shard's position slice (detached + polled) ──
log "fork shard $SHARD_INDEX/$NUM_SHARDS ($MODEL): max-new-tokens=$MAX_NEW_TOKENS n-prior=$N_PRIOR T=$TEMPERATURE micro-batch=$HF_GEN_MICRO_BATCH"
run_detached_and_wait collect \
  "HF_GEN_MICRO_BATCH=$HF_GEN_MICRO_BATCH .venv/bin/python sesgo/forking/collect_forking_shard.py --model $MODEL \
     --shard-index $SHARD_INDEX --num-shards $NUM_SHARDS \
     --n-prior $N_PRIOR --max-new-tokens $MAX_NEW_TOKENS --temperature $TEMPERATURE"
[ $? -eq 0 ] || { log "FATAL: shard collect failed"; exit 8; }

# ── 8. Verify the shard file landed BEFORE we destroy ──
SHARD_FILE="$OUT_DIR_REL/forking_shard_${SHARD_INDEX}_of_${NUM_SHARDS}.json"
NPOS="$(run_on_box ".venv/bin/python -c \"import json; print(len(json.load(open('$SHARD_FILE'))['positions']))\" 2>/dev/null || echo 0")"
NPOS="$(printf '%s' "$NPOS" | tr -dc '0-9')"
log "remote $SHARD_FILE has ${NPOS:-0} positions"
[ "${NPOS:-0}" -ge 1 ] || { log "FATAL: empty/missing shard file -- aborting"; exit 9; }

# ── 9. SYNC the shard file BACK into a DISJOINT quarantine sync/forkshards/shard_<k>/ ──
log "sync_back -> sync/forkshards/shard_$SHARD_INDEX/sesgo/forking/"
SYNC_SUBDIR="forkshards/shard_$SHARD_INDEX" STUDIES="forking" INSTANCE="$INSTANCE" \
  bash "$HERE/sync_back.sh" 2>&1 | sed "s/^/[forkshard$SHARD_INDEX back] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_back failed"; exit 10; }

log "SUCCESS: shard $SHARD_INDEX/$NUM_SHARDS forked for $BARE_MODEL ($NPOS positions); quarantined under sync/forkshards/shard_$SHARD_INDEX/"
exit 0
