#!/usr/bin/env bash
#
# run_one_forking_base_box.sh — PHASE 1 box of the SHARDED forking-paths fleet.
#
# Stands up EXACTLY ONE Vast.ai GPU box, runs item SELECTION + the BASE-PATH decode
# (select_forking_item.py -> decode_forking_base_path.py) for $MODEL, and syncs the
# two small artifacts the shard boxes need — selected_item.json + base_path.json —
# back into a DISJOINT local quarantine (sync/forkbase/...), then DESTROYS the box.
# Destroy is on an EXIT trap so the box is torn down on success, failure, OR
# interruption — it can never be left billing.
#
# This is the cheap phase: it does NOT fork any positions (that is the shard boxes'
# job). It only greedily decodes the full CoT once and enumerates every branch
# prefix, so a single shorter-lived box suffices. The shard fleet then loads the
# resulting base_path.json N times and forks disjoint position slices in parallel.
#
# Lifecycle mirrors run_one_forking_box_32b.sh (search offer -> create -> poll
# running -> wait sshd -> sync_up -> at_setup -> select -> decode -> sync_back ->
# destroy-on-EXIT). Env knobs mirror the 32b script.
#
# Optional env overrides:
#   GPU_NAME (H100_SXM) NUM_GPUS (1) MAX_PRICE (3.0) DISK (160) MIN_RELIABILITY (0.985)
#   N_SAMPLES (50) BASE_MAX_NEW_TOKENS (768) NEAR_WINDOW (0)
#   SELECT_CATEGORIES (gender) SELECT_LANGUAGES (es) SELECT_N_PILOT (12) SELECT_MAX_ITEMS (10)

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="/root/apart"

MODEL="${MODEL:-Qwen/Qwen3-14B}"
BARE_MODEL="${MODEL##*/}"
GPU_NAME="${GPU_NAME:-H100_SXM}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_PRICE="${MAX_PRICE:-3.0}"
DISK="${DISK:-160}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.985}"

N_SAMPLES="${N_SAMPLES:-50}"
BASE_MAX_NEW_TOKENS="${BASE_MAX_NEW_TOKENS:-768}"
NEAR_WINDOW="${NEAR_WINDOW:-0}"

SELECT_SESGO_DIR="${SELECT_SESGO_DIR:-datasets/SESGO}"
SELECT_CATEGORIES="${SELECT_CATEGORIES:-gender}"
SELECT_LANGUAGES="${SELECT_LANGUAGES:-es}"
SELECT_N_PILOT="${SELECT_N_PILOT:-12}"
SELECT_MAX_ITEMS="${SELECT_MAX_ITEMS:-10}"

IID_FILE="$HERE/.forkbase.iid"
INSTANCE=""

log() { echo "[forkbase $(date +%H:%M:%S)] $*"; }

# ── DESTROY-ON-EXIT: tear the box down no matter how we leave this script ──
destroy_box() {
  local rc=$?
  if [ -n "$INSTANCE" ]; then
    log "DESTROY instance $INSTANCE (exit rc=$rc)"
    INSTANCE="$INSTANCE" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure 2>&1 | sed "s/^/[forkbase destroy] /"
    sleep 5
    printf 'y\n' | vastai destroy instance "$INSTANCE" >/dev/null 2>&1 || true
    rm -f "$IID_FILE"
  fi
  return $rc
}
trap destroy_box EXIT

run_on_box() {  # run a command on this box via direct ssh (fresh endpoint each call)
  local cmd="$1"
  . "$HERE/_ssh_target.sh"
  _resolve_ssh_target || return 1
  ssh -F /dev/null -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=240 \
      -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" \
      "cd $REMOTE_ROOT 2>/dev/null || true; $cmd"
}

# ── 1. SEARCH for a matching verified offer (reliability floor enforced) ──
QUERY="gpu_name=${GPU_NAME} num_gpus=${NUM_GPUS} verified=true rentable=true direct_port_count>=1 gpu_ram>=79 disk_space>=${DISK} dph_total<=${MAX_PRICE}"
log "search offers: $QUERY (rel2 floor $MIN_RELIABILITY enforced post-hoc)"
OFFERS_JSON="$(vastai search offers "$QUERY" -o 'dph_total+' --raw 2>/dev/null)"
OFFER_ID="$(printf '%s' "$OFFERS_JSON" | MIN_REL="$MIN_RELIABILITY" python3 -c '
import sys, json, os
floor=float(os.environ["MIN_REL"]); o=json.load(sys.stdin)
flt=[r for r in o if r.get("reliability2",0)>=floor]
print(flt[0]["id"] if flt else "")')"
[ -n "$OFFER_ID" ] || { log "FATAL: no matching offers for $GPU_NAME"; exit 2; }

# ── 2. CREATE the instance (PyTorch image, SSH, direct) ──
PORTAL_ENV='-p 1111:1111 -e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1" -e JUPYTER_DIR="/" -e DATA_DIRECTORY="/workspace/"'
CREATE_JSON="$(vastai create instance "$OFFER_ID" \
  --image 'vastai/pytorch:@vastai-automatic-tag' \
  --env "$PORTAL_ENV" --onstart-cmd 'entrypoint.sh' \
  --disk "$DISK" --ssh --direct --raw)"
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
INSTANCE="$INSTANCE" bash "$HERE/sync_up.sh" 2>&1 | sed "s/^/[forkbase up] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_up failed"; exit 5; }
log "at_setup (uv sync, cu124 pin, device check)"
run_on_box "bash cloud/at_setup.sh" 2>&1 | sed "s/^/[forkbase setup] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: at_setup failed"; exit 6; }

# ── 6. SELECT the forking item ON the box for THIS model ──
log "select forking item ($MODEL)"
run_on_box ".venv/bin/python sesgo/forking/select_forking_item.py --model $MODEL \
  --sesgo-dir $SELECT_SESGO_DIR --categories $SELECT_CATEGORIES --languages $SELECT_LANGUAGES \
  --n-pilot $SELECT_N_PILOT --max-items $SELECT_MAX_ITEMS" 2>&1 | sed "s/^/[forkbase select] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: select failed"; exit 7; }

# ── 7. DECODE the base path (greedy CoT + branch-prefix enumeration) ──
log "decode base path ($MODEL): base-max-new-tokens=$BASE_MAX_NEW_TOKENS n-samples=$N_SAMPLES near-window=$NEAR_WINDOW"
run_on_box ".venv/bin/python sesgo/forking/decode_forking_base_path.py --model $MODEL \
  --base-max-new-tokens $BASE_MAX_NEW_TOKENS --n-samples $N_SAMPLES --near-window $NEAR_WINDOW" 2>&1 | sed "s/^/[forkbase decode] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: decode failed"; exit 8; }

# ── 8. Verify base_path.json is non-empty BEFORE we destroy ──
NPOS="$(run_on_box ".venv/bin/python -c \"import json; print(len(json.load(open('out/sesgo/forking/$BARE_MODEL/base_path.json'))['base_token_ids']))\" 2>/dev/null || echo 0")"
NPOS="$(printf '%s' "$NPOS" | tr -dc '0-9')"
log "remote base_path.json has ${NPOS:-0} base positions"
[ "${NPOS:-0}" -ge 1 ] || { log "FATAL: empty/missing base_path.json -- aborting"; exit 9; }

# ── 9. SYNC selected_item.json + base_path.json BACK into sync/forkbase/ ──
log "sync_back -> sync/forkbase/sesgo/forking/"
SYNC_SUBDIR="forkbase" STUDIES="forking" INSTANCE="$INSTANCE" bash "$HERE/sync_back.sh" 2>&1 | sed "s/^/[forkbase back] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_back failed"; exit 10; }

log "SUCCESS: base path decoded for $BARE_MODEL ($NPOS positions); quarantined under sync/forkbase/"
exit 0
