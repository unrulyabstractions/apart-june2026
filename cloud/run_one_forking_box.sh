#!/usr/bin/env bash
#
# run_one_forking_box.sh — full lifecycle for ONE forking-paths box, end to end.
#
# Stands up EXACTLY ONE Vast.ai GPU box, runs the corrected FULL-CoT forking-paths
# collection for Qwen3-0.6B (capture {O_t} over the WHOLE base CoT — max-positions=0
# — with a larger continuation budget to drive down "unparseable" rollouts), then
# analyzes + plots ON the box, syncs results back into a DISJOINT local quarantine
# (sync/fork/...), and DESTROYS the box. Destroy is wired to an EXIT trap so the box
# is torn down on success, failure, OR interruption — it can never be left billing.
#
# It reuses the existing cloud/ building blocks (vast_launch search/create logic,
# _ssh_target.sh, sync_up.sh, at_setup.sh, sync_back.sh). Non-interactive.
#
# The forking study is NOT in fleet_model_run.sh's study switch, so the four run-by-
# path drivers are invoked DIRECTLY here via the venv interpreter (.venv/bin/python,
# never `uv run` — see at_setup.sh for why). It reuses the EXISTING selected_item.json
# (pushed up as its own narrow step) so the SAME ambiguous item is captured (item
# selection is stochastic; re-running it on the box would pick a different item).
#
# Required env: none (defaults target the task's single Qwen3-0.6B box).
# Optional:
#   GPU_NAME        Vast gpu_name filter (default RTX_4090)
#   NUM_GPUS        gpus per box (default 1)
#   MAX_PRICE       $/hr ceiling (default 0.80)
#   DISK            GB disk (default 50)
#   MIN_RELIABILITY reliability2 floor (default 0.985)
#   N_PRIOR N_SAMPLES MAX_NEW_TOKENS BASE_MAX_NEW_TOKENS TEMPERATURE (run knobs)
#
# Writes a run log to cloud/.fork.log and the instance id to cloud/.fork.iid (a
# DEDICATED tracking file, so it never resolves/destroys the separate stability
# boxes tracked by cloud/.stab_*.iid).

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="/root/apart"

MODEL="Qwen/Qwen3-0.6B"
BARE_MODEL="Qwen3-0.6B"
GPU_NAME="${GPU_NAME:-RTX_4090}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_PRICE="${MAX_PRICE:-0.80}"
DISK="${DISK:-50}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.985}"

# FULL-CoT run knobs (task spec): branch EVERY base-path position (max-positions=0),
# 768-token base decode + 768-token continuations (larger budget so rollouts reach
# </think>+answer and parse), n-prior 60, n-samples 40, temperature 1.0.
N_PRIOR="${N_PRIOR:-60}"
N_SAMPLES="${N_SAMPLES:-40}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
BASE_MAX_NEW_TOKENS="${BASE_MAX_NEW_TOKENS:-768}"
TEMPERATURE="${TEMPERATURE:-1.0}"

IID_FILE="$HERE/.fork.iid"
INSTANCE=""

log() { echo "[fork $(date +%H:%M:%S)] $*"; }

# ── DESTROY-ON-EXIT: tear the box down no matter how we leave this script ──
destroy_box() {
  local rc=$?
  if [ -n "$INSTANCE" ]; then
    log "DESTROY instance $INSTANCE (exit rc=$rc)"
    INSTANCE="$INSTANCE" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure 2>&1 | sed "s/^/[fork destroy] /"
    sleep 5
    local still
    still="$(INSTANCE_ID="$INSTANCE" python3 -c '
import json, os, subprocess, sys
iid=int(os.environ["INSTANCE_ID"]); token=None
for _ in range(40):
    cmd=["vastai","show","instances-v1","--raw"]
    if token: cmd+=["--next-token",token]
    try: d=json.loads(subprocess.run(cmd,capture_output=True,text=True).stdout)
    except Exception: break
    rows=d if isinstance(d,list) else d.get("instances",d.get("results",[]))
    if any(isinstance(r,dict) and r.get("id")==iid for r in rows): print("yes"); sys.exit()
    token=d.get("next_token") if isinstance(d,dict) else None
    if not token or not rows: break
print("no")' 2>/dev/null)"
    if [ "$still" = "yes" ]; then
      log "WARN: instance $INSTANCE still listed; retrying destroy"
      printf 'y\n' | vastai destroy instance "$INSTANCE" 2>&1 | sed "s/^/[fork destroy2] /"
    else
      log "confirmed instance $INSTANCE no longer listed"
    fi
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
      -o ServerAliveInterval=15 -o ServerAliveCountMax=120 \
      -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" \
      "cd $REMOTE_ROOT 2>/dev/null || true; $cmd"
}

# ── 1. SEARCH for a matching verified offer (reliability floor enforced) ──
QUERY="gpu_name=${GPU_NAME} num_gpus>=${NUM_GPUS} verified=true rentable=true direct_port_count>=1 disk_space>=${DISK} dph_total<=${MAX_PRICE}"
log "search offers: $QUERY (rel2 floor $MIN_RELIABILITY enforced post-hoc)"
OFFERS_JSON="$(vastai search offers "$QUERY" -o 'dph_total+' --raw 2>/dev/null)"
OFFER_ID="$(printf '%s' "$OFFERS_JSON" | MIN_REL="$MIN_RELIABILITY" python3 -c '
import sys, json, os
floor=float(os.environ["MIN_REL"])
o=json.load(sys.stdin)
flt=[r for r in o if r.get("reliability2",0)>=floor]
print(flt[0]["id"] if flt else "")')"
if [ -z "$OFFER_ID" ]; then
  log "FATAL: no matching offers for $GPU_NAME (reliability2>=$MIN_RELIABILITY, <=\$$MAX_PRICE/hr)"
  exit 2
fi
printf '%s' "$OFFERS_JSON" | OFFER_ID="$OFFER_ID" python3 -c '
import sys, json, os
oid=int(os.environ["OFFER_ID"])
o=next(r for r in json.load(sys.stdin) if r["id"]==oid)
print("[offer] id=%s %sx %s vram=%sMB $%.3f/hr reliability2=%.4f disk=%sGB net=%sMbps" % (
    o.get("id"), o.get("num_gpus"), o.get("gpu_name"),
    o.get("gpu_total_ram", o.get("gpu_ram",0)), o.get("dph_total",0.0),
    o.get("reliability2",0.0), o.get("disk_space"), o.get("inet_down")))'

# ── 2. CREATE the instance (PyTorch image, SSH, direct) ──
PORTAL_ENV='-p 1111:1111 -e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1" -e JUPYTER_DIR="/" -e DATA_DIRECTORY="/workspace/"'
CREATE_JSON="$(vastai create instance "$OFFER_ID" \
  --image 'vastai/pytorch:@vastai-automatic-tag' \
  --env "$PORTAL_ENV" --onstart-cmd 'entrypoint.sh' \
  --disk "$DISK" --ssh --direct --raw)"
INSTANCE="$(printf '%s' "$CREATE_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("new_contract",""))')"
if [ -z "$INSTANCE" ]; then
  log "FATAL: create failed: $CREATE_JSON"
  exit 3
fi
echo "$INSTANCE" > "$IID_FILE"
log "created instance $INSTANCE; waiting for 'running'"

# ── 3. POLL until running (walk paginated instances-v1) ──
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
for i in $(seq 1 100); do
  ST="$(instance_status)"
  log "  [$i] status: $ST"
  [ "$ST" = "running" ] && break
  sleep 15
done
[ "$ST" = "running" ] || { log "FATAL: instance never reached running"; exit 4; }

# ── 4. Wait for sshd to actually accept connections ──
log "waiting for sshd"
for i in $(seq 1 40); do
  if run_on_box "echo ssh_ok" 2>/dev/null | grep -q ssh_ok; then log "sshd up"; break; fi
  sleep 10
done

# ── 5. SYNC code UP (reuse sync_up.sh) ──
log "sync_up (code)"
INSTANCE="$INSTANCE" bash "$HERE/sync_up.sh" 2>&1 | sed "s/^/[fork up] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_up failed"; exit 5; }

# ── 5b. Push the EXISTING selected_item.json (under out/, excluded by sync_up) ──
# The collect driver reads out/forking/<MODEL>/selected_item.json. Item
# selection is stochastic, so we reuse the locally-chosen item (idx=6, the gender
# ambiguous item already piloted) instead of re-running selection on the box.
log "push selected_item.json (the fixed forking item)"
. "$HERE/_ssh_target.sh"; _resolve_ssh_target || { log "FATAL: ssh endpoint"; exit 5; }
RSYNC_E="ssh -F /dev/null -o StrictHostKeyChecking=accept-new -i $SSH_KEY -p $SSH_PORT"
ssh -F /dev/null -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -p "$SSH_PORT" \
  "$SSH_USER@$SSH_HOST" "mkdir -p $REMOTE_ROOT/out/forking/$BARE_MODEL"
rsync -ah -e "$RSYNC_E" \
  "$REPO_ROOT/out/forking/$BARE_MODEL/selected_item.json" \
  "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/out/forking/$BARE_MODEL/selected_item.json"
[ $? -eq 0 ] || { log "FATAL: selected_item.json push failed"; exit 5; }

# ── 6. uv sync + cu124 torch pin + device check (reuse at_setup.sh) ──
log "at_setup (uv sync, cu124 pin, device check)"
run_on_box "bash cloud/at_setup.sh" 2>&1 | sed "s/^/[fork setup] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: at_setup failed"; exit 6; }

# ── 7. RUN the FULL-CoT forking pipeline DIRECTLY (4 run-by-path drivers) ──
# Pull the model on the box happens inside collect (HF from_pretrained). max-positions
# 0 == branch EVERY base-CoT token; 768-token budgets cut unparseable rollouts.
log "collect FULL-CoT forking rollouts ($MODEL): max-positions=0 base/cont=$BASE_MAX_NEW_TOKENS/$MAX_NEW_TOKENS n-prior=$N_PRIOR n-samples=$N_SAMPLES T=$TEMPERATURE"
run_on_box ".venv/bin/python experiment/forking/collect_forking_rollouts.py --model $MODEL \
  --max-positions 0 --base-max-new-tokens $BASE_MAX_NEW_TOKENS --max-new-tokens $MAX_NEW_TOKENS \
  --n-prior $N_PRIOR --n-samples $N_SAMPLES --temperature $TEMPERATURE" 2>&1 | sed "s/^/[fork collect] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: collect failed"; exit 7; }

# ── 8. Verify the trajectory is non-empty & FULL (>> 60) BEFORE we destroy ──
NPOS="$(run_on_box ".venv/bin/python -c \"import json; print(len(json.load(open('out/forking/$BARE_MODEL/forking_trajectory.json'))['positions']))\" 2>/dev/null || echo 0")"
NPOS="$(printf '%s' "$NPOS" | tr -dc '0-9')"
log "remote forking_trajectory.json has ${NPOS:-0} base positions (old capped run was 60)"
if [ "${NPOS:-0}" -lt 1 ]; then
  log "FATAL: empty/missing forking_trajectory.json on box -- aborting (no sync, will destroy)"
  exit 8
fi

log "analyze forking dynamics"
run_on_box ".venv/bin/python experiment/forking/analyze_forking_dynamics.py --model $MODEL" 2>&1 | sed "s/^/[fork analyze] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: analyze failed"; exit 9; }

log "plot commit dynamics"
run_on_box ".venv/bin/python experiment/forking/plot_forking_commit_dynamics.py" 2>&1 | sed "s/^/[fork plotcommit] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || log "WARN: plot_forking_commit_dynamics returned non-zero (continuing)"

log "plot forking dynamics"
run_on_box ".venv/bin/python experiment/forking/plot_forking_dynamics.py --model $MODEL" 2>&1 | sed "s/^/[fork plotdyn] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || log "WARN: plot_forking_dynamics returned non-zero (continuing)"

# ── 9. Print the FULL-CoT report facts FROM THE BOX (before teardown) ──
log "extracting report facts from the box"
run_on_box ".venv/bin/python - <<'PYEOF'
import json
d=json.load(open('out/forking/Qwen3-0.6B/forking_trajectory.json'))
pos=d['positions']
toks=d.get('base_token_texts',[])
base=d.get('base_path_text','')
print('REPORT_N_POSITIONS=%d' % len(pos))
print('REPORT_N_BASE_TOKENS=%d' % len(toks))
print('REPORT_TAIL_TOKENS=%r' % (''.join(toks[-40:]) if toks else base[-300:]))
print('REPORT_HAS_THINK_CLOSE=%s' % ('</think>' in base))
print('REPORT_PRIOR=%s' % [round(x,3) for x in d.get('prior_histogram',[])])
print('REPORT_FINAL=%s' % [round(x,3) for x in d.get('final_histogram',[])])
import glob, os
dumps=sorted(glob.glob('out/forking/Qwen3-0.6B/forking_positions/pos_*.json'))
tot=unp=0
for f in dumps:
    e=json.load(open(f))
    alts=e.get('alternates',e.get('rollouts',[]))
    for a in alts:
        rs=a.get('rollouts',a.get('samples',[a]))
        if isinstance(rs,dict): rs=[rs]
        for r in rs:
            lab=r.get('label', r.get('parsed_label')) if isinstance(r,dict) else None
            tot+=1
            if lab in (None,'','unparseable','UNPARSEABLE','none'): unp+=1
print('REPORT_ROLLOUT_TOTAL=%d' % tot)
print('REPORT_ROLLOUT_UNPARSEABLE=%d' % unp)
print('REPORT_UNPARSEABLE_FRAC=%.4f' % ((unp/tot) if tot else 0.0))
print('REPORT_N_DUMP_FILES=%d' % len(dumps))
PYEOF" 2>&1 | sed "s/^/[fork report] /"

# ── 10. SYNC results BACK into a DISJOINT quarantine sync/fork/ ──
log "sync_back -> sync/fork/forking/"
SYNC_SUBDIR="fork" STUDIES="forking" INSTANCE="$INSTANCE" bash "$HERE/sync_back.sh" 2>&1 | sed "s/^/[fork back] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_back failed (box will still be destroyed)"; exit 10; }

log "SUCCESS: FULL-CoT forking captured for $BARE_MODEL ($NPOS positions); quarantined under sync/fork/"
# EXIT trap destroys the box now.
exit 0
