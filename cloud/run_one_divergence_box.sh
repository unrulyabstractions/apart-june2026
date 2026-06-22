#!/usr/bin/env bash
#
# run_one_divergence_box.sh — full lifecycle for ONE divergence box, end to end.
#
# Stands up EXACTLY ONE Vast.ai GPU box, runs the re-scoped DIVERGENCE study for
# ONE model (the collector picks ONE representative ambiguous SESGO prompt and
# samples its THINKING generation --n-thinking times, recording each draw's mean
# next-token vocab entropy), renders the divergence figures ON the box, syncs the
# results back into a DISJOINT local quarantine (sync/div/...), and DESTROYS the
# box. Destroy is wired to an EXIT trap so the box is torn down on success,
# failure, OR interruption — it can never be left billing.
#
# Reuses the existing cloud/ building blocks (vast_launch search/create logic,
# _ssh_target.sh, sync_up.sh, at_setup.sh, sync_back.sh) and the .venv/bin/python
# interpreter on the box (NEVER `uv run` — see at_setup.sh for why). Non-interactive.
#
# It tracks its instance in a DEDICATED file (cloud/.div_<bare>.iid) so it never
# resolves/destroys the separate forking box (cloud/.fork.iid) or any other box.
#
# Required env: MODEL  (HF model name, e.g. Qwen/Qwen3-0.6B).
# Optional:
#   GPU_NAME        Vast gpu_name filter (default RTX_4090)
#   MIN_GPU_RAM     min per-gpu VRAM in MB (default 0 == no floor)
#   NUM_GPUS        gpus per box (default 1)
#   MAX_PRICE       $/hr ceiling (default 0.80)
#   DISK            GB disk (default 60)
#   MIN_RELIABILITY reliability2 floor (default 0.985)
#   N_THINKING TEMPERATURE MAX_NEW_TOKENS   (run knobs; defaults 100 / 1.0 / 512)

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="/root/apart"

MODEL="${MODEL:?set MODEL to an HF model name, e.g. Qwen/Qwen3-0.6B}"
BARE_MODEL="${MODEL##*/}"
GPU_NAME="${GPU_NAME:-RTX_4090}"
MIN_GPU_RAM="${MIN_GPU_RAM:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_PRICE="${MAX_PRICE:-0.80}"
DISK="${DISK:-60}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.985}"

# DIVERGENCE knobs (task spec): 100 deep CoT draws of ONE prompt at temperature 1.0.
N_THINKING="${N_THINKING:-100}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

IID_FILE="$HERE/.div_${BARE_MODEL}.iid"
INSTANCE=""

log() { echo "[div:$BARE_MODEL $(date +%H:%M:%S)] $*"; }

# ── DESTROY-ON-EXIT: tear the box down no matter how we leave this script ──
destroy_box() {
  local rc=$?
  if [ -n "$INSTANCE" ]; then
    log "DESTROY instance $INSTANCE (exit rc=$rc)"
    INSTANCE="$INSTANCE" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure 2>&1 | sed "s/^/[div destroy] /"
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
      printf 'y\n' | vastai destroy instance "$INSTANCE" 2>&1 | sed "s/^/[div destroy2] /"
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
      -o ServerAliveInterval=15 -o ServerAliveCountMax=240 \
      -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" \
      "cd $REMOTE_ROOT 2>/dev/null || true; $cmd"
}

# ── 1. SEARCH for a matching verified offer (reliability + VRAM floors enforced) ──
QUERY="gpu_name=${GPU_NAME} num_gpus>=${NUM_GPUS} verified=true rentable=true direct_port_count>=1 disk_space>=${DISK} dph_total<=${MAX_PRICE}"
log "search offers: $QUERY (rel2>=$MIN_RELIABILITY, gpu_ram>=${MIN_GPU_RAM}MB enforced post-hoc)"
OFFERS_JSON="$(vastai search offers "$QUERY" -o 'dph_total+' --raw 2>/dev/null)"
OFFER_ID="$(printf '%s' "$OFFERS_JSON" | MIN_REL="$MIN_RELIABILITY" MIN_RAM="$MIN_GPU_RAM" python3 -c '
import sys, json, os
floor=float(os.environ["MIN_REL"]); ram=float(os.environ["MIN_RAM"])
o=json.load(sys.stdin)
def gpuram(r): return r.get("gpu_total_ram", r.get("gpu_ram", 0)) or 0
flt=[r for r in o if r.get("reliability2",0)>=floor and gpuram(r)>=ram]
print(flt[0]["id"] if flt else "")')"
if [ -z "$OFFER_ID" ]; then
  log "FATAL: no matching offers for $GPU_NAME (reliability2>=$MIN_RELIABILITY, gpu_ram>=${MIN_GPU_RAM}MB, <=\$$MAX_PRICE/hr)"
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
for i in $(seq 1 120); do
  ST="$(instance_status)"
  log "  [$i] status: $ST"
  [ "$ST" = "running" ] && break
  sleep 15
done
[ "$ST" = "running" ] || { log "FATAL: instance never reached running"; exit 4; }

# ── 4. Wait for sshd to actually accept connections ──
# HARD GATE: provisioning sshd + ssh-key propagation flakes (connection refused /
# "Permission denied (publickey)") for the first minutes. We must NOT fall through
# to sync_up until a real command round-trips, or rsync dies and we waste the box.
# Require ssh_ok before proceeding; abort (→ destroy) if it never comes up.
log "waiting for sshd (hard gate)"
SSH_OK=0
for i in $(seq 1 60); do
  if run_on_box "echo ssh_ok" 2>/dev/null | grep -q ssh_ok; then log "sshd up (attempt $i)"; SSH_OK=1; break; fi
  sleep 10
done
[ "$SSH_OK" -eq 1 ] || { log "FATAL: sshd/ssh-key never came up after 60 tries; aborting (will destroy)"; exit 4; }

# ── 5. SYNC code UP (reuse sync_up.sh) ──
log "sync_up (code + SESGO prompts)"
INSTANCE="$INSTANCE" bash "$HERE/sync_up.sh" 2>&1 | sed "s/^/[div up] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_up failed"; exit 5; }

# ── 5b. Push the EXISTING divergence prompt_dataset.json (under out/, excluded) ──
# The collector reads out/sesgo/divergence/prompt_dataset.json and picks ONE
# representative ambiguous prompt from it. Pushing the SAME local dataset (rather
# than regenerating on the box) guarantees both boxes select the IDENTICAL prompt.
log "push divergence prompt_dataset.json (the fixed prompt grid)"
. "$HERE/_ssh_target.sh"; _resolve_ssh_target || { log "FATAL: ssh endpoint"; exit 5; }
RSYNC_E="ssh -F /dev/null -o StrictHostKeyChecking=accept-new -i $SSH_KEY -p $SSH_PORT"
ssh -F /dev/null -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -p "$SSH_PORT" \
  "$SSH_USER@$SSH_HOST" "mkdir -p $REMOTE_ROOT/out/sesgo/divergence"
rsync -ah -e "$RSYNC_E" \
  "$REPO_ROOT/out/sesgo/divergence/prompt_dataset.json" \
  "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/out/sesgo/divergence/prompt_dataset.json"
[ $? -eq 0 ] || { log "FATAL: prompt_dataset.json push failed"; exit 5; }

# ── 6. uv sync + cu124 torch pin + device check (reuse at_setup.sh) ──
log "at_setup (uv sync, cu124 pin, device check)"
run_on_box "bash cloud/at_setup.sh" 2>&1 | sed "s/^/[div setup] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: at_setup failed"; exit 6; }

# ── 7. RUN the DIVERGENCE collector DIRECTLY (one prompt, deep CoT sampling) ──
log "collect divergence samples ($MODEL): n-items=1 n-thinking=$N_THINKING T=$TEMPERATURE max-new=$MAX_NEW_TOKENS"
run_on_box ".venv/bin/python sesgo/divergence/collect_divergence_samples.py --model $MODEL \
  --n-items 1 --n-thinking $N_THINKING --temperature $TEMPERATURE --max-new-tokens $MAX_NEW_TOKENS" 2>&1 | sed "s/^/[div collect] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: collect failed"; exit 7; }

# ── 8. Verify the output BEFORE we destroy (non-empty, has per-draw entropy) ──
NDRAW="$(run_on_box ".venv/bin/python -c \"import json; d=json.load(open('out/sesgo/divergence/$BARE_MODEL/response_samples.json')); s=d['samples'][0]; print(len(s['thinking']['vocab_entropies']))\" 2>/dev/null || echo 0")"
NDRAW="$(printf '%s' "$NDRAW" | tr -dc '0-9')"
log "remote response_samples.json: ${NDRAW:-0} per-draw vocab entropies on item 0"
if [ "${NDRAW:-0}" -lt 1 ]; then
  log "FATAL: empty/missing response_samples.json on box -- aborting (no sync, will destroy)"
  exit 8
fi

# ── 9. Render the divergence figures ON the box ──
log "render divergence figures"
run_on_box ".venv/bin/python sesgo/divergence/visualize_divergence_samples.py \
  out/sesgo/divergence/$BARE_MODEL/response_samples.json" 2>&1 | sed "s/^/[div viz] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || log "WARN: visualize returned non-zero (continuing to sync)"

# ── 10. Print the report facts FROM THE BOX (before teardown) ──
log "extracting report facts from the box"
run_on_box ".venv/bin/python - <<PYEOF
import json
d=json.load(open('out/sesgo/divergence/$BARE_MODEL/response_samples.json'))
s=d['samples'][0]; th=s['thinking']
ve=th['vocab_entropies']; n=len(ve)
mean=sum(ve)/n if n else 0.0
std=(sum((x-mean)**2 for x in ve)/n)**0.5 if n else 0.0
print('REPORT_QID=%s' % s['question_id'])
print('REPORT_SAMPLE_IDX=%s' % s['sample_idx'])
print('REPORT_BIAS_CATEGORY=%s' % s.get('bias_category'))
print('REPORT_CONTEXT=%s' % s.get('context_condition'))
print('REPORT_N_DRAWS_TOTAL=%d' % n)
print('REPORT_N_DRAWS_PARSED=%d' % th['sample_size'])
print('REPORT_OUTCOME_DIST_TOU=%s' % [round(x,4) for x in th['mean']])
print('REPORT_VOCAB_ENTROPY_MEAN=%.4f' % mean)
print('REPORT_VOCAB_ENTROPY_STD=%.4f' % std)
PYEOF" 2>&1 | sed "s/^/[div report] /"

# ── 11. SYNC results BACK into a DISJOINT quarantine sync/div/ ──
log "sync_back -> sync/div/sesgo/divergence/"
SYNC_SUBDIR="div" STUDIES="divergence" INSTANCE="$INSTANCE" bash "$HERE/sync_back.sh" 2>&1 | sed "s/^/[div back] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_back failed (box will still be destroyed)"; exit 10; }

log "SUCCESS: divergence captured for $BARE_MODEL (${NDRAW} draws); quarantined under sync/div/"
# EXIT trap destroys the box now.
exit 0
