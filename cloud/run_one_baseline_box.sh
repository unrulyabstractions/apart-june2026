#!/usr/bin/env bash
#
# run_one_selection_box.sh — full lifecycle for ONE selection box, end to end.
#
# Stands up exactly ONE Vast.ai GPU box, runs the SESGO SELECTION collection for a
# SINGLE model over the FULL selection grid (all 5 scaffold conditions — the
# no-scaffold baseline plus the four debiasing scaffolds — across both context
# conditions, teacher-forced non-thinking readouts plus sampled thinking draws),
# renders the per-model selection plots ON the box, syncs results back into a
# DISJOINT local quarantine (sync/<TAG>/...), and then DESTROYS the box. Destroy is
# wired to an EXIT trap so the box is torn down on success, failure, OR
# interruption — it can never be left billing.
#
# Reuses the existing cloud/ building blocks (vast_launch search/create logic,
# _ssh_target.sh, at_setup.sh, fleet_model_run.sh, the viz driver). Non-interactive.
# This is the selection twin of run_one_stability_box.sh.
#
# Required env:
#   MODEL        HF model name (e.g. Qwen/Qwen3-32B)
#   BARE_MODEL   bare name for paths (e.g. Qwen3-32B)
#   GPU_NAME     Vast gpu_name filter (e.g. H100_SXM, RTX_4090)
#   NUM_GPUS     gpus per box
#   MAX_PRICE    $/hr ceiling
#   DISK         GB disk
#   TAG          quarantine subdir under sync/ (e.g. sel32)
# Optional:
#   N_THINKING       sampled thinking draws per prompt (default 3, matches 0.6B ref)
#   MAX_NEW_TOKENS   max new tokens per thinking draw (default 256, matches ref)
#   BATCH_SIZE       prompts per forward pass on the box (default 32)
#   HF_TOKEN         for gated models
#   MIN_RELIABILITY  reliability2 floor (default 0.985)
#
# Writes a run log to cloud/.base_<TAG>.log and the instance id to
# cloud/.base_<TAG>.iid (so a watcher can find/destroy it out-of-band if needed).

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="/root/apart"

: "${MODEL:?set MODEL}"
: "${BARE_MODEL:?set BARE_MODEL}"
: "${GPU_NAME:?set GPU_NAME}"
: "${NUM_GPUS:?set NUM_GPUS}"
: "${MAX_PRICE:?set MAX_PRICE}"
: "${DISK:?set DISK}"
: "${TAG:?set TAG}"
N_THINKING="${N_THINKING:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.985}"
# OOM guards for a 32B model on one 80GB card: the selection grid mixes short and
# LONG (scaffolded) prompts, and a wide teacher-forced/generation batch of long
# prompts spiked >80GB near the end of a prior run (CUDA OOM, rc=7). These cap
# sequences per forward/generation pass and reduce allocator fragmentation, without
# changing the math (same readouts, just smaller chunks). Unset == no cap.
HF_FORWARD_MICRO_BATCH="${HF_FORWARD_MICRO_BATCH:-}"
HF_GEN_MICRO_BATCH="${HF_GEN_MICRO_BATCH:-}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# SUBSAMPLE: fraction (0-1) of the selection grid to query. UNSET == full grid
# (backward-compatible no-op). Set to match a reference run's slice; the on-box
# fleet_model_run.sh forwards it to collect_selection_samples.py --subsample.
SUBSAMPLE="${SUBSAMPLE:-}"

IID_FILE="$HERE/.base_${TAG}.iid"
INSTANCE=""

log() { echo "[$TAG $(date +%H:%M:%S)] $*"; }

# ── DESTROY-ON-EXIT: tear the box down no matter how we leave this script ──
destroy_box() {
  local rc=$?
  if [ -n "$INSTANCE" ]; then
    log "DESTROY instance $INSTANCE (exit rc=$rc)"
    INSTANCE="$INSTANCE" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure 2>&1 | sed "s/^/[$TAG destroy] /"
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
      printf 'y\n' | vastai destroy instance "$INSTANCE" 2>&1 | sed "s/^/[$TAG destroy2] /"
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
      -o ServerAliveInterval=15 -o ServerAliveCountMax=40 \
      -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" \
      "cd $REMOTE_ROOT 2>/dev/null || true; $cmd"
}

# ── 1. SEARCH for a matching verified offer (reliability floor enforced) ──
# Two prior boxes were lost to host preemption/reset on lower-reliability, low-net
# offers, so auto-selection ranks by reliability2 DESC (then net DESC), not price —
# stability over a few cents. An explicit OFFER_ID env var pins a hand-picked offer.
QUERY="gpu_name=${GPU_NAME} num_gpus>=${NUM_GPUS} verified=true rentable=true direct_port_count>=1 disk_space>=${DISK} dph_total<=${MAX_PRICE}"
log "search offers: $QUERY (rel2 floor $MIN_RELIABILITY, ranked by reliability then net)"
OFFERS_JSON="$(vastai search offers "$QUERY" -o 'dph_total+' --raw 2>/dev/null)"
OFFER_ID="${OFFER_ID:-}"
if [ -z "$OFFER_ID" ]; then
  OFFER_ID="$(printf '%s' "$OFFERS_JSON" | MIN_REL="$MIN_RELIABILITY" python3 -c '
import sys, json, os
floor=float(os.environ["MIN_REL"])
o=json.load(sys.stdin)
# Prefer the most reliable, well-connected, single-80GB-card offer above the floor.
flt=[r for r in o if r.get("reliability2",0)>=floor]
flt.sort(key=lambda r:(-r.get("reliability2",0), -(r.get("inet_down") or 0)))
print(flt[0]["id"] if flt else "")')"
fi
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

# ── 5. SYNC code + SESGO prompt sources UP (reuse sync_up.sh) ──
# Vast SSH endpoints flap intermittently while a box is freshly booted, and a
# single transient resolve/connection failure used to abort an otherwise-healthy
# box. So retry sync_up and at_setup a few times before giving up.
log "sync_up"
up_ok=""
for try in 1 2 3 4 5; do
  INSTANCE="$INSTANCE" bash "$HERE/sync_up.sh" 2>&1 | sed "s/^/[$TAG up] /"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then up_ok=1; break; fi
  log "sync_up attempt $try failed; retrying in 20s"; sleep 20
done
[ -n "$up_ok" ] || { log "FATAL: sync_up failed after retries"; exit 5; }

# ── 6. uv sync + cu124 torch pin + device check (reuse at_setup.sh) ──
log "at_setup (uv sync, cu124 pin, device check)"
setup_ok=""
for try in 1 2 3 4 5; do
  run_on_box "bash cloud/at_setup.sh" 2>&1 | sed "s/^/[$TAG setup] /"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then setup_ok=1; break; fi
  log "at_setup attempt $try failed (transient SSH?); retrying in 25s"; sleep 25
done
[ -n "$setup_ok" ] || { log "FATAL: at_setup failed after retries"; exit 6; }

# ── 7. RUN the selection collection DETACHED (canonical on-box driver, FULL grid) ──
# CRITICAL ROBUSTNESS FIX: prior runs lost otherwise-complete results because the
# single long-lived SSH that ran the collection dropped mid/post-run ("Connection
# closed by remote host"), aborting the launcher before it could sync. So we launch
# the collection under nohup/setsid (survives SSH drops), then POLL for completion
# over fresh short-lived connections. A dropped poll just retries; the run keeps
# going on the box and its periodic checkpoint is crash-safe regardless.
log "collect selection DETACHED (n_thinking=$N_THINKING, max_new_tokens=$MAX_NEW_TOKENS, batch=$BATCH_SIZE, fwd_mb=${HF_FORWARD_MICRO_BATCH:-off}, gen_mb=${HF_GEN_MICRO_BATCH:-off}, alloc=${PYTORCH_CUDA_ALLOC_CONF:-default}) for $MODEL"
RUN_ENV="export HF_TOKEN='${HF_TOKEN:-}' HF_FORWARD_MICRO_BATCH='${HF_FORWARD_MICRO_BATCH:-}' HF_GEN_MICRO_BATCH='${HF_GEN_MICRO_BATCH:-}' PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF:-}' MODEL='$MODEL' STUDIES='baseline' N_THINKING='$N_THINKING' MAX_NEW_TOKENS='$MAX_NEW_TOKENS' BATCH_SIZE='$BATCH_SIZE' SUBSAMPLE='$SUBSAMPLE'"
# Start detached; write a done-marker ($?-stamped) the poller watches for.
run_on_box "cd $REMOTE_ROOT; rm -f base_run.done base_run.log; setsid bash -c \"$RUN_ENV; bash cloud/fleet_model_run.sh > base_run.log 2>&1; echo \\\$? > base_run.done\" >/dev/null 2>&1 & echo started" \
  2>&1 | sed "s/^/[$TAG run] /"

RESULT_JSON="out/sesgo/baseline/$BARE_MODEL/response_samples.json"
# Poll until the done-marker appears (run finished) or we time out. Each poll is a
# fresh connection, so transient SSH drops are harmless. Up to ~75 min.
RC=""
for i in $(seq 1 300); do
  sleep 15
  DONE="$(run_on_box "cat $REMOTE_ROOT/base_run.done 2>/dev/null" 2>/dev/null | tr -dc '0-9')"
  if [ -n "$DONE" ]; then RC="$DONE"; break; fi
  # progress heartbeat every ~2.5 min
  if [ $((i % 10)) -eq 0 ]; then
    NSP="$(run_on_box ".venv/bin/python -c \"import json;print(len(json.load(open('$RESULT_JSON'))['samples']))\" 2>/dev/null" 2>/dev/null | tr -dc '0-9')"
    log "  [poll $i] on-box samples=${NSP:-?} (waiting for done-marker)"
  fi
done
if [ -z "$RC" ]; then log "FATAL: collection did not finish within poll window"; exit 7; fi
if [ "$RC" != "0" ]; then
  log "FATAL: selection collection exited rc=$RC; tail of on-box log:"
  run_on_box "tail -20 $REMOTE_ROOT/base_run.log" 2>&1 | sed "s/^/[$TAG runlog] /"
  exit 7
fi
log "collection finished cleanly (rc=0)"

# ── 8. Verify the result is non-empty BEFORE we destroy the box ──
NS="$(run_on_box "[ -x .venv/bin/python ] && .venv/bin/python -c \"import json; print(len(json.load(open('out/sesgo/baseline/$BARE_MODEL/response_samples.json'))['samples']))\" 2>/dev/null || echo 0")"
NS="$(printf '%s' "$NS" | tr -dc '0-9')"
log "remote response_samples.json has ${NS:-0} samples"
if [ "${NS:-0}" -lt 1 ]; then
  log "FATAL: empty/missing response_samples.json on box -- aborting (no sync, will destroy)"
  exit 8
fi

# ── 9. Render the per-model selection plots ON the box ──
log "render selection plots"
run_on_box ".venv/bin/python sesgo/baseline/visualize_baseline_samples.py out/sesgo/baseline/$BARE_MODEL/response_samples.json" \
  2>&1 | sed "s/^/[$TAG viz] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || log "WARN: plot render returned non-zero (continuing to sync samples)"

# ── 10. SYNC results BACK into a DISJOINT quarantine sync/<TAG>/ ──
log "sync_back -> sync/$TAG/"
SYNC_SUBDIR="$TAG" STUDIES="baseline" INSTANCE="$INSTANCE" bash "$HERE/sync_back.sh" 2>&1 | sed "s/^/[$TAG back] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_back failed (box will still be destroyed)"; exit 9; }

log "SUCCESS: $NS samples collected for $BARE_MODEL on $GPU_NAME; quarantined under sync/$TAG/"
# EXIT trap destroys the box now.
exit 0
