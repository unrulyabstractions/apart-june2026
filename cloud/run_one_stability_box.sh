#!/usr/bin/env bash
#
# run_one_stability_box.sh — full lifecycle for ONE stability box, end to end.
#
# Stands up exactly ONE Vast.ai GPU box, runs the SESGO STABILITY collection for a
# SINGLE model (matching the existing Qwen3-0.6B reference: ITEMS=12 -> 12 distinct
# items x 36 surface variants = 432 samples), renders the per-model stability
# plots ON the box, syncs results back into a DISJOINT local quarantine
# (sync/<TAG>/...), and then DESTROYS the box. Destroy is wired to an EXIT trap so
# the box is torn down on success, failure, OR interruption — it can never be left
# billing.
#
# Reuses the existing cloud/ building blocks (vast_launch search/create logic,
# _ssh_target.sh, at_setup.sh, fleet_model_run.sh, the viz driver). Non-interactive.
#
# Required env:
#   MODEL        HF model name (e.g. Qwen/Qwen3-32B)
#   BARE_MODEL   bare name for paths (e.g. Qwen3-32B)
#   GPU_NAME     Vast gpu_name filter (e.g. H100_SXM, RTX_4090)
#   NUM_GPUS     gpus per box
#   MAX_PRICE    $/hr ceiling
#   DISK         GB disk
#   TAG          quarantine subdir under sync/ (e.g. q32)
# Optional:
#   ITEMS        distinct items to keep (default 12 -> 432 samples, matches ref)
#   HF_TOKEN     for gated models (Llama)
#   MIN_RELIABILITY  reliability2 floor (default 0.985)
#
# Writes a run log to cloud/.stab_<TAG>.log and the instance id to
# cloud/.stab_<TAG>.iid (so a watcher can find/destroy it out-of-band if needed).

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
ITEMS="${ITEMS:-12}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.985}"

IID_FILE="$HERE/.stab_${TAG}.iid"
INSTANCE=""

log() { echo "[$TAG $(date +%H:%M:%S)] $*"; }

# ── DESTROY-ON-EXIT: tear the box down no matter how we leave this script ──
destroy_box() {
  local rc=$?
  if [ -n "$INSTANCE" ]; then
    log "DESTROY instance $INSTANCE (exit rc=$rc)"
    # vast_destroy honors INSTANCE from the env and auto-answers the inner [y/N].
    INSTANCE="$INSTANCE" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure 2>&1 | sed "s/^/[$TAG destroy] /"
    # Verify teardown by walking the paginated instances-v1 listing (v0 = HTTP 410).
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
# NOTE: the vastai CLI rejects 'reliability2' as a QUERY filter field (warns and
# drops it), so we enforce MIN_RELIABILITY in Python on the returned reliability2.
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

# ── 3. POLL until running ──
# instances-v1 --raw returns a PAGINATED dict ({instances:[...], next_token}), and
# our box may be on a later page, so walk every page (subprocess per page) to find
# this instance's status. Returns the status string or 'missing'.
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
log "sync_up"
INSTANCE="$INSTANCE" bash "$HERE/sync_up.sh" 2>&1 | sed "s/^/[$TAG up] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_up failed"; exit 5; }

# ── 6. uv sync + cu124 torch pin + device check (reuse at_setup.sh) ──
log "at_setup (uv sync, cu124 pin, device check)"
run_on_box "bash cloud/at_setup.sh" 2>&1 | sed "s/^/[$TAG setup] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: at_setup failed"; exit 6; }

# ── 7. RUN the stability collection (canonical on-box driver) ──
log "collect stability (ITEMS=$ITEMS) for $MODEL"
run_on_box "export HF_TOKEN='${HF_TOKEN:-}'; MODEL='$MODEL' STUDIES='stability' ITEMS='$ITEMS' bash cloud/fleet_model_run.sh" \
  2>&1 | sed "s/^/[$TAG run] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: stability collection failed"; exit 7; }

# ── 8. Verify the result is non-empty BEFORE we destroy the box ──
NS="$(run_on_box "[ -x .venv/bin/python ] && .venv/bin/python -c \"import json; print(len(json.load(open('out/sesgo/stability/$BARE_MODEL/response_samples.json'))['samples']))\" 2>/dev/null || echo 0")"
NS="$(printf '%s' "$NS" | tr -dc '0-9')"
log "remote response_samples.json has ${NS:-0} samples"
if [ "${NS:-0}" -lt 1 ]; then
  log "FATAL: empty/missing response_samples.json on box -- aborting (no sync, will destroy)"
  exit 8
fi

# ── 9. Render the per-model stability plots ON the box ──
log "render stability plots"
run_on_box ".venv/bin/python sesgo/stability/visualize_stability_samples.py out/sesgo/stability/$BARE_MODEL/response_samples.json" \
  2>&1 | sed "s/^/[$TAG viz] /"
# Plots are a nice-to-have; do not abort the whole run if the renderer hiccups.
[ "${PIPESTATUS[0]}" -eq 0 ] || log "WARN: plot render returned non-zero (continuing to sync samples)"

# ── 10. SYNC results BACK into a DISJOINT quarantine sync/<TAG>/ ──
log "sync_back -> sync/$TAG/"
SYNC_SUBDIR="$TAG" STUDIES="stability" INSTANCE="$INSTANCE" bash "$HERE/sync_back.sh" 2>&1 | sed "s/^/[$TAG back] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_back failed (box will still be destroyed)"; exit 9; }

log "SUCCESS: $NS samples collected for $BARE_MODEL on $GPU_NAME; quarantined under sync/$TAG/"
# EXIT trap destroys the box now.
exit 0
