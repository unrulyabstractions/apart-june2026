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
#   MODES        space-separated readout modes on this box (default "nonthinking";
#                Qwen wants "nonthinking thinking"; reasoning ckpts want "thinking")
#   SHARD_INDEX  this box's shard index (default 0)
#   SHARD_COUNT  total shards for this model (default 1; >1 shards the FULL dataset)
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
MODES="${MODES:-nonthinking}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
LIMIT="${LIMIT:-}"   # cap prompts (validation / throughput sizing); empty = FULL dataset
MAX_REASONING="${MAX_REASONING:-512}"  # thinking-token cap before force-closing the CoT;
                                       # keep low so small reasoners don't spin to the cap
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

# Status of $INSTANCE by walking the paginated instances-v1 listing (the box may be on a
# later page). Returns the status string or 'missing'.
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

# ── SECURE A GOOD HOST (steps 1-6 in a FRESH-HOST retry loop) ──
# A box must NEVER die because of a bad host. If provisioning OR setup fails on the chosen
# offer (never reaches running / no sshd / sync_up exhausts retries / at_setup exhausts
# retries = a persistently bad network or corrupt host), we DESTROY that instance and try a
# DIFFERENT offer — only the readout proceeds once a host is fully set up. Idempotent steps
# (sync_up/at_setup) still retry in-place first; only a host that fails ALL retries is abandoned.
MAX_HOST_TRIES="${MAX_HOST_TRIES:-8}"
host_ok=0
for host_try in $(seq 1 "$MAX_HOST_TRIES"); do
  log "===== host attempt $host_try/$MAX_HOST_TRIES ====="
  # Tear down any instance left over from a previous (bad) attempt before trying a fresh one.
  if [ -n "$INSTANCE" ]; then
    log "abandoning bad-host instance $INSTANCE"
    INSTANCE="$INSTANCE" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure >/dev/null 2>&1
    INSTANCE=""; rm -f "$IID_FILE"
  fi

  # 1. search a verified offer above the reliability floor
  OFFERS_JSON="$(vastai search offers "$QUERY" -o 'dph_total+' --raw 2>/dev/null)"
  OFFER_ID="$(printf '%s' "$OFFERS_JSON" | MIN_REL="$MIN_RELIABILITY" TRY="$host_try" python3 -c '
import sys, json, os
floor=float(os.environ["MIN_REL"])
flt=[r for r in json.load(sys.stdin) if r.get("reliability2",0)>=floor]
# walk down the price-sorted list across attempts so a bad host is not re-picked
print(flt[(int(os.environ["TRY"])-1) % len(flt)]["id"] if flt else "")')"
  [ -z "$OFFER_ID" ] && { log "no matching offers; wait 30s + retry"; sleep 30; continue; }
  printf '%s' "$OFFERS_JSON" | OFFER_ID="$OFFER_ID" python3 -c '
import sys, json, os
oid=int(os.environ["OFFER_ID"])
o=next(r for r in json.load(sys.stdin) if r["id"]==oid)
print("[offer] id=%s %sx %s vram=%sMB $%.3f/hr reliability2=%.4f net=%sMbps" % (
    o.get("id"), o.get("num_gpus"), o.get("gpu_name"), o.get("gpu_total_ram", o.get("gpu_ram",0)),
    o.get("dph_total",0.0), o.get("reliability2",0.0), o.get("inet_down")))' | sed "s/^/[$TAG] /"

  # 2. create the instance
  PORTAL_ENV='-p 1111:1111 -e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1" -e JUPYTER_DIR="/" -e DATA_DIRECTORY="/workspace/"'
  CREATE_JSON="$(vastai create instance "$OFFER_ID" \
    --image 'vastai/pytorch:@vastai-automatic-tag' \
    --env "$PORTAL_ENV" --onstart-cmd 'entrypoint.sh' \
    --disk "$DISK" --ssh --direct --raw)"
  INSTANCE="$(printf '%s' "$CREATE_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("new_contract",""))')"
  [ -z "$INSTANCE" ] && { log "create failed; retry fresh offer"; INSTANCE=""; sleep 10; continue; }
  echo "$INSTANCE" > "$IID_FILE"
  log "created instance $INSTANCE; waiting for 'running'"

  # 3. poll until running (give up on THIS host after ~12min, try a fresh one)
  ST=""; for i in $(seq 1 50); do
    ST="$(instance_status)"; log "  [$i] status: $ST"
    [ "$ST" = "running" ] && break; sleep 15
  done
  [ "$ST" = "running" ] || { log "never reached running on this host; trying a fresh one"; continue; }

  # 4. wait for sshd
  sshd_ok=0
  for i in $(seq 1 40); do
    run_on_box "echo ssh_ok" 2>/dev/null | grep -q ssh_ok && { sshd_ok=1; log "sshd up"; break; }
    sleep 10
  done
  [ "$sshd_ok" = 1 ] || { log "sshd never came up; trying a fresh host"; continue; }

  # 5. sync_up (retried in-place; a host that fails ALL retries is BAD -> fresh host)
  sync_up_ok=0
  for attempt in 1 2 3 4 5; do
    log "sync_up (attempt $attempt/5)"
    INSTANCE="$INSTANCE" bash "$HERE/sync_up.sh" 2>&1 | sed "s/^/[$TAG up] /"
    [ "${PIPESTATUS[0]}" -eq 0 ] && { sync_up_ok=1; break; }
    log "sync_up attempt $attempt FAILED; waiting 25s"; sleep 25
  done
  [ "$sync_up_ok" = 1 ] || { log "sync_up exhausted retries -> BAD HOST, trying a fresh one"; continue; }

  # 6. at_setup (retried in-place; failing ALL retries = corrupt/bad host -> fresh host)
  setup_ok=0
  for attempt in 1 2 3 4; do
    log "at_setup (attempt $attempt/4)"
    run_on_box "INSTALL_KERNELS='${INSTALL_KERNELS:-0}' KERNELS_VERSION='${KERNELS_VERSION:-0.12.3}' TORCH_PKGS='${TORCH_PKGS:-}' TORCH_INDEX='${TORCH_INDEX:-}' bash cloud/at_setup.sh" 2>&1 | sed "s/^/[$TAG setup] /"
    [ "${PIPESTATUS[0]}" -eq 0 ] && { setup_ok=1; break; }
    log "at_setup attempt $attempt FAILED; waiting 30s"; sleep 30
  done
  [ "$setup_ok" = 1 ] || { log "at_setup exhausted retries -> BAD HOST, trying a fresh one"; continue; }

  host_ok=1; break   # fully provisioned + set up on a good host
done
[ "$host_ok" = 1 ] || { log "FATAL: could not secure a good host after $MAX_HOST_TRIES tries"; exit 6; }

# ── 6.5 RESUME FROM A SAVED PARTIAL (keep progress when sharding/restarting) ──
# If RESUME_FROM points at a local dir holding a previous partial (out/<study>/<bare>-<mode>/
# response_samples.json), push it up to THIS box's matching slice paths (incl. the shard
# subdir). The resumable readout then skips every prompt_id already done and only runs the
# remainder — so re-launching as shards never re-does finished work.
RESUME_SUF=""; [ "${SHARD_COUNT:-1}" -gt 1 ] && RESUME_SUF="/shard_${SHARD_INDEX}_of_${SHARD_COUNT}"
if [ -n "${RESUME_FROM:-}" ] && [ -d "$RESUME_FROM" ]; then
  . "$HERE/_ssh_target.sh"; INSTANCE="$INSTANCE" _resolve_ssh_target 2>/dev/null
  for MODE in $MODES; do
    for STUDY in stability forked; do
      src="$RESUME_FROM/$STUDY/${BARE_MODEL}-${MODE}/response_samples.json"
      [ -f "$src" ] || continue
      dst="/root/apart/out/$STUDY/${BARE_MODEL}-${MODE}${RESUME_SUF}"
      run_on_box "mkdir -p $dst" >/dev/null 2>&1
      rsync -a -e "ssh $SSH_EPHEMERAL_OPTS -i ${SSH_KEY} -p $SSH_PORT" \
        "$src" "$SSH_USER@$SSH_HOST:$dst/response_samples.json" 2>/dev/null \
        && log "seeded resume: $STUDY:$MODE <- $(uv run python -c "import json;print(len(json.load(open('$src'))['samples']))" 2>/dev/null) samples"
    done
  done
fi

# ── 7. RUN the readout DETACHED on the box, then POLL for completion ──
# The whole build+readout pipeline runs on the box under `setsid nohup`
# (cloud/on_box_stability_run.sh), writing a terminal marker (out/.STAB_DONE | .STAB_FAILED).
# A dropped ssh connection CANNOT interrupt it — the job runs detached and we just reconnect
# each poll. The readout is checkpoint-resumable, and the on-box script only marks DONE once
# every slice reaches its full expected sample count, so DONE means truly complete.
log "launch detached on-box run (survives ssh drops)"
launch_ok=0
for attempt in 1 2 3 4 5; do
  # Two layers so the launch can never hang the driver:
  #  (1) `mkdir -p out` FIRST — sync_up excludes out/, so without it the >out/stab_run.log
  #      redirect fails and the on-box script never starts (a silent no-op).
  #  (2) the remote detach is wrapped in a SUBSHELL `( ... & )` so the remote shell returns
  #      immediately, and the launch ssh is run in the LOCAL background (then killed) so even
  #      a sticky ssh channel can't block us. We then VERIFY via a FRESH ssh that the on-box
  #      script actually began writing — never trust a bare echo.
  ( run_on_box "cd $REMOTE_ROOT && mkdir -p out && rm -f out/.STAB_DONE out/.STAB_FAILED && ( MODEL='$MODEL' BARE_MODEL='$BARE_MODEL' MODES='$MODES' MAX_REASONING='$MAX_REASONING' SHARD_INDEX='${SHARD_INDEX:-0}' SHARD_COUNT='${SHARD_COUNT:-1}' LIMIT='${LIMIT:-}' HF_TOKEN='${HF_TOKEN:-}' setsid bash cloud/on_box_stability_run.sh >out/stab_run.log 2>&1 </dev/null & )" ) >/dev/null 2>&1 &
  lpid=$!
  sleep 12
  started="$(run_on_box "grep -q '\[on_box\]' out/stab_run.log 2>/dev/null && echo YES || echo NO" 2>/dev/null | tr -dc 'A-Z')"
  kill "$lpid" 2>/dev/null  # reap the (possibly sticky) launch ssh; the on-box job is detached
  [ "$started" = YES ] && { launch_ok=1; log "on-box run confirmed started"; break; }
  log "launch attempt $attempt: on-box run did NOT start; retrying in 10s"; sleep 10
done
[ "$launch_ok" = 1 ] || { log "FATAL: on-box run never started after 5 attempts"; exit 7; }

# Poll for the terminal marker. An ssh blip during a poll yields empty state and is simply
# ignored — the detached job keeps running and the next poll reconnects.
poll_start=$(date +%s); poll_max=$((20 * 3600))
while true; do
  sleep 60
  state="$(run_on_box "if [ -f out/.STAB_DONE ]; then echo DONE; elif [ -f out/.STAB_FAILED ]; then echo FAILED; else echo RUNNING; fi" 2>/dev/null | tr -dc 'A-Z')"
  prog="$(run_on_box "tail -n1 out/stab_run.log 2>/dev/null" 2>/dev/null | tail -n1)"
  log "poll state=${state:-blip} | ${prog}"
  [ "$state" = DONE ] && { log "on-box run DONE"; break; }
  if [ "$state" = FAILED ]; then
    run_on_box "tail -n 30 out/stab_run.log" 2>&1 | sed "s/^/[$TAG runlog] /"
    log "FATAL: on-box run reported FAILED"; exit 8
  fi
  [ $(( $(date +%s) - poll_start )) -gt $poll_max ] && { log "FATAL: on-box run exceeded ${poll_max}s"; exit 8; }
done

# ── 8. SYNC results BACK into the gitignored quarantine sync/<TAG>/ (--ignore-existing,
#    no --delete; pulls the flat out/stability/ + out/forked/ trees). ──
log "sync_back -> sync/$TAG/ (retried)"
back_ok=0
for attempt in 1 2 3 4 5; do
  SYNC_SUBDIR="$TAG" STUDIES="stability forked" INSTANCE="$INSTANCE" bash "$HERE/sync_back.sh" 2>&1 | sed "s/^/[$TAG back] /"
  [ "${PIPESTATUS[0]}" -eq 0 ] && { back_ok=1; break; }
  log "sync_back attempt $attempt FAILED; waiting 25s"; sleep 25
done
[ "$back_ok" = 1 ] || { log "FATAL: sync_back failed after 5 attempts (box will still be destroyed)"; exit 9; }

log "SUCCESS: $BARE_MODEL modes='$MODES' shard=$SHARD_INDEX/$SHARD_COUNT on $GPU_NAME; quarantined under sync/$TAG/"
# EXIT trap destroys the box now.
exit 0
