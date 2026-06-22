#!/usr/bin/env bash
#
# fleet_jobs_run.sh — the SEAMLESS, ALWAYS-MONITORED, NOTHING-SILENTLY-FAILS fleet
# orchestrator. The canonical entrypoint for running many boxes at once.
#
# What it guarantees (the four requirements, by construction):
#  * 100+ SEAMLESS — one shared state daemon serves ALL boxes' status/SSH from a
#    cache, so per-box API load is gone; the SSH endpoint is resolved ONCE per box
#    from the cache and inherited by sync_up/at_vast (no per-call API walk). A
#    bandwidth semaphore caps only the local-heavy sync steps; the GPU collects all
#    run concurrently.
#  * ALWAYS MONITORED — the monitor is auto-started here; every job is verified the
#    instant its output lands. You cannot run a fleet that is not monitored.
#  * NO OVERRIDE/MERGE — each box's output lands in its OWN jobs/<id>/output/ (never
#    out/); promotion is the separate, verified-only manual step.
#  * NOTHING SILENTLY FAILS — every box is a JOB with a manifest; every abort path
#    calls jr_fail (loud + recorded); the monitor reconciles at the end and screams
#    about any non-terminal job.
#
# Scope: ONE study per run (STUDIES must name a single study) so each job maps to an
# exact out/ slice (sesgo/<study>/<model>) and the model-identity assert is precise.
# Running >1 study is refused LOUDLY rather than silently mislabelling jobs.
#
# Usage (after fleet_launch.sh has written .id/.job files):
#   STUDIES=forking bash cloud/fleet_jobs_run.sh
#   STUDIES=divergence BATCH_SIZE=64 BW_CONCURRENCY=12 bash cloud/fleet_jobs_run.sh
#
# Then, after it returns:
#   bash cloud/promote_verified_jobs.sh             # dry-run: what would promote + refusals
#   bash cloud/promote_verified_jobs.sh --promote   # move verified payloads into out/

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
FLEET_DIR="${FLEET_DIR:-$HERE/.fleet}"
JOBS_DIR="${JOBS_DIR:-$REPO_ROOT/jobs}"
STUDIES="${STUDIES:-forking}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_THINKING="${N_THINKING:-8}"
BW_CONCURRENCY="${BW_CONCURRENCY:-10}"   # max concurrent sync_up/sync_back (laptop uplink)
export FLEET_DIR JOBS_DIR

# shellcheck source=/dev/null
. "$HERE/job_registry.sh"
# shellcheck source=/dev/null
. "$HERE/fleet_state_lookup.sh"

# ── Guard rails (nothing silently mislabelled) ─────────────────────────────
[ -d "$FLEET_DIR" ] || { echo "FATAL: no fleet dir ($FLEET_DIR). Run cloud/fleet_launch.sh first." >&2; exit 1; }
if [ "$(echo "$STUDIES" | wc -w | tr -d ' ')" -ne 1 ]; then
  echo "FATAL: STUDIES='$STUDIES' has more than one study. This orchestrator runs ONE study per fleet so each job maps to an exact out/ slice. Run them as separate fleets." >&2
  exit 2
fi
shopt -s nullglob
id_files=("$FLEET_DIR"/*.id)
[ ${#id_files[@]} -gt 0 ] || { echo "FATAL: no *.id files in $FLEET_DIR (launch a fleet first)." >&2; exit 1; }

# ── Bandwidth semaphore (mkdir is atomic across the backgrounded run_one_job subshells) ──
SEMDIR="$FLEET_DIR/.bw_sem"; rm -rf "$SEMDIR"; mkdir -p "$SEMDIR"
_bw_acquire() {
  while :; do
    local i
    for i in $(seq 1 "$BW_CONCURRENCY"); do
      if mkdir "$SEMDIR/slot$i" 2>/dev/null; then BW_SLOT="$SEMDIR/slot$i"; return; fi
    done
    sleep 1
  done
}
_bw_release() { [ -n "${BW_SLOT:-}" ] && rmdir "$BW_SLOT" 2>/dev/null; BW_SLOT=""; }

# ── Wait (bounded) for a box to report 'running' VIA THE CACHE (no per-poll API walk) ──
wait_running_cached() {
  local iid="$1" i st
  for i in $(seq 1 "${RUN_TRIES:-80}"); do
    st="$(cached_instance_status "$iid")"
    [ "$st" = running ] && return 0
    sleep 15
  done
  return 1
}

# ── Per-box lifecycle: every exit path is RECORDED. RUN_ONE_BODY can be overridden to
#    a mock for offline testing (MOCK_RUN=1) — the lifecycle/bookkeeping is identical.
run_one_job() {
  local tag="$1" jid="$2"
  local iid model sidx scount ngpu jdir log
  iid="$(cat "$FLEET_DIR/$tag.id")"
  IFS=$'\t' read -r model sidx scount ngpu < "$FLEET_DIR/$tag.job"
  jdir="$JOBS_DIR/$jid"; log="$jdir/box.log"
  jr_set "$jid" instance_id "$iid" >/dev/null
  jr_set "$jid" status launched >/dev/null

  {
    echo "[$jid] tag=$tag instance=$iid model=$model shard=$sidx/$scount study=$STUDIES"

    if [ "${MOCK_RUN:-0}" = 1 ]; then
      # Offline test hook: a mock provides the box's behaviour without any cloud.
      "${MOCK_BODY:?MOCK_BODY must be set when MOCK_RUN=1}" "$jid" "$jdir" "$model" || {
        jr_fail "$jid" "mock body failed"; return 1; }
      jr_set "$jid" status synced >/dev/null
      echo "[$jid] (mock) synced"
      return 0
    fi

    # 1. Wait until 'running' (cached). Never came up -> loud fail + destroy.
    if ! wait_running_cached "$iid"; then
      jr_fail "$jid" "never reached 'running' within budget"
      INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure || true
      return 1
    fi

    # 2. Resolve the SSH endpoint ONCE from the cache; export so sync_up/at_vast inherit
    #    it (no per-call API walk). If it can't resolve, that's a loud failure.
    if ! cached_ssh_target "$iid"; then
      jr_fail "$jid" "SSH endpoint never resolved (cache+live)"
      INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure || true
      return 1
    fi
    export SSH_HOST SSH_PORT SSH_USER

    # 2b. Probe real sshd readiness (API 'running' precedes sshd). Bounded.
    local ready=0 i
    for i in $(seq 1 "${WAIT_SSH_TRIES:-60}"); do
      if INSTANCE="$iid" bash "$HERE/at_vast.sh" "true" >/dev/null 2>&1; then ready=1; break; fi
      sleep 10
    done
    [ "$ready" = 1 ] || { jr_fail "$jid" "sshd never became reachable"; INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure || true; return 1; }

    # 3. Push code + build env (bandwidth-capped, retried). Failure -> loud fail + destroy.
    _bw_acquire
    local setup_ok=0
    for i in $(seq 1 "${SETUP_TRIES:-20}"); do
      if INSTANCE="$iid" bash "$HERE/sync_up.sh" && \
         INSTANCE="$iid" bash "$HERE/at_vast.sh" "bash cloud/at_setup.sh"; then setup_ok=1; break; fi
      echo "[$jid] setup attempt $i failed; retrying"; sleep "${SETUP_BACKOFF:-30}"
    done
    _bw_release
    [ "$setup_ok" = 1 ] || { jr_fail "$jid" "sync_up/at_setup failed after retries"; INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure || true; return 1; }

    jr_set "$jid" status running >/dev/null

    # 4. Run THIS box's single-study pipeline. Nonzero -> loud fail + destroy.
    local hf_device_map=""; [ "${ngpu:-1}" -gt 1 ] 2>/dev/null && hf_device_map="auto"
    if ! INSTANCE="$iid" bash "$HERE/at_vast.sh" \
        "HF_TOKEN='${HF_TOKEN:-}' HF_DEVICE_MAP='$hf_device_map' MODEL='$model' SHARD_INDEX=$sidx SHARD_COUNT=$scount STUDIES='$STUDIES' BATCH_SIZE=$BATCH_SIZE N_THINKING=$N_THINKING bash cloud/fleet_model_run.sh"; then
      jr_fail "$jid" "on-box run (fleet_model_run.sh) returned nonzero"
      INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure || true
      return 1
    fi

    # 5. Pull results into jobs/<id>/output/ (bandwidth-capped). Never touches out/.
    _bw_acquire
    INSTANCE="$iid" STUDIES="$STUDIES" QUARANTINE_ROOT="$jdir/output" bash "$HERE/sync_back.sh"
    local sb=$?
    _bw_release
    [ "$sb" = 0 ] || { jr_fail "$jid" "sync_back failed (rc=$sb)"; INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure || true; return 1; }

    # 6. Mark synced (the monitor force-verifies from here) and self-destroy the box.
    jr_set "$jid" status synced >/dev/null
    INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure || true
    echo "[$jid] synced + box destroyed"
  } >>"$log" 2>&1
}

# ── 1. Start the shared state daemon (singleton; serves status/SSH to every box) ──
STATE_INTERVAL="${STATE_INTERVAL:-15}" bash "$HERE/fleet_state_daemon.sh" &
DAEMON_BG=$!
echo ">> state daemon started (pid $DAEMON_BG)"

# ── 2. Create EVERY job up front (so the monitor never sees an empty list and exits) ──
declare -a TAGS=() JIDS=()
for idf in "${id_files[@]}"; do
  tag="$(basename "$idf" .id)"
  IFS=$'\t' read -r model sidx scount ngpu < "$FLEET_DIR/$tag.job"
  jid="$(jr_new "$STUDIES" "$model" "$sidx" "$scount")" || { echo "FATAL: could not create job for $tag" >&2; exit 1; }
  TAGS+=("$tag"); JIDS+=("$jid")
  echo ">> job $jid  <-  box $tag ($model shard $sidx/$scount)"
done

# ── 3. Start the always-on monitor (now jr_list is non-empty) ──────────────
bash "$HERE/fleet_monitor_agent.sh" &
MON_BG=$!
echo ">> monitor started (pid $MON_BG); live dashboard: cat $FLEET_DIR/monitor.dash"

# ── 4. Drive every box CONCURRENTLY (bandwidth-capped only on the sync steps) ──
declare -a RUN_PIDS=()
for i in "${!TAGS[@]}"; do
  run_one_job "${TAGS[$i]}" "${JIDS[$i]}" &
  RUN_PIDS+=($!)
done
# Wait for the BOX jobs specifically (NOT the daemon/monitor, which we manage below).
for pid in "${RUN_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done

# ── 5. Let the monitor finish its last verify pass + reconciliation, then stop daemon ──
echo ">> all boxes finished; waiting for monitor's final verify + reconciliation ..."
wait "$MON_BG" 2>/dev/null || true
touch "$FLEET_DIR/.state_daemon.stop"; wait "$DAEMON_BG" 2>/dev/null || true

# ── 6. Summary ─────────────────────────────────────────────────────────────
echo
echo "================= FLEET DONE ================="
[ -f "$FLEET_DIR/monitor.dash" ] && cat "$FLEET_DIR/monitor.dash"
echo
echo "Per-job evidence:   jobs/<id>/VERIFY.txt   transitions: $FLEET_DIR/monitor.log"
echo "Promote VERIFIED into out/ (manual, gated):"
echo "    bash cloud/promote_verified_jobs.sh            # dry-run + refusals"
echo "    bash cloud/promote_verified_jobs.sh --promote  # move verified payloads"
echo "=============================================="
