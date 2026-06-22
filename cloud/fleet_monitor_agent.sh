#!/usr/bin/env bash
#
# fleet_monitor_agent.sh — the always-on monitor. Auto-started by the orchestrator so
# EVERY fleet run is monitored by construction (req. "each job ALWAYS monitored").
#
# It does three things, forever, until every job is terminal (or a stop file appears):
#
#  1. WATCH    — each cycle, render every job's live state (instance status from the
#                shared cache, manifest status, last log line) to monitor.dash.
#  2. FORCE-VERIFY — the moment a job reaches status=synced, it RUNS the deterministic
#                verifier on jobs/<id>/output and writes verdict VERIFIED|BROKEN to the
#                manifest. Verification is automatic and unskippable — not my choice.
#  3. RECONCILE — nothing silently fails: on exit it asserts EVERY job is terminal
#                (promoted|verified|broken|failed). Any job still created/launched/
#                running/synced, or whose box vanished without recording a result, is
#                reported LOUDLY as STALLED. A box that died is a failure, never silence.
#
# Outputs (under $FLEET_DIR): monitor.dash (overwritten snapshot), monitor.log
# (appended transition/verification events). PID + stop file for lifecycle.
#
# Deterministic gate only. For the deep adversarial pass (VIEW each figure, cross-check
# intent) the main agent additionally spawns the `verifier` subagent on the outputs.
#
# Usage:
#   bash cloud/fleet_monitor_agent.sh &        # background; exits when all jobs terminal
#   ONESHOT=1 bash cloud/fleet_monitor_agent.sh  # one pass (verify+render+reconcile), no loop
#   touch cloud/.fleet/.monitor.stop            # ask it to exit

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
FLEET_DIR="${FLEET_DIR:-$HERE/.fleet}"
JOBS_DIR="${JOBS_DIR:-$REPO_ROOT/jobs}"
INTERVAL="${MONITOR_INTERVAL:-20}"
DASH="$FLEET_DIR/monitor.dash"
MLOG="$FLEET_DIR/monitor.log"
STOP_FILE="$FLEET_DIR/.monitor.stop"
PID_FILE="$FLEET_DIR/.monitor.pid"
TERMINAL="promoted verified broken failed"

mkdir -p "$FLEET_DIR"
# shellcheck source=/dev/null
. "$HERE/job_registry.sh"
# shellcheck source=/dev/null
. "$HERE/fleet_state_lookup.sh"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
  echo "[monitor] already running (pid $(cat "$PID_FILE")); not starting another." >&2
  exit 0
fi
echo "$$" >"$PID_FILE"; rm -f "$STOP_FILE"

_log() { echo "[$(date '+%H:%M:%S')] $*" >>"$MLOG"; }
_is_terminal() { case " $TERMINAL " in *" $1 "*) return 0;; *) return 1;; esac; }

# Force-verify one synced job: run the deterministic verifier on the job's INTENDED
# output slice (verify_subpath) and record the verdict. A synced job NEVER stays
# unverified — this is the unskippable gate. Verifying the INTENDED slice (not "whatever
# landed in output/") also catches the wrong-model class (gotcha #9): a box that ran the
# wrong model writes a DIFFERENT slice, so the intended path is empty -> BROKEN.
_force_verify() {
  local jid="$1" sub payload vrc
  sub="$(jr_get "$jid" verify_subpath 2>/dev/null || echo '')"
  payload="$JOBS_DIR/$jid/output${sub:+/$sub}"
  ( cd "$REPO_ROOT" && uv run python scripts/verify_experiment_output.py "$payload" ) \
      >"$JOBS_DIR/$jid/VERIFY.txt" 2>&1
  vrc=$?
  if [ $vrc -eq 0 ]; then
    jr_set "$jid" verdict VERIFIED >/dev/null; jr_set "$jid" status verified >/dev/null
    _log "VERIFIED  $jid  (see jobs/$jid/VERIFY.txt)"
  else
    jr_set "$jid" verdict BROKEN >/dev/null; jr_set "$jid" status broken >/dev/null
    _log "BROKEN    $jid  -> $(tail -1 "$JOBS_DIR/$jid/VERIFY.txt")"
  fi
}

# One full pass: verify any synced jobs, then render the dashboard. Echoes the number
# of jobs NOT yet terminal (0 == fleet done).
_pass() {
  local jid status inst_status pending=0 total=0
  local n_run=0 n_sync=0 n_ver=0 n_broke=0 n_fail=0 n_prom=0
  : >"$DASH.tmp"
  {
    printf '%-46s %-10s %-9s %-8s %s\n' "JOB" "BOX" "STATUS" "VERDICT" "LAST LOG"
    printf '%s\n' "----------------------------------------------------------------------------------------------"
  } >>"$DASH.tmp"
  while IFS= read -r jid; do
    [ -n "$jid" ] || continue
    total=$((total+1))
    status="$(jr_get "$jid" status 2>/dev/null || echo '?')"
    # Auto-verify the instant a box's output is synced (the FORCE step).
    if [ "$status" = synced ]; then _force_verify "$jid"; status="$(jr_get "$jid" status)"; fi
    local verdict iid; verdict="$(jr_get "$jid" verdict 2>/dev/null || echo '')"
    iid="$(jr_get "$jid" instance_id 2>/dev/null || echo '')"
    inst_status="-"; [ -n "$iid" ] && inst_status="$(cached_instance_status "$iid" 2>/dev/null || echo '?')"
    local last="-"; [ -f "$JOBS_DIR/$jid/box.log" ] && last="$(tail -1 "$JOBS_DIR/$jid/box.log" 2>/dev/null | cut -c1-60)"
    printf '%-46s %-10s %-9s %-8s %s\n' "${jid:0:46}" "$inst_status" "$status" "${verdict:--}" "$last" >>"$DASH.tmp"
    case "$status" in
      running|created|launched) n_run=$((n_run+1));;
      synced) n_sync=$((n_sync+1));;
      verified) n_ver=$((n_ver+1));;
      broken) n_broke=$((n_broke+1));;
      failed) n_fail=$((n_fail+1));;
      promoted) n_prom=$((n_prom+1));;
    esac
    _is_terminal "$status" || pending=$((pending+1))
  done < <(jr_list)
  {
    printf '%s\n' "----------------------------------------------------------------------------------------------"
    printf 'total=%d  running=%d  synced=%d  VERIFIED=%d  BROKEN=%d  FAILED=%d  promoted=%d   (updated %s)\n' \
      "$total" "$n_run" "$n_sync" "$n_ver" "$n_broke" "$n_fail" "$n_prom" "$(date '+%H:%M:%S')"
    [ $((n_broke+n_fail)) -gt 0 ] && printf '!! %d job(s) BROKEN/FAILED — see monitor.log and jobs/<id>/VERIFY.txt\n' "$((n_broke+n_fail))"
  } >>"$DASH.tmp"
  mv -f "$DASH.tmp" "$DASH"
  echo "$pending"
}

# Final reconciliation — the anti-silent-failure backstop. Any non-terminal job at the
# end is a LOUD stall (its box died, or a step swallowed an error without recording it).
_reconcile() {
  local jid status stalled=0
  _log "=== RECONCILIATION ==="
  while IFS= read -r jid; do
    [ -n "$jid" ] || continue
    status="$(jr_get "$jid" status 2>/dev/null || echo '?')"
    if ! _is_terminal "$status"; then
      stalled=$((stalled+1))
      _log "STALLED   $jid  status=$status  — box died or a step failed without recording. INVESTIGATE."
      echo "!! STALLED: $jid (status=$status) — NOT terminal. Investigate; nothing is silently OK." >&2
    fi
  done < <(jr_list)
  if [ "$stalled" -eq 0 ]; then
    _log "All jobs terminal (verified/broken/failed/promoted). No silent failures."
  else
    _log "$stalled job(s) STALLED — see above. THIS IS A LOUD FAILURE."
  fi
}

_log "monitor started (pid $$, interval ${INTERVAL}s)"
if [ "${ONESHOT:-0}" = 1 ]; then
  _pass >/dev/null; _reconcile; rm -f "$PID_FILE"; exit 0
fi
while :; do
  [ -e "$STOP_FILE" ] && { _log "stop file seen"; break; }
  pending="$(_pass)"
  [ "$pending" -eq 0 ] && { _log "all jobs terminal; monitor exiting."; break; }
  sleep "$INTERVAL"
done
_reconcile
rm -f "$PID_FILE"
