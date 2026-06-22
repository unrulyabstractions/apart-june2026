#!/usr/bin/env bash
#
# promote_verified_jobs.sh — the MANUAL, verification-gated move of job output into out/.
#
# This is the ONLY path from jobs/<id>/output/ into out/. It is deliberately manual
# (you run it) and it is impossible to promote anything that does not re-verify.
#
# NO-PROXY GATE: it does NOT trust the manifest's verdict. For every job it RE-RUNS
# the deterministic verifier on the actual payload (scripts/verify_experiment_output.py).
# Only a job whose payload re-verifies NOW is eligible. A manifest that claims VERIFIED
# but whose payload no longer verifies is REFUSED and flagged loudly.
#
# NOTHING SILENTLY FAILS: every job prints a line; every refusal prints its reason;
# every file that already exists in out/ (an --ignore-existing skip) is listed as a
# CONFLICT, never dropped quietly. A non-empty refusal list makes the script exit 1.
#
# NEVER CLOBBERS: the move is rsync --ignore-existing, no --delete — an existing out/
# file is kept and reported as a conflict, never overwritten. The jobs/ dir is left
# intact as an audit record (gitignored).
#
# Usage:
#   bash cloud/promote_verified_jobs.sh            # DRY-RUN: show what would promote + every refusal
#   bash cloud/promote_verified_jobs.sh --promote  # actually move verified payloads into out/
#   JOB=<job-id> bash cloud/promote_verified_jobs.sh [--promote]   # one job only

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
JOBS_DIR="${JOBS_DIR:-$REPO_ROOT/jobs}"
OUT="${PROMOTE_OUT_DIR:-$REPO_ROOT/out}"   # override only for sandbox testing
# shellcheck source=/dev/null
. "$HERE/job_registry.sh"

DO_PROMOTE=0; [ "${1:-}" = "--promote" ] && DO_PROMOTE=1

[ -d "$JOBS_DIR" ] || { echo "No jobs/ dir; nothing to promote."; exit 0; }

# Files under output/ that ALREADY exist in out/ — the --ignore-existing skips. Listed
# so a corrected result silently NOT promoted can never hide. Echoes one rel path/line.
_conflicts() {
  local out_payload="$1"
  OUT_PAYLOAD="$out_payload" OUT_ROOT="$OUT" python3 -c '
import os
src=os.environ["OUT_PAYLOAD"]; dst=os.environ["OUT_ROOT"]
for dirpath,_,files in os.walk(src):
    for f in files:
        rel=os.path.relpath(os.path.join(dirpath,f), src)
        if os.path.exists(os.path.join(dst, rel)):
            print(rel)
'
}

declare -a PROMOTED=() REFUSED=() CONFLICTS=()

process_job() {
  local jid="$1" dir payload sub vpath status verdict nfiles
  dir="$JOBS_DIR/$jid"
  payload="$dir/output"
  [ -f "$dir/manifest.json" ] || { echo "[$jid] SKIP: no manifest (not a job dir)"; return; }
  status="$(jr_get "$jid" status 2>/dev/null || echo '?')"
  verdict="$(jr_get "$jid" verdict 2>/dev/null || echo '')"
  sub="$(jr_get "$jid" verify_subpath 2>/dev/null || echo '')"
  vpath="$payload${sub:+/$sub}"

  # 1. the INTENDED output slice must be non-empty. An empty intended slice means the
  #    box produced nothing there, or ran the WRONG model and wrote a different slice
  #    (gotcha #9). Either way: refuse, never promote blindly.
  nfiles=$(find "$vpath" -type f 2>/dev/null | wc -l | tr -d ' ')
  if [ "$nfiles" -eq 0 ]; then
    echo "[$jid] REFUSE: intended slice ${sub:-output/} is EMPTY (wrong model or no output; status=$status verdict=$verdict)"
    REFUSED+=("$jid: empty intended slice ${sub:-output/} (status=$status)"); return
  fi

  # 2. AUTHORITATIVE re-verification of the INTENDED slice (ignores the manifest verdict).
  local vout vrc
  vout="$(cd "$REPO_ROOT" && uv run python scripts/verify_experiment_output.py "$vpath" 2>&1)"; vrc=$?
  if [ $vrc -ne 0 ]; then
    echo "[$jid] REFUSE: payload did NOT re-verify (manifest verdict=${verdict:-none}):"
    printf '        %s\n' "$vout" | sed 's/^        - /        /'
    REFUSED+=("$jid: re-verification failed (manifest said '${verdict:-none}')")
    jr_set "$jid" verdict BROKEN 2>/dev/null || true
    return
  fi

  # 3. eligible. Compute conflicts (existing out/ files this move will NOT overwrite).
  local conf; conf="$(_conflicts "$payload")"
  if [ -n "$conf" ]; then
    while IFS= read -r c; do CONFLICTS+=("$jid: $c"); done <<<"$conf"
  fi

  if [ $DO_PROMOTE -eq 1 ]; then
    rsync -a --ignore-existing "$payload/" "$OUT/"
    jr_set "$jid" status promoted 2>/dev/null || true
    jr_set "$jid" verdict VERIFIED 2>/dev/null || true
    local nconf=0; [ -n "$conf" ] && nconf=$(wc -l <<<"$conf" | tr -d ' ')
    echo "[$jid] PROMOTED: $nfiles files -> out/ (re-verified OK; $nconf existing files kept — see summary)"
    PROMOTED+=("$jid")
  else
    echo "[$jid] WOULD PROMOTE: $nfiles files (re-verified OK)${conf:+, $(wc -l <<<"$conf"|tr -d ' ') conflicts will be KEPT-EXISTING}"
  fi
}

if [ -n "${JOB:-}" ]; then
  process_job "$JOB"
else
  while IFS= read -r jid; do process_job "$jid"; done < <(jr_list)
fi

echo
echo "================ PROMOTION SUMMARY ================"
echo "promoted/eligible : ${#PROMOTED[@]}"
echo "refused           : ${#REFUSED[@]}"
if [ ${#REFUSED[@]} -gt 0 ]; then
  printf '  ✗ %s\n' "${REFUSED[@]}"
fi
echo "conflicts (existing out/ files KEPT, cloud copy ignored): ${#CONFLICTS[@]}"
if [ ${#CONFLICTS[@]} -gt 0 ]; then
  printf '  ! %s\n' "${CONFLICTS[@]}"
  echo "  -> if any of these is a CORRECTED result, the existing out/ file was kept;"
  echo "     remove the stale out/ file by hand and re-run to take the new one."
fi
[ $DO_PROMOTE -eq 0 ] && echo "(dry-run; re-run with --promote to move the eligible payloads)"
echo "=================================================="

# A non-empty refusal list is a LOUD non-zero exit so this can never pass unnoticed.
[ ${#REFUSED[@]} -gt 0 ] && exit 1 || exit 0
