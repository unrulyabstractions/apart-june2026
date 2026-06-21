#!/usr/bin/env bash
#
# fleet_destroy.sh — destroy EVERY fleet box CONCURRENTLY (the safety net).
#
# fleet_run.sh self-destroys each box as it finishes, so this is the BACKSTOP:
# run it if a launch was aborted, a box hung, or you just want to be sure no
# instance is still billing. It reads cloud/.fleet/*.id and tears each down in
# parallel via the existing vast_destroy.sh (which needs the explicit
# --yes-i-am-really-sure to actually act).
#
# Usage:
#   bash cloud/fleet_destroy.sh --yes-i-am-really-sure
#   bash cloud/fleet_destroy.sh                      # dry: prints what it WOULD do

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FLEET_DIR="$HERE/.fleet"
CONFIRM="${1:-}"

[ -d "$FLEET_DIR" ] || { echo "No fleet dir ($FLEET_DIR); nothing to destroy."; exit 0; }

shopt -s nullglob
ids=("$FLEET_DIR"/*.id)
[ ${#ids[@]} -gt 0 ] || { echo "No .id files in $FLEET_DIR; nothing to destroy."; exit 0; }

for idf in "${ids[@]}"; do
  tag="$(basename "$idf" .id)"
  iid="$(cat "$idf")"
  if [ "$CONFIRM" = "--yes-i-am-really-sure" ]; then
    echo "[$tag] destroying instance $iid ..."
    INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure &
  else
    echo "[$tag] WOULD destroy instance $iid (pass --yes-i-am-really-sure to act)"
  fi
done
wait

if [ "$CONFIRM" = "--yes-i-am-really-sure" ]; then
  rm -f "$FLEET_DIR"/*.id "$FLEET_DIR"/*.job
  echo ">> All fleet boxes destroyed; .id/.job records cleared."
fi
