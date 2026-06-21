#!/usr/bin/env bash
#
# fill_fleet_driver.sh — launch ONE narrow fill sub-fleet from a plan file, then
# drive every box (setup -> run -> partial-resumable -> sync-back -> self-destroy)
# to completion. Fire-and-forget: spawn this in the background per study.
#
# It is a thin orchestrator over the existing scripts (no new launch/run logic):
#   FLEET_DIR + FLEET_PLAN_FILE  ->  fleet_launch.sh   (create boxes, non-interactive)
#   FLEET_DIR + study knobs      ->  fleet_run.sh      (run + box-replacement resume)
#
# Required env:
#   FLEET_DIR        — distinct per sub-fleet (e.g. cloud/.fleet_fill_stab)
#   FLEET_PLAN_FILE  — the narrow TSV plan (e.g. cloud/.plan_fill_stab.tsv)
#   STUDIES          — the study this fleet collects (e.g. "stability")
#   HF_TOKEN         — for gated models (Llama-3.1-70B); harmless otherwise
# Optional study knobs pass straight through to fleet_run.sh (all backward-compat):
#   ITEMS, SUBSAMPLE, N_THINKING, MAX_NEW_TOKENS, BATCH_SIZE, PARTIAL_SYNC_EVERY

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

: "${FLEET_DIR:?set FLEET_DIR}"
: "${FLEET_PLAN_FILE:?set FLEET_PLAN_FILE}"
: "${STUDIES:?set STUDIES}"

echo "[fill_driver] === $STUDIES === FLEET_DIR=$FLEET_DIR plan=$FLEET_PLAN_FILE"
echo "[fill_driver] knobs: ITEMS=${ITEMS:-} SUBSAMPLE=${SUBSAMPLE:-} N_THINKING=${N_THINKING:-} BATCH_SIZE=${BATCH_SIZE:-} MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}"

# 1. Launch the boxes (non-interactive). FLEET_CONFIRM=1 skips the y/N prompt.
FLEET_CONFIRM=1 FLEET_DIR="$FLEET_DIR" FLEET_PLAN_FILE="$FLEET_PLAN_FILE" \
  bash "$HERE/fleet_launch.sh"
echo "[fill_driver] launch returned; ids under $FLEET_DIR/*.id"

# 2. Drive every launched box to completion + self-destroy (this blocks until the
#    slowest box finishes, then exits — the whole point of fire-and-forget here).
FLEET_DIR="$FLEET_DIR" bash "$HERE/fleet_run.sh"
echo "[fill_driver] === $STUDIES DONE — results quarantined under sync/box-* ==="
