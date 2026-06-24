#!/usr/bin/env bash
#
# run_forking_sharded_fleet.sh — orchestrate the POSITION-SHARDED forking fleet.
#
# Runs the three-phase sharded forking-paths pipeline across N+1 cloud boxes:
#
#   STEP A  ONE phase-1 base box selects the item + decodes the base path, syncing
#           selected_item.json + base_path.json down to sync/forkbase/. We WAIT for
#           base_path.json to land locally before fanning out.
#   STEP B  NUM_SHARDS phase-2 shard boxes launch IN PARALLEL (one bg job + log per
#           shard, /tmp/forkshard_<k>.log). Each pushes the local artifacts UP, forks
#           positions[k::NUM_SHARDS], and syncs forking_shard_<k>_of_<N>.json down to
#           sync/forkshards/shard_<k>/. A shard box dying is logged but does NOT abort
#           the fleet — the merge step LOUDLY reports any missing positions.
#   STEP C  LOCALLY: promote every shard into one dir, run merge_forking_shards.py
#           (reassembles positions by REAL index, validates 0..P-1 coverage), then
#           analyze + plot_forking_dynamics.py on the merged trajectory.
#
# Env:
#   MODEL (Qwen/Qwen3-14B)  NUM_SHARDS (5)
#   plus every knob the per-box scripts accept (GPU_NAME, MAX_PRICE, N_PRIOR,
#   N_SAMPLES, MAX_NEW_TOKENS, BASE_MAX_NEW_TOKENS, HF_GEN_MICRO_BATCH, ...).
#
# Usage:
#   MODEL=Qwen/Qwen3-14B NUM_SHARDS=5 bash cloud/run_forking_sharded_fleet.sh

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-14B}"
BARE_MODEL="${MODEL##*/}"
NUM_SHARDS="${NUM_SHARDS:-5}"

FORKBASE_ARTIFACTS="$REPO_ROOT/sync/forkbase/forking/$BARE_MODEL"
SHARDS_PROMOTED="$REPO_ROOT/sync/forkmerged/$BARE_MODEL"
FINAL_OUT="$REPO_ROOT/out/forking/$BARE_MODEL"

log() { echo "[fleet $(date +%H:%M:%S)] $*"; }

export MODEL  # propagate the model to every per-box script

# ── STEP A: base box (select + decode base path), WAIT for base_path.json ──
# Skip the base box entirely when a valid base_path.json + selected_item.json
# already exist locally (a prior fleet already decoded this model's base path).
# The base CoT only needs decoding ONCE, so reuse it and go straight to shards.
if [ -f "$FORKBASE_ARTIFACTS/base_path.json" ] && [ -f "$FORKBASE_ARTIFACTS/selected_item.json" ]; then
  log "STEP A: reusing existing base_path.json + selected_item.json (skip base box)"
else
  log "STEP A: launching phase-1 base box for $MODEL"
  bash "$HERE/run_one_forking_base_box.sh" 2>&1 | sed "s/^/[fleet base] /"
  BASE_RC="${PIPESTATUS[0]}"
  if [ "$BASE_RC" -ne 0 ] || [ ! -f "$FORKBASE_ARTIFACTS/base_path.json" ]; then
    log "FATAL: base box failed (rc=$BASE_RC) or base_path.json never landed at $FORKBASE_ARTIFACTS"
    exit 1
  fi
fi
NPOS="$(python3 -c "import json; print(len(json.load(open('$FORKBASE_ARTIFACTS/base_path.json'))['base_token_ids']))" 2>/dev/null || echo 0)"
log "STEP A done: base_path.json landed with ${NPOS:-?} positions"

# ── STEP B: NUM_SHARDS shard boxes IN PARALLEL ──
log "STEP B: launching $NUM_SHARDS shard boxes in parallel"
declare -a PIDS=()
for k in $(seq 0 $((NUM_SHARDS - 1))); do
  klog="/tmp/forkshard_${k}.log"
  log "  launching shard $k/$NUM_SHARDS  (log: $klog)"
  SHARD_INDEX="$k" NUM_SHARDS="$NUM_SHARDS" \
    bash "$HERE/run_one_forking_shard_box.sh" >"$klog" 2>&1 &
  PIDS+=("$!")
done

# ── STEP C(wait): join every shard box, logging which succeeded / died ──
log "STEP C: waiting for $NUM_SHARDS shard boxes to finish"
declare -a FAILED=()
for k in $(seq 0 $((NUM_SHARDS - 1))); do
  if wait "${PIDS[$k]}"; then
    log "  shard $k FINISHED ok (log: /tmp/forkshard_${k}.log)"
  else
    log "  shard $k DIED (rc=$?) -- continuing; merge will report any missing positions (log: /tmp/forkshard_${k}.log)"
    FAILED+=("$k")
  fi
done
[ "${#FAILED[@]}" -eq 0 ] && log "all $NUM_SHARDS shard boxes succeeded" \
  || log "WARNING: ${#FAILED[@]} shard box(es) failed: ${FAILED[*]} (merge will flag gaps)"

# ── STEP C(promote): gather every shard file into ONE dir for the merge ──
log "promoting shard files into $SHARDS_PROMOTED"
mkdir -p "$SHARDS_PROMOTED"
find "$REPO_ROOT/sync/forkshards" -name "forking_shard_*_of_*.json" -exec cp -n {} "$SHARDS_PROMOTED/" \; 2>/dev/null
N_FOUND="$(find "$SHARDS_PROMOTED" -name 'forking_shard_*_of_*.json' | wc -l | tr -d ' ')"
log "promoted $N_FOUND/$NUM_SHARDS shard files"
[ "$N_FOUND" -ge 1 ] || { log "FATAL: no shard files to merge"; exit 2; }

# Seed the final out dir with the selected_item + base_path so analyze/plot resolve it.
mkdir -p "$FINAL_OUT"
cp -n "$FORKBASE_ARTIFACTS/selected_item.json" "$FINAL_OUT/" 2>/dev/null || true
cp -n "$FORKBASE_ARTIFACTS/base_path.json" "$FINAL_OUT/" 2>/dev/null || true

# ── STEP C(merge): reassemble shards into one ordered trajectory LOCALLY ──
log "STEP C: merge shards -> $FINAL_OUT/forking_trajectory.json"
( cd "$REPO_ROOT" && uv run python experiment/forking/merge_forking_shards.py \
    --in-dir "$SHARDS_PROMOTED" --out-dir "$FINAL_OUT" ) 2>&1 | sed "s/^/[fleet merge] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: merge failed"; exit 3; }

# ── STEP C(analyze + plot): drive the existing downstream on the merged file ──
log "analyze + plot the merged trajectory"
( cd "$REPO_ROOT" && uv run python experiment/forking/analyze_forking_dynamics.py --model "$MODEL" ) 2>&1 | sed "s/^/[fleet analyze] /"
( cd "$REPO_ROOT" && uv run python experiment/forking/plot_forking_dynamics.py --model "$MODEL" ) 2>&1 | sed "s/^/[fleet plot] /"

log "DONE: sharded forking trajectory for $BARE_MODEL at $FINAL_OUT/forking_trajectory.json"
log "      figures: $FINAL_OUT/forking_dynamics.png , forking_branching_tree.png"
exit 0
