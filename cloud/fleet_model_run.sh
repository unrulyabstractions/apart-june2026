#!/usr/bin/env bash
#
# fleet_model_run.sh — ON-BOX driver: run ONE model's (shard's) batched SESGO
# pipeline. Invoked remotely by fleet_run.sh via at_vast.sh, e.g.:
#   MODEL='google/gemma-2-2b-it' SHARD_INDEX=0 SHARD_COUNT=1 \
#     STUDIES='baseline divergence' BATCH_SIZE=32 bash cloud/fleet_model_run.sh
#
# Writes ONLY to this box's own DISJOINT slice:
#   out/sesgo/<study>/<bare-model>/[shard_<k>_of_<K>/]samples.json
# so concurrent boxes (other models / other shards) never share a path. The
# --batch-size flag drives vLLM continuous batching on the GPU (the cloud fast
# path); on this box vLLM is in the image, so the HuggingFace+CUDA path or the
# vLLM backend both honor --batch-size.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root on the remote (/root/apart)

# at_setup pinned a cu124 torch (+ nvidia-cu12 libs) into .venv. `uv run` ALWAYS
# re-syncs the venv to the lockfile first, which silently reinstalls the cu130 wheel
# and breaks CUDA on any pre-CUDA-13 host (UV_NO_SYNC did NOT prevent this in
# practice). So we invoke the venv interpreter DIRECTLY — it is already fully built
# by at_setup, runs from the repo root on sys.path, and never re-syncs.
PY="$PWD/.venv/bin/python"
[ -x "$PY" ] || { echo "[fleet_model_run] FATAL: $PY missing (run at_setup first)" >&2; exit 1; }

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
STUDIES="${STUDIES:-baseline divergence}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_THINKING="${N_THINKING:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
# SUBSAMPLE: fraction (0-1) of the grid to query per box. UNSET == full grid
# (backward-compatible no-op). When set, each box subsamples the SAME strided
# slice (deterministic) before taking its shard, so the K shards still tile one
# subsampled grid. Only the thinking studies (divergence/selection/geometry)
# accept --subsample; baseline/stability ignore it.
SUBSAMPLE="${SUBSAMPLE:-}"
# Build the optional --subsample flag once so each study case stays one line.
SUBSAMPLE_ARG=""
[ -n "$SUBSAMPLE" ] && SUBSAMPLE_ARG="--subsample $SUBSAMPLE"
# ITEMS: number of DISTINCT items (question_ids) to keep for the STABILITY study,
# keeping ALL 36 surface variants of each. UNSET == full item×variant grid
# (backward-compatible no-op). Stability consistency is per-item ACROSS its
# variants, so a flat --subsample would leave it undefined; --items keeps whole
# items, which is the valid way to take a tractable stability slice (~60 items ×
# 36 variants per box). Only the stability case reads ITEMS; other studies ignore it.
ITEMS="${ITEMS:-}"
ITEMS_ARG=""
[ -n "$ITEMS" ] && ITEMS_ARG="--items $ITEMS"
# GENERATE_ALL_DATA: opt-in pass-through to the generator. UNSET == es-original
# studies only (backward-compatible no-op); set to 1 so the generator ALSO writes
# out/sesgo/full_data/prompt_dataset.json (all langs x all origins x {none+3
# scaffolds}) that the full_data study below consumes. Exported so the
# generator subprocess inherits it.
export GENERATE_ALL_DATA="${GENERATE_ALL_DATA:-}"
# HF_FORWARD_MICRO_BATCH: cap on sequences per teacher-forced forward pass so a wide
# batch of long (scaffolded) prompts never OOMs the GPU. Exported so the on-box
# collect subprocess (model_runner) inherits it. UNSET == one pass (no-op).
export HF_FORWARD_MICRO_BATCH="${HF_FORWARD_MICRO_BATCH:-}"

echo "[fleet_model_run] model=$MODEL shard=$SHARD_INDEX/$SHARD_COUNT studies='$STUDIES' batch=$BATCH_SIZE subsample='${SUBSAMPLE:-full}' items='${ITEMS:-full}' all_data='${GENERATE_ALL_DATA:-off}'"

# Regenerate the prompt datasets on the box (reads the synced xlsx). With
# GENERATE_ALL_DATA=1 this also writes out/sesgo/full_data/prompt_dataset.json.
echo "[fleet_model_run] generate prompt datasets"
"$PY" sesgo/generate/generate_prompt_dataset.py

# DEFENSE IN DEPTH: if sync_up did not deliver the SESGO prompt xlsx (e.g. a
# half-synced box), generate writes a prompt_dataset.json with ZERO samples and the
# collect below silently produces an EMPTY response_samples.json. Abort loudly here
# so the box is never billed for an empty run and never syncs an empty result.
for study in $STUDIES; do
  ds="out/sesgo/$study/prompt_dataset.json"
  n="$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1])).get('samples',[])))" "$ds" 2>/dev/null || echo 0)"
  if [ "${n:-0}" -lt 1 ]; then
    echo "[fleet_model_run] FATAL: $ds has $n prompts -- prompt sources missing (sync_up incomplete?). Aborting." >&2
    exit 1
  fi
  echo "[fleet_model_run] $study prompt_dataset has $n prompts"
done

SHARD_ARGS="--shard-index $SHARD_INDEX --shard-count $SHARD_COUNT"

run_study() {
  local study="$1"
  case "$study" in
    baseline)
      "$PY" sesgo/baseline/collect_baseline_samples.py \
        --model "$MODEL" --out-dir out --batch-size "$BATCH_SIZE" $SHARD_ARGS ;;
    full_data)
      # Full-data baseline: read the full-grid prompt dataset (all langs x all
      # origins x {none+3 scaffolds}) and route output to its OWN full_data tree
      # so it never clobbers the es-original baseline. Same readouts as baseline.
      "$PY" sesgo/baseline/collect_baseline_samples.py \
        out/sesgo/full_data/prompt_dataset.json \
        --model "$MODEL" --out-dir out --study full_data \
        --batch-size "$BATCH_SIZE" $SHARD_ARGS ;;
    selection)
      "$PY" sesgo/selection/collect_selection_samples.py \
        --model "$MODEL" --out-dir out --batch-size "$BATCH_SIZE" \
        --n-thinking "$N_THINKING" --max-new-tokens "$MAX_NEW_TOKENS" $SUBSAMPLE_ARG $SHARD_ARGS ;;
    divergence)
      "$PY" sesgo/divergence/collect_divergence_samples.py \
        --model "$MODEL" --out-dir out --batch-size "$BATCH_SIZE" \
        --n-thinking "$N_THINKING" --max-new-tokens "$MAX_NEW_TOKENS" $SUBSAMPLE_ARG $SHARD_ARGS ;;
    stability)
      "$PY" sesgo/stability/collect_stability_samples.py \
        --model "$MODEL" --out-dir out $ITEMS_ARG $SHARD_ARGS ;;
    geometry)
      "$PY" sesgo/geometry/collect_geometry_samples.py \
        --model "$MODEL" --out-dir out --batch-size "$BATCH_SIZE" \
        --n-thinking "$N_THINKING" $SUBSAMPLE_ARG $SHARD_ARGS ;;
    *) echo "[fleet_model_run] unknown study: $study" >&2; return 1 ;;
  esac
}

for study in $STUDIES; do
  echo "[fleet_model_run] === $study (model=$MODEL, shard=$SHARD_INDEX/$SHARD_COUNT) ==="
  run_study "$study"
done

echo "[fleet_model_run] done. Slice: out/sesgo/<study>/$(basename "$MODEL")/"
