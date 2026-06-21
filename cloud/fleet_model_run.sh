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

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
STUDIES="${STUDIES:-baseline divergence}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_THINKING="${N_THINKING:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

echo "[fleet_model_run] model=$MODEL shard=$SHARD_INDEX/$SHARD_COUNT studies='$STUDIES' batch=$BATCH_SIZE"

# Regenerate the prompt datasets on the box (full grid; reads the synced xlsx).
echo "[fleet_model_run] generate prompt datasets"
uv run python sesgo/generate/generate_prompt_dataset.py

SHARD_ARGS="--shard-index $SHARD_INDEX --shard-count $SHARD_COUNT"

run_study() {
  local study="$1"
  case "$study" in
    baseline)
      uv run python sesgo/baseline/collect_baseline_samples.py \
        --model "$MODEL" --out-dir out --batch-size "$BATCH_SIZE" $SHARD_ARGS ;;
    selection)
      uv run python sesgo/selection/collect_selection_samples.py \
        --model "$MODEL" --out-dir out --batch-size "$BATCH_SIZE" \
        --n-thinking "$N_THINKING" --max-new-tokens "$MAX_NEW_TOKENS" $SHARD_ARGS ;;
    divergence)
      uv run python sesgo/divergence/collect_divergence_samples.py \
        --model "$MODEL" --out-dir out --batch-size "$BATCH_SIZE" \
        --n-thinking "$N_THINKING" --max-new-tokens "$MAX_NEW_TOKENS" $SHARD_ARGS ;;
    stability)
      uv run python sesgo/stability/collect_stability_samples.py \
        --model "$MODEL" --out-dir out $SHARD_ARGS ;;
    geometry)
      uv run python sesgo/geometry/collect_geometry_samples.py \
        --model "$MODEL" --out-dir out --batch-size "$BATCH_SIZE" \
        --n-thinking "$N_THINKING" $SHARD_ARGS ;;
    *) echo "[fleet_model_run] unknown study: $study" >&2; return 1 ;;
  esac
}

for study in $STUDIES; do
  echo "[fleet_model_run] === $study (model=$MODEL, shard=$SHARD_INDEX/$SHARD_COUNT) ==="
  run_study "$study"
done

echo "[fleet_model_run] done. Slice: out/sesgo/<study>/$(basename "$MODEL")/"
