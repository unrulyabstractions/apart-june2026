#!/usr/bin/env bash
#
# launch_stability_sweep.sh — fan the FULL stability sweep across the cloud, ONE box per
# (model, mode-group). Each box runs cloud/run_one_stability_box.sh (HF/CUDA backend, the
# 6,930-prompt orthogonal dataset, self-destruct trap) in the BACKGROUND and pulls its
# slice into the gitignored sync/<TAG>/ quarantine.
#
# Usage:  bash cloud/launch_stability_sweep.sh <tier>
#   tier = small | mid | big | llamabig   (or 'list' to just print the manifest)
#
# Tiers map to GPU classes (vast gpu_name filters). Llama is gated -> HF_TOKEN required
# (must be exported); gemma-4 / Qwen3.5 / Ministral-3 are ungated.
#
# Manifest row:  tag | model | bare | modes | gpu | num_gpus | maxprice | disk | gated
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TIER="${1:-list}"

# ── small: <=9B on a single RTX 4090 (24GB) ──────────────────────────────────────
SMALL=(
  "q08b|Qwen/Qwen3.5-0.8B|Qwen3.5-0.8B|nonthinking thinking|RTX_4090|1|0.80|60|no"
  "l1b|meta-llama/Llama-3.2-1B-Instruct|Llama-3.2-1B-Instruct|nonthinking|RTX_4090|1|0.80|60|yes"
  "gE2B|google/gemma-4-E2B-it|gemma-4-E2B-it|nonthinking|RTX_4090|1|0.80|60|no"
  "gE4B|google/gemma-4-E4B-it|gemma-4-E4B-it|nonthinking|RTX_4090|1|0.80|70|no"
  "q2b|Qwen/Qwen3.5-2B|Qwen3.5-2B|nonthinking thinking|RTX_4090|1|0.80|70|no"
  "q4b|Qwen/Qwen3.5-4B|Qwen3.5-4B|nonthinking thinking|RTX_4090|1|0.80|80|no"
  "q9b|Qwen/Qwen3.5-9B|Qwen3.5-9B|nonthinking thinking|RTX_4090|1|0.90|90|no"
  "l3b|meta-llama/Llama-3.2-3B-Instruct|Llama-3.2-3B-Instruct|nonthinking|RTX_4090|1|0.80|70|yes"
  "l8b|meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct|nonthinking|RTX_4090|1|0.90|90|yes"
  "min3i|mistralai/Ministral-3-3B-Instruct-2512|Ministral-3-3B-Instruct-2512|nonthinking|RTX_4090|1|0.80|70|no"
  "min3r|mistralai/Ministral-3-3B-Reasoning-2512|Ministral-3-3B-Reasoning-2512|thinking|RTX_4090|1|0.80|70|no"
  "min8i|mistralai/Ministral-3-8B-Instruct-2512|Ministral-3-8B-Instruct-2512|nonthinking|RTX_4090|1|0.90|90|no"
  "min8r|mistralai/Ministral-3-8B-Reasoning-2512|Ministral-3-8B-Reasoning-2512|thinking|RTX_4090|1|0.90|90|no"
)
# ── mid: 12-14B on a 48GB card (A6000) ───────────────────────────────────────────
MID=(
  "g12b|google/gemma-4-12B-it|gemma-4-12B-it|nonthinking|RTX_A6000|1|1.30|120|no"
  "min14i|mistralai/Ministral-3-14B-Instruct-2512|Ministral-3-14B-Instruct-2512|nonthinking|RTX_A6000|1|1.30|120|no"
  "min14r|mistralai/Ministral-3-14B-Reasoning-2512|Ministral-3-14B-Reasoning-2512|thinking|RTX_A6000|1|1.30|120|no"
)
# ── big: 27-31B on an 80GB A100 ──────────────────────────────────────────────────
BIG=(
  "g31b|google/gemma-4-31B-it|gemma-4-31B-it|nonthinking|A100_SXM4|1|2.50|160|no"
  "q27b|Qwen/Qwen3.5-27B|Qwen3.5-27B|nonthinking thinking|A100_SXM4|1|2.50|160|no"
)
# ── llamabig: 70B across 2x 80GB (HF device_map=auto) ────────────────────────────
LLAMABIG=(
  "l70b|meta-llama/Llama-3.3-70B-Instruct|Llama-3.3-70B-Instruct|nonthinking|A100_SXM4|2|5.00|260|yes"
)

case "$TIER" in
  small) ROWS=("${SMALL[@]}") ;;
  mid)   ROWS=("${MID[@]}") ;;
  big)   ROWS=("${BIG[@]}") ;;
  llamabig) ROWS=("${LLAMABIG[@]}") ;;
  list)  ROWS=("${SMALL[@]}" "${MID[@]}" "${BIG[@]}" "${LLAMABIG[@]}") ;;
  *) echo "unknown tier '$TIER' (small|mid|big|llamabig|list)"; exit 1 ;;
esac

for row in "${ROWS[@]}"; do
  IFS='|' read -r tag model bare modes gpu ngpu price disk gated <<< "$row"
  if [ "$TIER" = list ]; then printf '  %-7s %-44s modes=%-22s %s x%s\n' "$tag" "$model" "$modes" "$gpu" "$ngpu"; continue; fi
  if [ "$gated" = yes ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "  SKIP $tag ($model) — gated but HF_TOKEN not exported"; continue; fi
  tokenenv=""; [ "$gated" = yes ] && tokenenv="HF_TOKEN=$HF_TOKEN"
  # Mistral Ministral-3 ships finegrained-fp8 weights -> needs the pinned kernels package.
  kernelsenv=""; case "$model" in *[Mm]inistral*) kernelsenv="INSTALL_KERNELS=1";; esac
  env $tokenenv $kernelsenv MODEL="$model" BARE_MODEL="$bare" MODES="$modes" \
      GPU_NAME="$gpu" NUM_GPUS="$ngpu" MAX_PRICE="$price" DISK="$disk" TAG="$tag" \
      nohup bash "$HERE/run_one_stability_box.sh" > "$HERE/.stab_${tag}.driver.log" 2>&1 &
  echo "  launched tag=$tag model=$model modes='$modes' gpu=${gpu}x${ngpu} pid=$!"
  sleep 4
done
[ "$TIER" = list ] || echo "tier '$TIER' launched; logs: cloud/.stab_<tag>.driver.log"
