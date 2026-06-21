#!/usr/bin/env bash
#
# at_run.sh — run the FULL divergence + stability collections ON the Vast.ai box.
#
# Pushed up by sync_up.sh, then invoked remotely, e.g.:
#   bash cloud/at_vast.sh "bash cloud/at_run.sh"
#   bash cloud/at_vast.sh "bash cloud/at_run.sh divergence"   # just one study
#   bash cloud/at_vast.sh "bash cloud/at_run.sh stability"
#
# Steps (writes ONLY to the remote out/sesgo/...):
#   1. Regenerate all five prompt datasets from datasets/SESGO/prompts/*.xlsx
#      -> out/sesgo/{...}/prompt_dataset.json
#   2. FULL divergence run (NO subsample) -> out/sesgo/divergence/<MODEL>/samples.json
#   3. FULL stability run  (NO subsample) -> out/sesgo/stability/<MODEL>/samples.json
#
# These are the FULL collections: no --subsample flag is passed, so every prompt
# in the regenerated datasets is queried. Pull the results back with
# cloud/sync_back.sh (which only ever copies NEW files into the local sync/ dir).

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root on the remote (/root/apart)

WHICH="${1:-both}"   # both | divergence | stability
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

# ── 1. Regenerate the five SESGO prompt datasets on the box ────────────
# Reads datasets/SESGO/prompts/*.xlsx (synced up by sync_up.sh's narrow step) and
# writes out/sesgo/<study>/prompt_dataset.json. Full grid (no --limit).
echo "[at_run] generate prompt datasets"
uv run python sesgo/baseline/generate_prompt_dataset.py

# ── 2. FULL divergence collection (no subsample) ───────────────────────
if [ "$WHICH" = "both" ] || [ "$WHICH" = "divergence" ]; then
  echo "[at_run] collect_divergence_samples (FULL, model=$MODEL)"
  uv run python sesgo/baseline/collect_divergence_samples.py --model "$MODEL" --out-dir out
fi

# ── 3. FULL stability collection (no subsample) ────────────────────────
if [ "$WHICH" = "both" ] || [ "$WHICH" = "stability" ]; then
  echo "[at_run] collect_stability_samples (FULL, model=$MODEL)"
  uv run python sesgo/baseline/collect_stability_samples.py --model "$MODEL" --out-dir out
fi

echo "[at_run] done. Remote results under out/sesgo/{divergence,stability}/."
echo "[at_run] pull them back LOCALLY (safe) with: bash cloud/sync_back.sh"
