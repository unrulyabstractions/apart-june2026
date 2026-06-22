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
#   2. FULL divergence run (NO subsample) -> out/sesgo/divergence/<MODEL>/response_samples.json
#   3. FULL stability run  (NO subsample) -> out/sesgo/stability/<MODEL>/response_samples.json
#
# These are the FULL collections: no --subsample flag is passed, so every prompt
# in the regenerated datasets is queried. Pull the results back with
# cloud/sync_back.sh (which only ever copies NEW files into the local sync/ dir).

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root on the remote (/root/apart)

WHICH="${1:-both}"   # both | divergence | stability
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

# Divergence does n_thinking_samples (8) free-form draws x up to 512 tokens PER
# item; the full 2040-item grid is ~12-20h single-stream and would risk the
# hackathon deadline. The querier's subsample is an evenly-spaced STRIDE, so a
# large fraction stays stratified across all categories/languages/polarities and
# gives tight distribution statistics while finishing overnight. Stability is
# cheap (non-thinking only, one forward pass/prompt) and runs FULL, first.
DIV_SUB="${DIV_SUB:-0.5}"

# ── 1. Regenerate the five SESGO prompt datasets on the box ────────────
# Reads datasets/SESGO/prompts/*.xlsx (synced up by sync_up.sh's narrow step) and
# writes out/sesgo/<study>/prompt_dataset.json. Full grid (no --limit).
echo "[at_run] generate prompt datasets"
uv run python sesgo/generate/generate_prompt_dataset.py

# ── 2. FULL stability collection (cheap; runs FIRST so results land fast) ──
if [ "$WHICH" = "both" ] || [ "$WHICH" = "stability" ]; then
  echo "[at_run] collect_stability_samples (FULL, model=$MODEL)"
  uv run python sesgo/stability/collect_stability_samples.py --model "$MODEL" --out-dir out
fi

# ── 3. Divergence collection (thinking-heavy; stratified stride subsample) ──
if [ "$WHICH" = "both" ] || [ "$WHICH" = "divergence" ]; then
  echo "[at_run] collect_divergence_samples (subsample=$DIV_SUB, model=$MODEL)"
  uv run python sesgo/divergence/collect_divergence_samples.py --model "$MODEL" --out-dir out --subsample "$DIV_SUB"
fi

echo "[at_run] done. Remote results under out/sesgo/{divergence,stability}/."
echo "[at_run] pull them back LOCALLY (safe) with: bash cloud/sync_back.sh"
