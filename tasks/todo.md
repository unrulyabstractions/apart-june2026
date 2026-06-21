# Multimodel workstream: run SESGO with Llama, Gemma, Mistral (not just Qwen)

## Goal
Make baseline / geometry / divergence (and stability/selection by extension)
collect cleanly with Llama, Gemma, Mistral in addition to Qwen3-0.6B.

## Root causes (from scan, verified)
1. `SesgoQuerier._load_model` builds `TernaryChoiceRunner(model_name)` with no
   backend -> defaults to MLX on Apple Silicon. MLX cannot reliably load
   Llama/Gemma/Mistral. Geometry already forces HF; baseline/divergence/
   stability/selection go through the querier and inherit the MLX default.
2. Geometry `find_positions` hardcodes Qwen tokens (`<|im_start|>`, `<think>`,
   `</think>`). Llama/Gemma/Mistral lack these -> only the "answer" position is
   found. Each family has a SINGLE-TOKEN assistant-turn marker:
     Qwen  `<|im_start|>`   Llama `<|start_header_id|>`
     Gemma `<start_of_turn>` Mistral `[/INST]`
   and only Qwen has the think markers.
3. `skip_thinking_prefix` already returns "" for non-reasoning models (no-op) —
   verified, no change needed.

## Plan
- [x] Inspect runner / backend / querier / collect scripts / parsing.
- [x] Empirically confirm each model's turn-boundary + think tokens (single-token).
- [x] Confirm gated repos accessible with HF_TOKEN (Llama/Gemma/Mistral all OK).
- [ ] Add model-aware structural markers at the RUNNER layer
      (`chat_template_markers.py` + `ModelRunner.structural_markers`).
- [ ] Make geometry `find_positions` consume the runner's markers (model-agnostic).
- [ ] Force HuggingFace backend in `SesgoQuerier._load_model` so MLX-incompatible
      models load reliably (config-driven, defaulting to HF for these studies).
- [ ] Generate prompt datasets (small) into worktree out/.
- [ ] VERIFY minimal collect for EACH of Llama/Gemma/Mistral:
      baseline --subsample 0.004 ; geometry --subsample 0.004 --n-thinking 1 ;
      divergence --subsample 0.004 --n-thinking 2.
- [ ] Update docs touched; commit with clear messages.

## Model repo ids (chosen)
- Llama:   meta-llama/Llama-3.2-1B-Instruct (gated, accessible via HF_TOKEN)
- Gemma:   google/gemma-2-2b-it (gated, accessible via HF_TOKEN)
- Mistral: mistralai/Mistral-7B-Instruct-v0.3 (gated, accessible via HF_TOKEN)
  (mirrors unsloth/* confirmed working as fallback)

## Review

### Changes
- NEW `src/inference/chat_template_markers.py`: `ChatTemplateMarkers`
  (BaseSchema) + `structural_markers_for(name)` — per-family assistant-turn token
  and (reasoning-only) think markers. ChatML default for unknown models.
- `src/inference/model_runner.py`:
  - `structural_markers` property (surfaces the markers on any runner).
  - Extracted `detect_backend_for_name` / `is_cloud_api_name` module-level helpers
    (decide cloud-vs-local before a runner exists); `_detect_backend` delegates.
  - BUG FIX: `google/` HF org prefix no longer misroutes to the Gemini API
    (only real `gemini*` names do), so `google/gemma-2-2b-it` loads locally.
- `sesgo/geometry/collect_geometry_samples.py`: `find_positions` consumes
  `runner.structural_markers` instead of hardcoding Qwen tokens; new
  `_marker_position` helper (empty marker -> skipped). Docstrings updated.
- `src/datasets/sesgo_eval/sesgo_querier.py`: `_load_model` pins the HF backend
  for local models (MLX can't load Llama/Gemma/Mistral), cloud names auto-detect.
- Docs: `src/inference/README.md`, `src/datasets/sesgo_eval/README.md`.

### Verification (subsample 0.004; all wrote samples.json + activations)
| model | baseline NT abst. | divergence NT/Th abst. (ent) | geometry positions |
|-------|------|------|------|
| Llama-3.2-1B-Instruct | 100% (n=1) | 100%/100% (0.693) | turn+answer (think_* N/A) |
| gemma-2-2b-it | 100% (n=1) | 100%/100% (0.693) | turn+answer (think_* N/A) |
| Mistral-7B-Instruct-v0.3 | 100% (n=1) | 100%/100% (0.000) | turn+answer (think_* N/A) |
| Qwen3-0.6B (regression) | — | — | turn+think_open+think_close+answer |

Turn token verified per family: `<|start_header_id|>` (Llama),
`<start_of_turn>` (Gemma), `[/INST]` (Mistral), `<|im_start|>` (Qwen).

### Caveats
- Mistral 7B is the slowest (largest); all three families load from GATED repos
  via HF_TOKEN (mirrors `unsloth/*` confirmed as fallback, not needed).
- Gemma's greedy `answer` token can be whitespace (`\n\n`) before the label —
  faithful capture, just a model-specific quirk worth noting downstream.
- A pre-existing verbose debug print lives in `choice_utils.py`
  (`encode_into_trajectory_ids`); not in scope, left untouched.

---

# Workstream: thread ALL per-sample labels + funnel geometry viz through every axis

## Feature 1 — thread every per-sample label to records the viz reads
- [ ] Add `origin_label` helper + `target_text`/`other_text` access on SesgoItem (bbq already present).
- [ ] Add fields to SesgoPromptSample: bbq, target_identity, other_identity, label_style(present), gold_label(present), scaffold_id(present). Set in sesgo_prompt_generator from item.
- [ ] Add fields to SesgoSample: bbq, target_identity, other_identity. Set in sesgo_querier.query_sample from prompt sample (MINIMAL/LOCALIZED — worktree edits this file).
- [ ] Add fields to GeometrySample: bbq, target_identity, other_identity, label_style. Set in collect_geometry_samples (MINIMAL — worktree edits this file).
- [ ] origin label helper: False->"original", True->"BBQ-adapted".

## Feature 2 — geometry viz funnels representation through ALL axes
- [ ] analyze_geometry: carry every per-sample label into projections.json rows + per-axis silhouette/separation for every axis (incl high-cardinality target/other identity).
- [ ] visualize_geometry_samples: PCA scatter colored by EACH axis (scaffold, origin, language, bias_category, question_polarity, target_identity, other_identity, gold_label, label_style). One file per axis `pca_by_<axis>.png` at representative layer + answer position; cap high-cardinality at top-K + "other".
- [ ] Keep centroid-shift + explained-variance plots.

## Verify (MANDATORY visual)
- [ ] Regenerate prompts, re-collect geometry (subsample 0.006) + baseline (0.02), re-analyze, re-viz.
- [ ] READ every new PNG as an image; confirm legends legible for every axis. Iterate.
- [ ] py_compile all changed files.

## Review (this workstream) — DONE
- Feature 1: bbq + target_identity + other_identity now on SesgoPromptSample (set in
  sesgo_prompt_generator from item), SesgoSample (set in sesgo_querier query_sample),
  and GeometrySample (+label_style; set in collect_geometry_samples). origin_label
  helper added to sesgo_item.py (False->"original", True->"BBQ-adapted").
- Feature 2: analyze_geometry now carries ALL per-sample axes into projections.json
  rows + per-axis silhouette/separation for all 8 non-scaffold axes. viz renders
  pca_by_<axis>.png for 9 axes + pca_axes_grid.png; high-cardinality identity axes
  capped at top-8 + (other) with caption; per-group centroid anchoring keeps minority
  groups in-frame. Centroid-shift + EV plots kept.
- Verified visually: read pca_axes_grid, pca_by_{scaffold_id,origin,language,
  bias_category,question_polarity,target_identity,other_identity,gold_label}.png —
  every legend legible; fixed language panel that was clipping the en group off-view.
- Worktree-shared files (sesgo_querier.py, collect_geometry_samples.py) touched with
  minimal, localized additive edits only.
- All changed .py files py_compile OK. Docs updated (sesgo + sesgo_eval + prompt).
