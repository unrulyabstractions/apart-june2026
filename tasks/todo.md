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

# Batching workstream: batched generation + parallel cloud fleet

## Goal
Batched within-box generation (vLLM CUDA fast path, HF batched verified locally)
+ maximally-parallel per-model cloud fleet with safe concurrent sync-back.

## Plan
- [ ] `src/inference/batched_generation.py` — reusable batched-HF primitives
      (LEFT-pad + attention mask; batched generate; batched teacher-forced forward).
- [ ] HuggingFaceBackend: honor an attention mask in `forward` / `run_with_cache`
      (padding currently contaminates logits in a batch). Single-sample unchanged.
- [ ] `src/inference/vllm_batched_backend.py` — vLLM CUDA backend (generation +
      teacher-forced scoring). Import guarded; raises clearly off-CUDA.
- [ ] ModelRunner: `generate_batch`, `compute_trajectories_batch` (mask-aware),
      `run_with_cache_batch` (LEFT-pad, per-sample position offsets).
- [ ] TernaryChoiceRunner: `choose3_batch` — ONE batched forward over 3N continuations.
- [ ] SesgoQuerier: `query_dataset` batches over samples; `--batch-size` plumbed.
- [ ] `--batch-size` on baseline/selection/divergence/geometry collect scripts.
- [ ] Geometry: batched capture; confirm `--n-thinking 0` short-circuits sampling.
- [ ] Cloud: model->GPU sizing map (BaseSchema), `cloud/fleet_launch.sh`,
      `cloud/fleet_run.sh`, `cloud/fleet_sync_back.sh`, sharding, self-destruct.
- [ ] `cloud/Dockerfile` + `cloud/prefetch_model_weights.py`; vLLM CUDA-only extra.
- [ ] VERIFY locally on Qwen3-0.6B: batched==single + faster. Cost/wall-clock table.

## Review (batching)
(filled in at end)
