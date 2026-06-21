# Forking-Paths Dynamics Pipeline (sesgo/forking/ + src/dynamics/forking_paths/)

Branch: forking-paths-dynamics. ONE ambiguous SESGO item; track per-token outcome
distribution O_t over a thinking trajectory; detect the forking token; compute the
dynamic states (pull/drift/potential) + diversity + survival; plot.

## Reuse (do NOT reimplement)
- SESGO load: src.datasets.sesgo.load_items, SesgoPromptDatasetGenerator
- thinking gen + parsing: parse_chosen_label, TernaryChoiceRunner.generate/generate_batch
- backend: ModelRunner (HF local pilot, vLLM cloud generate_batch)
- math: l2_distance/l2_norm; structure_aware orientation/deviance/core_entropy/
  expected_deviance/deviance_variance; shannon_entropy; probs_to_logprobs
- io: load_json/save_json_atomic, shard_out_dir, BaseSchema

## BaseSchema types (src/dynamics/forking_paths/)
- [x] ForkOutcomeSet, OutcomeHistogram (O_t), AltTokenRollouts, ForkPosition,
      ForkingTrajectory, ChangePointResult, DynamicStatesSeries, DiversitySeries,
      SurvivalSeries

## Logic modules (src/dynamics/forking_paths/, <=150 lines, unique names)
- [x] forking_outcome_mapping.py, outcome_histogram_builder.py,
      forking_path_capture.py, semantic_drift_series.py, bayesian_change_point.py,
      forking_dynamic_states.py, forking_diversity_series.py,
      forking_survival_analysis.py

## Run-by-path drivers (sesgo/forking/)
- [x] select_forking_item.py, collect_forking_rollouts.py,
      analyze_forking_dynamics.py, plot_forking_dynamics.py, forking_plot_styles.py

## Docs
- [x] sesgo/forking/{README,EXPLANATION}.md, src/dynamics/forking_paths/{README,EXPLANATION}.md

## Verify
- [x] uv sync ; TINY local pilot (Qwen3-0.6B, small N) end-to-end ; commit

## Review (forking-paths)

### What shipped
- NEW package `src/dynamics/forking_paths/` (15 logic modules + types, all <=150
  lines, globally-unique names, 3-line auto-export __init__): outcome set/mapping,
  o_{t,w}/o_t histograms, top-K alternates, base-path branch plan, batched capture,
  semantic drift, Bayesian RJ-MCMC change-point (self-contained BEAST replacement;
  Rbeast not a dep), pull/drift/potential states, diversity, survival, item
  selection, analysis bundle. README + EXPLANATION.
- NEW `sesgo/forking/` drivers (run-by-path): select_forking_item,
  collect_forking_rollouts, analyze_forking_dynamics, plot_forking_dynamics, plus
  forking_plot_styles + forking_item_io. README + EXPLANATION.
- `ModelRunner.continue_from_text_batch` — raw-continuation batched primitive
  (no chat-template re-wrap) so every (t,w) branch rides vLLM continuous batching
  on the cloud / HF batched decode locally. inference/README updated.

### Reuse (no reimplementation)
parse_chosen_label; src.dynamics pull/drift/potential/normalized_norm;
l2_distance/normalize/shannon_entropy/probs_to_logprobs/expected_deviance/
deviance_variance; load_items + SesgoPromptDatasetGenerator; save_json_atomic;
shard_out_dir; BaseSchema.

### CPD calibration
NIG-marginal segment evidence (b0=0.005) + geometric prior (penalty 2.0): clean
outcome step -> BF~18 at the true index; flat control -> p(m=0)~0.95.

### Chosen sample (pilot, Qwen3-0.6B, gender/es)
sample_idx=6 (q=28fa1cf1...), the highest pilot outcome-entropy item (1.099 nats);
capture gave o_0=[0,.25,.25,.5] -> o_T=[0,1,0,0] (outcome collapse onto OTHER).

### Verification
All 5 stages ran end-to-end on the tiny pilot (6 branched positions, S=4, N=4);
forking_dynamics.png rendered. ruff clean on all new+modified files. Pre-existing
unrelated test_imports failure (src.datasets 'other'/'generate_dataset') untouched.

---

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

### What shipped
- Within-box batching: `batched_padding_helpers.py` (left-pad + mask + offsets),
  HF backend mask-aware `forward`/`run_with_cache` + `generate_batch`, runner
  `generate_batch`/`compute_trajectories_batch`/`run_with_cache_batch`,
  `choose3_batch`, `query_chunk`, `--batch-size` on all 5 collects. Geometry
  batched capture (shared `geometry_capture_helpers.py`); `--n-thinking 0`
  short-circuits thinking (verified).
- vLLM CUDA backend (`vllm_batched_backend.py` + `vllm_option_scoring.py`),
  `ModelBackend.VLLM`, `_init_vllm`, `cloud` extra in pyproject. Import-guarded;
  raises off-CUDA. NOT run on the Mac (per constraint).
- Parallel fleet: `fleet_sizing.py` (sizing map + shard plan), `fleet_launch.sh`
  (concurrent create), `fleet_run.sh` (concurrent drive + self-destruct),
  `fleet_model_run.sh`, `fleet_destroy.sh`. Sharding: `shard_slicing.py` +
  `shard_output_paths.py` + `--shard-index/--shard-count` on all collects.
  Concurrent-safe sync-back: `SYNC_SUBDIR` per-box quarantine, merge strips the
  box prefix; `--ignore-existing`/no-`--delete` preserved.
- Custom image: `cloud/Dockerfile` (torch+vLLM+deps+weights) +
  `prefetch_model_weights.py`. Implement+document only; not built/pushed.

### Local verification (Qwen3-0.6B, MPS)
- baseline bs1 vs bs8: max|prob|=1.2e-4, max|logit|=4.7e-2, 0 greedy-label
  mismatches (10/10 same answer).
- selection bs1 vs bs8 (thinking): max|prob|=4.4e-5, 0 prediction mismatches,
  42.8s -> 17.0s (2.5x).
- geometry bs1 vs bs8 (--n-thinking 0): 0 position/token mismatches, mean
  cos-sim >= 0.999996 across all 4 structural positions, 19.1s -> 12.6s (1.5x);
  80 activation files both paths.
- shard tiling: 3-way exact/disjoint/complete; sharded baseline loaded 5/10,
  wrote shard_0_of_2/.
- Full test suite passes (432 tests; only unrelated mental_risk fixture-missing
  failures). ruff clean on all touched files.
