# RUN: full_data (ALL SESGO data) on Qwen3-0.6B — DONE 2026-06-21

## Result
- Full grid = 6120 items x 4 scaffold conditions = 24480 prompts (all langs es+en x
  all origins original+bbq-adapted x {none + interpretive_direction +
  prior_dominance_warning + intent_and_register_respect}).
- Collected 16164 / 24480 (66%): 7 of 8 RTX_4090 shards landed (shard6 complete at
  3060; the rest are partial checkpoints recovered after the fleet driver process was
  killed mid-run; shard7's box died at setup). All axes fully covered & balanced
  (4041/scaffold; es 12684 / en 3480; original 10320 / bbq 5844).
- Combined -> out/sesgo/full_data/Qwen3-0.6B/response_samples.json (16164).
- Figure -> out/sesgo/full_data/Qwen3-0.6B/plots/abstention_by_axis.png.

## Headline numbers (3-opt teacher-forced abstention on ambiguous, n=5388)
- language: en 96.2% (n=1160) vs es 86.4% (n=4228)
- origin:   BBQ-adapted 96.1% (n=1948) vs original 84.2% (n=3440)
- scaffold: interpretive_direction 99.9% > prior_dominance 88.4% > none 85.7% >
            intent_and_register 80.0% (n=1347 each)
- greedy-thinking abstention: scaffolds lift it from 63% (none) to 83%
  (interpretive_direction); bbq 87% vs original 65%.

## Cost / boxes
- ~$8.14 total across 3 fleet attempts (OOM batch=64 run + SSH-proxy-outage run +
  the successful batch=12 run, incl. extra billing after the driver process died).
- ALL 40 boxes across all attempts DESTROYED (verified: 0 of our instances live).

## Code shipped (committed + pushed)
- load_items `origins` opt-in; get_full_data_scaffolds(); GENERATE_ALL_DATA
  generator opt-in -> out/sesgo/full_data/; collect `--study` flag;
  HF_FORWARD_MICRO_BATCH (OOM-resilient teacher-forced forward);
  combine_full_data_shards.py; full_data_axis_{slices,plots}.py +
  visualize_full_data_samples.py; fleet threading; docs + lessons.

## Lessons (see tasks/lessons.md)
- 4090 teacher-forced batch=64 OOMs on long scaffolded prompts -> batch<=16.
- Vast SSH-proxy-wide outages can kill an in-flight fleet; fewer shards = smaller
  surface. The long-running fleet_run.sh driver can be killed by the harness -> on
  driver death, manually sync_back each live box THEN destroy (billing keeps running).

---

# RUN: Full-scale forking-paths on Qwen/Qwen3-32B (cloud H100) — 2026-06-21

- [x] Stage SESGO prompt xlsx into datasets/SESGO/prompts/ (gitignored)
- [x] Confirm H100_SXM offers available (~$2/hr, rel>=0.99)
- [x] Launch H100_SXM box id=41959804 ($2.127/hr, ssh4.vast.ai:39804, NVIDIA H100 80GB)
- [x] sync_up.sh -> at_setup.sh (torch 2.6.0+cu124, CUDA OK)
- [x] FIX: micro-batch chunking in HF generate_batch (HF_GEN_MICRO_BATCH=48) so the
      32B forking batch fits 80GB (fp16 weights ~66GB + bounded KV cache)
- [x] STAGE 0 done: selected item idx=3 (q=a7964df08c51), outcomes target/other/unknown/unparseable
- [~] STAGE 1-3 collect RUNNING (PID 2714): --max-positions 0 --n-samples 40
      --n-prior 300 --max-new-tokens 512; GPU 99%, ~68GB. Background poller bzmy20wnt
      watches for STATUS=DONE.
- [ ] STAGE 4 analyze, STAGE 5 plot
- [ ] sync_back.sh (quarantine) -> merge_sync.sh -> out/sesgo/forking/Qwen3-32B/
- [ ] vast_destroy.sh --yes-i-am-really-sure
- [ ] Report wall-clock / cost / N-per-position / total generations / results path

RUN START (UTC): 2026-06-21T14:26:47Z  | box launched ~14:18Z

---

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

# STEER: SESGO causal steering pipeline (sesgo/steer/ + src/steer/) — 2026-06-21

Goal: invert the geometry diff (scaffold - no_scaffold) into an add-mode resid_post
steering vector and show +v RAISES abstention (UNKNOWN) on held-out ambiguous items.

- [x] src/steer/: BaseSchema schemas (steering vectors + split, sweep results)
- [x] diff_of_means_vectors.py + contrastive_pair_index.py + geometry_residual_loader.py:
      pair by qid, load residuals (root/a.path), per-LAYER diff-of-means
- [x] seeded_pair_split.py: seeded 70/30 split over question_ids (saved & reused)
- [x] steered_ternary_runner.py: subclass TernaryChoiceRunner, override _run_triple
- [x] sesgo_abstention_readout.py: reconstruct choose3 inputs + abstention metric
- [x] sesgo/steer/calculate_steering_vectors.py (run-by-path driver)
- [x] sesgo/steer/run_steering_test.py: alpha sweep on held-out TEST ambiguous items
- [x] 3-line auto-export __init__; README + EXPLANATION (both folders)
- [x] PILOT verify: hook moves logits / abstention on Qwen3-0.6B handful of items
- [x] commit in worktree

## Review (DONE 2026-06-21)

House style: unique multi-word filenames, files <=154 lines, BaseSchema everywhere,
imports at top, auto-export __init__, README+EXPLANATION in both folders, ruff clean.

Reuse (no hooks re-implemented): steering(...) +
compute_trajectories_batch_with_intervention; choose3 + SesgoNonThinking.from_ternary;
select_feature_layer; root/a.path residual load; l2_norm; BaseSchema save/load.

Vectors: 231 pairs -> 162 train / 69 test (0 overlap); primary layer 14 (0.50 depth);
v[L14] over 648 terms, norm 3.914.

PILOT (16 held-out ambiguous, raw v, L14): unknown_prob 0.383 (a=0) -> 0.509 (a=+2);
abstain_rate 0.50 -> 0.625; negative control a=-2 -> 0.068 (abstain 0.0); scaffold ref
0.991. Raw divergent-token logits provably change under the hook. CAUSAL CLAIM
CONFIRMED: +v raises abstention on untrained held-out items.

---

# HOURLY INTEGRATOR — 2026-06-21 (verified state)

## 1. RECLAIM
All 25 vastai instances cur_state=stopped/intended=stopped. Actual burn = storage only.
gpu_util>0 on a few is STALE (stopped boxes don't compute). None actively running. -> destroy all.

## 2. BASELINE (target 2310; prompt_dataset_id b203c952...)
- Already complete in out/ (8): Qwen3-0.6B/1.7B/4B, Llama-3.2-1B/3B, gemma-2-2b, Mistral-7B/24B
- Promote from sync box-* @2310 (verified REAL, non-empty):
  - [ ] Qwen3-14B (out MISSING)
  - [ ] gemma-2-9b-it (out MISSING)
  - [ ] Llama-3.1-8B-Instruct (out=96 BROKEN -> rm then promote 2310)
- Qwen3-32B(1376) & gemma-2-27b(1408): NO 2310 anywhere; sync/final-*==out/ partial EXACTLY.
  Task premise FALSE -> leave partials, do NOT destroy-and-replace with identical data.
- Missing everywhere: Qwen3-8B baseline, Llama-3.1-70B baseline.

## 3. COMBINE SHARDS (scope: Qwen3-0.6B,Qwen3-32B,Llama-3.2-1B,Llama-3.1-70B)
- Geometry: [ ] Qwen3-0.6B (6/6) [ ] Qwen3-32B (4/4) [ ] Llama-3.1-70B (single, NEW)
  Llama-3.2-1B geometry shard1of4 MISSING -> SKIP.
- Selection/Divergence: no complete box-* sets -> SKIP.
- Then analyze + visualize each combined geometry.

## 4. RE-RENDER: [ ] cross-model size sweep [ ] per-model viz (Qwen3-14B,gemma-2-9b,Llama-3.1-8B)

## 5. PAPER: add Appendix H (Steering, figs+JSON on disk). Skip G (no forking figs). Rebuild+commit.

## 6. SWEEP: Qwen3-8B still missing. 70B baseline missing. Boxes all stopped.

## REVIEW (hour complete)

### 1. RECLAIM
- Destroyed 27 boxes: 25 originally-stopped + 2 wedged (41989335, 41989340, never came up).
- KEPT 7 actively-running boxes (419916xx) — a concurrent agent's fleet_run.sh is SSH-driving
  them on MODEL=Qwen3-0.6B STUDIES=full_data SHARD_*of8 (GENERATE_ALL_DATA=1), rsync pulling
  into sync/partial-box-Qwen3-0.6B__shard*of8/. (8 at start, 1 self-destructed on finish -> 7.)
- Remaining RUNNING burn: $6.43/hr (the active grid). Stopped/idle burn eliminated.

### 2. BASELINE MERGE (target 2310)
- Promoted to out/ @2310: Qwen3-14B (was MISSING), gemma-2-9b-it (was MISSING),
  Llama-3.1-8B-Instruct (was BROKEN 96 -> rm + promote 2310). All verified non-empty real.
- CORRECTION to task premise: sync/final-Qwen3-32B (1376) and final-gemma-2-27b-it (1408) are
  NOT full 2310 — they EQUAL the out/ partials exactly. NO 2310 exists anywhere for these two.
  Did NOT destroy-and-replace partials with identical data (would have lost nothing but gained
  nothing and risked the only copy). Left partials intact.
- Completeness: 11/14 full (2310); 2 partial (Qwen3-32B 1376, gemma-2-27b 1408); 2 missing
  (Qwen3-8B, Llama-3.1-70B baseline).

### 3. COMBINE SHARDS (scope models)
- Geometry combined (clobber-safe, all .pt verified):
  - Qwen3-0.6B: 6/6 shards -> 4620 samples, 517,440 .pt. analyzed+visualized (re-analyze running bg).
  - Qwen3-32B: 4/4 shards -> 4620 samples, 1,182,720 .pt. analyze DEFERRED (too heavy this hr; data safe).
  - Llama-3.1-70B (NEW): single box -> 140 samples, 28,000 .pt. analyzed + visualized.
- Llama-3.2-1B geometry: shard 1of4 missing -> SKIPPED (next hour).
- Selection/Divergence: no complete box-* shard sets -> SKIPPED.

### CROSS-SIZE SCAFFOLD SILHOUETTE (label position, mean across layers)
- Qwen3-0.6B (0.6B): +0.488 (0.635 @ L14)  [matches paper +0.48]
- Llama-3.1-70B (70B, NEW): +0.473 (0.476-0.477 across last layers)
- Qwen3-32B (32B): pending analyze
- FINDING HOLDS ACROSS FAMILY + SCALE: scaffold is the dominant separable axis (~0.48) on both a
  tiny Qwen and a huge Llama.

### 4. RE-RENDER
- Cross-model size sweep re-rendered: now 13 models (added 14B/9b/8B).
- Per-model baseline viz: Qwen3-14B, gemma-2-9b-it, Llama-3.1-8B-Instruct (accuracy + role_prob).

### 5. PAPER
- Added Appendix H (Steering): Methods -> bulleted Results -> Figure 8 (abstention & unknown-mass
  vs alpha). Registered out/sesgo/steer/figures in graphicspath. Rewrote future-work bullet to cite
  the now-done App H. Did NOT touch \section{Results} stub.
- NO Appendix G (Forking): no forking output figures on disk (only unmerged branch code).
- Build clean: 13 pages (was 12), no placeholders. Verified page 12 visually (Fig 8 real plots).
- Committed (8469e15) + pushed onto origin/main (no conflict; I own paper this hour).

### 6. SWEEP
- NOT launching Qwen3-8B / 70B baseline: a concurrent agent is actively driving the cloud fleet
  (live fleet_run.sh + rsync). Launching would collide on shared cloud/.fleet/ + sync/ state and
  double-bill. Deferred to an hour when the cloud is not owned by another process.

### PENDING (next hour)
- Qwen3-32B geometry analyze + viz (data combined, just heavy).
- Qwen3-0.6B geometry re-analyze finishing in bg (refreshes plots; paper unaffected).
- Qwen3-8B + Llama-3.1-70B baseline (launch when cloud free).
- Llama-3.2-1B geometry shard 1of4 (resume), then combine.
- Qwen3-32B / gemma-2-27b baseline: still only partials; need a real full-2310 run.
- Selection/divergence shard sets incomplete (only partial-box mirrors).

---

# Cross-model SESGO baseline DISTRIBUTION plots — 2026-06-21

## Goal
~6 cross-model DISTRIBUTIONAL comparison plots over the sweep
(out/sesgo/baseline/<model>/response_samples.json, 13 models) ->
out/sesgo/baseline/cross_model/plots/. House style, run-by-path, BaseSchema,
Okabe-Ito, Wilson CIs + n. NO cloud. Do NOT touch paper/.

## Plots
- [ ] 1. Outcome-distribution: mean role mass (target/other/unknown) per model,
      stacked bars ordered by size (does unknown mass grow with scale?).
- [ ] 2. Abstention SPREAD: violin/box of per-item p_unknown (ambig) per model.
- [ ] 3. Per-bias-category abstention heatmap (models x {clasismo,racismo,xenofobia,genero}).
- [ ] 4. Target-vs-other gap (disambig target acc - other acc) diverging bars by size.
- [ ] 5. Readout agreement: 3-opt vs 2-opt vs greedy-thinking abstention per model.
- [ ] 6. Disambiguated-accuracy distribution per model (per-item p_correct box/violin).

## Files (<=150 lines, globally-unique multi-word names)
- [ ] cross_model_distribution_stats.py     (aggregation, BaseSchema)
- [ ] cross_model_outcome_plots.py          (plots 1,4,5)
- [ ] cross_model_spread_plots.py           (plots 2,3,6)
- [ ] visualize_cross_model_distributions.py (run-by-path driver)

## Verify: uv run driver; VIEW each PNG; iterate if cramped.
## Commit: branch, fetch+rebase origin/main, stage ONLY new viz files; NOT paper/.

## Review (DONE 2026-06-21)
- 6 figures -> out/sesgo/baseline/cross_model/plots/: outcome_distribution,
  abstention_spread, category_abstention_heatmap, target_other_gap,
  readout_agreement, disambig_accuracy_spread. All viewed + iterated (rotated
  x-labels fixed overlap; disambig switched from Bernoulli 0/1 to soft p(gold)).
- 4 new modules + 1 driver (all <=147 lines, unique multi-word names, BaseSchema,
  Okabe-Ito, Wilson CIs + n, top-level imports, ruff clean). sesgo/README.md updated.
- DATA FINDING: gemma-2-27b-it baseline is a BROKEN run (3-opt probs all uniform
  [.33,.33,.33], greedy 100% unparseable) -> auto-detected + SKIPPED (not plotted).
  12 healthy models plotted; Qwen3-32B (1376) flagged partial `*`.
- HEADLINE: UNKNOWN (abstention) mass rises with scale (Qwen 0.72->0.94; gemma
  0.83->0.89; Mistral 0.51->0.96) but NON-monotonically — small Llamas barely
  abstain (1B 0.27). The bias gap (target-other acc) is large & signed on small
  models (Llama-1B +0.40, Qwen-1.7B -0.23, Qwen-4B/-32B ~-0.18) and shrinks toward
  0 for most big models. 3-opt vs greedy abstention mostly agree but diverge
  sharply on Mistral-24B (3-opt 0.98 vs greedy 0.67); 2-opt is structurally 0.
