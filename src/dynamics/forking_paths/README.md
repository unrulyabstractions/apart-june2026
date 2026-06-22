# `src/dynamics/forking_paths/` — Forking-Paths O_t dynamics

Reusable, model-agnostic logic for the forking-paths analysis of an autoregressive
generation: estimate the per-token **outcome distribution** `O_t`, detect the
**forking token**, and derive the dynamic states / diversity / survival series.
All I/O and model orchestration lives in the run-by-path drivers under
`sesgo/forking/`; this package is pure computation over typed `BaseSchema` records.

Implements the union of two methods:
- **Forking Paths in Neural Text Generation** (Bigelow et al., arXiv:2412.07961):
  Monte-Carlo branching at each token, outcome histograms `o_{t,w}`/`o_t`,
  Bayesian change-point detection, and survival analysis.
- **The Homogenization Problem in LLMs**, App. H (Rios-Sialer, arXiv:2601.06116):
  the same `O_t` is the system barycenter (pull), with drift / potential states,
  forking magnitude `Δ_t`, and the Sec. 7 diversity scores.

## Modules

| File | Role |
|------|------|
| `forking_outcome_set.py` | `ForkOutcomeSet`: ordered categorical outcomes (SESGO roles + `unparseable`) and the one-hot R-vector. |
| `forking_path_types.py` | `BaseSchema` records: `AltTokenRollouts`, `ForkPosition`, `ForkingTrajectory`, `ChangePointResult`, `DynamicStatesSeries`, `DiversitySeries`, `SurvivalSeries`. |
| `forking_rollout_dump.py` | `BaseSchema` records for the per-position RAW dump: `RolloutDumpEntry` (one continuation's raw text + parsed label + token info), `ForkingPositionDump`. |
| `forking_position_dump_writer.py` | `build_position_dump` + `write_position_dump`: assemble one position's raw rollouts and atomically write `pos_<NNN>.json` (crash-safe, incremental). |
| `forking_outcome_mapping.py` | `R(rollout) -> outcome label` (reuses the SESGO `parse_chosen_label`). |
| `outcome_histogram_builder.py` | `o_{t,w}` (Eq. 1) and `o_t` (Eq. 2) prob-weighted histograms. |
| `forking_top_k_tokens.py` | per-position top-K alternate tokens (`p >= 5%`, `k <= 10`). |
| `forking_branch_plan.py` | greedy base-path decode + per-`(t, w)` forced-prefix enumeration. |
| `forking_plan_forker.py` | `fork_plan_positions`: fork a SUBSET of plan positions in ONE batched decode (the shared inner loop; REAL index `t` preserved). |
| `forking_prior_resample.py` | `resample_prior`: o_0 from N full-resample draws of the bare prompt (shared by single-box capture + shard 0). |
| `forking_path_capture.py` | Stage 1-3 orchestration: forks ALL positions via `fork_plan_positions` -> `ForkingTrajectory`; with `dump_dir` set, also writes one `pos_<NNN>.json` raw rollout dump per base-path position. |
| `forking_plan_serialization.py` | flatten/rebuild a `BranchPlan` (`ForkingBranchRow`, `SerializedBranchPlan`) so the base path is decoded ONCE and shipped to the shard fleet. |
| `forking_shard_record.py` | `ForkingShard`: one fleet box's partial trajectory (its forked position slice + o_0 on shard 0 only). |
| `forking_shard_merge.py` | `merge_forking_shards`: reassemble positions 0..P-1 by REAL index, PAD any index no shard covered with a gap sentinel (prior o_0 = zero drift, no alternates) so a 4-of-5 fleet stays aligned, and RAISE loudly if shard 0 (the prior) is absent. |
| `semantic_drift_series.py` | `y_t = L2(o_0, o_t)` + variance-0.03 Gaussian noise. |
| `segment_evidence.py` | Bayesian marginal likelihood of a piecewise-linear segmentation. |
| `bayesian_change_point.py` | RJ-MCMC over change points -> `p(τ=t\|y)`, `p(m\|y)`, Bayes factor. |
| `forking_dynamic_states.py` | pull / drift / potential / forking-magnitude `Δ_t`. |
| `forking_diversity_series.py` | balance `H(O_t)`, disruption, `E[∂]`, `Var[∂]`. |
| `forking_survival_analysis.py` | hazard `h(t)` and survival `S(t)` (Eq. 3). |
| `forking_item_selection.py` | pilot outcome-entropy ranking of candidate items. |
| `forking_analysis_result.py` | `ForkingAnalysis` bundle + `analyze_forking_trajectory`. |

## Reuse
- `src.common.math`: `l2_distance`, `normalize`, `shannon_entropy`,
  `probs_to_logprobs`, `expected_deviance`, `deviance_variance`.
- `src.dynamics`: `pull`, `drift`, `potential`, `normalized_norm` (shared App. H
  metrics; the dimension-normalized `||·||/sqrt(dim)` is identical to the
  homogenization study).
- `src.datasets.sesgo_eval.parse_chosen_label`: the SESGO answer parser.
- `src.inference.ModelRunner.continue_from_text_batch`: the batched
  raw-continuation fast path (vLLM continuous batching on the cloud box).

## Hyperparameters (from the papers, hardcoded)
top-K alternates `k <= 10`, prob floor `5%`; drift noise variance `0.03`; Bayes
factor significance `> 9`; survival divergence `epsilon = 0.6` (L2). Continuations
per branch `S` and prior draws `N` are driver flags (paper: `S = 30`, `N = 300`).

## Sharded forking (cross-box parallelism)
The expensive forking step parallelizes across N boxes by POSITION. The base path
is decoded ONCE on one box (`branch_plan_to_serialized` -> `base_path.json`); each
shard box loads it (`serialized_to_branch_plan`) and forks only its slice
`[t for t in range(P) if t % N == k]` via `fork_plan_positions`, writing a
`ForkingShard`. Only shard 0 computes the prior o_0. A local merge
(`merge_forking_shards`) reassembles every position by REAL index `t` across the
full base length, PADDING any index no shard covered with an explicit gap sentinel
(prior o_0 = zero drift, no alternates, gap-marked token) so a 4-of-5 fleet still
produces a correctly-indexed, gap-marked figure (the padded indices are warned
LOUDLY, never silently compacted); a missing shard 0 / prior RAISES rather than
emitting an empty drift baseline. The result is a drop-in for analysis + plot.
Single-box `capture_forking_trajectory` is the all-positions case of the same
`fork_plan_positions`, so its output is byte-for-byte unchanged. The run-by-path
drivers + cloud fleet scripts live under `sesgo/forking/` and `cloud/`.

## CPD note
The paper uses Rbeast (BEAST). Rbeast is not a project dependency, so
`bayesian_change_point.py` implements the SAME model (piecewise-linear trend,
Gaussian noise, posteriors over change-point count + location) as a self-contained
reversible-jump MCMC. Calibrated so a clean outcome step clears `BF > 9` while a
flat series stays at `m = 0` (see `EXPLANATION.md`).
