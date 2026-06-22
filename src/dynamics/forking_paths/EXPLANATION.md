# Forking-Paths O_t dynamics — method & implementation notes

## The object: O_t

For a fixed prompt and one greedily-decoded base thinking path `x* = (x_1..x_T)`,
the analysis estimates, at every base-path token position `t`, a distribution over
the FINAL categorical outcome:

- `o_{t,w}` (Eq. 1) — condition on forcing the t-th token to an alternate `w`, then
  sample `S` continuations to completion; map each to a one-hot outcome `R`
  (TARGET / OTHER / UNKNOWN / unparseable) and average. Implemented in
  `outcome_histogram_builder.conditional_histogram`.
- `o_t` (Eq. 2) — average `o_{t,w}` over the alternate tokens `w`, weighted by the
  base-path next-token probability `p(x_t=w | x*_{<t})`. Implemented in
  `outcome_histogram_builder.position_histogram`.

`o_0` is the **prior**: an `N`-sample full-resample from the bare prompt (before any
base-path token is committed). The series `{o_0, o_1, ..., o_T}` is the multivariate
time-series the rest of the analysis operates on.

This `O_t` is exactly the Rios-Sialer **system barycenter** `⟨Λ_n⟩(x_t)` — the App. H
"pull" — when the structures `α_i` are the indicator scores for each outcome
category. So one captured object feeds BOTH papers' analyses.

## Capture (Stages 1-3) and batching

`forking_branch_plan.build_branch_plan`:
1. greedy-decode the base path (`generate_trajectory_from_prompt`, temp 0) to get
   the realized token ids + the prompt/base split;
2. recover the per-position **full-vocab logits** with ONE teacher-forced pass
   (`compute_trajectory`) — KV-cached generation returns only scalar logprobs, so
   the top-K source must come from a forward pass over the realized sequence;
3. read the top-K alternates per position (`forking_top_k_tokens`, `p >= 5%`,
   `k <= 10`, base token always included);
4. build each `(t, w)` forced PREFIX string = templated prompt + base tokens
   `x*_{<t}` + `w` (a leading BOS is stripped so the backend's `add_special_tokens`
   re-encode does not duplicate it).

`forking_plan_forker.fork_plan_positions` is the shared inner loop: given a SUBSET
of plan positions, it expands every prefix `S` times into ONE flat list and issues a
SINGLE `runner.continue_from_text_batch` call, then slices the flat rollouts back per
`(position, alternate)` and builds a `ForkPosition` carrying the REAL index `t`. On
the cloud GPU box this rides vLLM continuous batching; locally it rides the
HuggingFace left-padded batched decode. `continue_from_text_batch` is the
raw-continuation primitive — unlike `generate_batch`, it does NOT re-apply the chat
template (the prefixes are already fully rendered).
`forking_path_capture.capture_forking_trajectory` simply calls
`fork_plan_positions` over ALL positions, so the single-box trajectory is byte-for-
byte identical to the per-position forker's output; the prior o_0 comes from
`forking_prior_resample.resample_prior`.

`max_positions` caps how many leading positions are branched (the expensive part)
while still decoding the full base path for the text strip — the local-pilot knob.

## Sharded forking (cross-box parallelism)

For a long base path the forking is split across `N` boxes by POSITION:
1. **decode once** — one box runs `build_branch_plan`, then
   `forking_plan_serialization.branch_plan_to_serialized` flattens the nested
   `rows_per_position` into a 1D `list[ForkingBranchRow]` (a `SerializedBranchPlan`)
   written to `base_path.json`;
2. **fork a slice** — each shard box `k` reloads it
   (`serialized_to_branch_plan`) and calls `fork_plan_positions` for only
   `[t for t in range(P) if t % N == k]`, writing a `ForkingShard`; only shard 0
   pays the prior's `N`-draw cost (`resample_prior`);
3. **merge** — `forking_shard_merge.merge_forking_shards` reassembles every position
   by REAL index `t` across the full base length, PADDING any index no shard covered
   with an explicit gap sentinel (prior o_0 = zero drift, no alternates, gap-marked
   token) so the change-point τ / fork-token indices stay aligned to
   `base_token_texts` — the padded indices and any duplicates are warned LOUDLY,
   never silently compacted. A missing shard 0 / prior RAISES (the prior is the
   drift baseline for every position); o_T comes from the last position. The result
   is the exact `ForkingTrajectory` the analysis + plot drivers consume, so a
   4-of-5-shard fleet still yields a correctly-indexed, gap-marked figure.

## Semantic drift -> change points

`semantic_drift_series.outcome_drift_series` reduces `{o_t}` to the univariate
`y_t = ||o_0 - o_t||_2` (drift away from the starting belief), then
`add_drift_noise` adds variance-0.03 Gaussian noise (the paper's false-positive
control — BEAST re-normalizes `y`, so flat regions otherwise wiggle into spurious
change points).

`bayesian_change_point.detect_change_points` fits the paper's CPD model: `y` split
into `m+1` segments, each a degree-1 linear regression `y_t = β_i t + δ_i` with
Gaussian observation noise. Because Rbeast is not a dependency, we run a
**reversible-jump MCMC** (birth / death / move of interior change points) accepting
by the Metropolis ratio of the configuration's marginal likelihood
(`segment_evidence`) times a geometric prior on `m`. The sampler accumulates:
- `p(τ=t | y)` — visit frequency of each interior position (the forking-token
  localizer; the paper highlights `p(τ=t|y) > 0.7`);
- `p(m | y)` — frequency of each change-point count;
- Bayes factor `BF = p(m>=1|y) / p(m=0|y)`; `BF > 9` declares a forking token,
  localized at `argmax_t p(τ=t|y)`.

Boundaries `0` and `T` are fixed and excluded from the posteriors (paper boundary
handling).

### Segment evidence (why a plain BIC fails)

A token-level drift series has very short, often exactly-flat segments. A
maximum-likelihood / BIC score rewards splitting flat-then-step data into many tiny
segments without bound (residual → 0 ⇒ log-likelihood → +∞). `segment_evidence`
instead integrates out BOTH the regression coefficients (Zellner g-prior) and the
noise variance (Inverse-Gamma `a0=1, b0=0.005`), giving the closed-form Student-t
marginal likelihood whose built-in Occam factor keeps `σ` away from 0. Calibration
(`b0=0.005`, per-change penalty `2.0`): a clean outcome step (drift `0 → 0.99`)
gives `BF ≈ 18` at the true index; a flat control gives `p(m=0) ≈ 0.95`.

## Dynamic states (App. H)

`forking_dynamic_states.compute_dynamic_states`, reusing the shared
`src.dynamics` metrics (dimension-normalized `||·||/sqrt(dim)`):
- **pull** `||O_t||` — strength of the attractor at `t`;
- **drift** `||O_t - O_0||` — accumulated deviation from the prior;
- **potential** `||O_T - O_t||` — deviance still required to reach the end;
- **forking magnitude** `Δ_t = ||O_t - O_{t-1}||` — consecutive-barycenter first
  difference; `argmax_t Δ_t` is the most-forking token (Eq. H.9).

## Diversity (Sec. 7)

`forking_diversity_series.compute_diversity_series`, measured on the same rollouts:
- **balance** `H(O_t)` (barycenter entropy, want high);
- **disruption** `||O_t - O_0||` (default moved off the baseline);
- `E[∂_n]`, `Var[∂_n]` — mean/variance of the scalar deviance over the rollouts.
Homogenization is `E[∂_n] → 0` AND `Var[∂_n] → 0`.

## Survival (Eq. 3)

`forking_survival_analysis.compute_survival`: hazard
`h(t) = Σ_w p(x_t=w) · 1[ ||o_{t,w} - o_{t,w*}||_2 > ε ]` with `ε = 0.6`, and the
running survival `S(t) = Π_{t'<=t} (1 - h(t'))` — the probability the base path
"survives" with no outcome-changing alternate through position `t`.

## Local-pilot caveat

With a handful of positions and small `S` the drift signal is short and noisy, so
CPD will not flag a forking token (correct — there is not enough evidence). The
`most_forking_index` from the dynamic states still localizes the largest barycenter
jump. The cloud run (`S = 30`, `N = 300`, all positions, ~30 questions) is the
scientific configuration; everything here scales up by raising the driver flags.
