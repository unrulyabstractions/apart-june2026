# `sesgo/forking/` — why this study, and how the drivers fit together

## Motivation

SESGO ambiguous items have no evidence, so the gold answer is always UNKNOWN
("no hay suficiente información"). A model that reasons freely may nonetheless
*commit* to a group mid-reasoning. Forking-paths analysis pinpoints the **token**
at which that commitment crystallizes — the position where re-sampling a different
next token would divert the FINAL answer to a different outcome. On a bias
benchmark this is the token where the model "decides to be biased".

The same captured object doubles as the Rios-Sialer App. H trajectory: `O_t` is the
system barycenter (pull), and its evolution gives drift / potential / diversity, so
we read both a forking-token story and a homogenization story off one capture.

## Stage 0 — item selection (`select_forking_item.py`)

A forking token only exists if the outcome can flip, so the best demo item is the
one whose sampled thinking decodes DISAGREE. We render the canonical
single-permutation ambiguous grid, pilot a few sampled decodes per candidate item,
map each to its categorical outcome, and rank by the Shannon entropy of the pilot
outcome distribution. The top item is written to `selected_item.json` (the full
`SesgoPromptSample` so later stages parse answers with the identical
`position_labels`).

**Why the token budget matters:** Qwen3 reasoning models spend their budget inside
`<think>`; the answer parser only reads text AFTER `</think>`. Too small a budget
truncates mid-thought and every draw lands in the `unparseable` bucket (entropy 0,
useless). The verified pilot used `--max-new-tokens 600`; the divergence study's
512 default is the production value.

## Stages 1-3 — capture (`collect_forking_rollouts.py`)

Loads the selected item and calls `capture_forking_trajectory`. The base path is
greedily decoded once; the per-position full-vocab logits are recovered by ONE
teacher-forced pass; the top-K alternates per position are read; and EVERY
`(position, alternate)` branch prefix is sampled `S` times in a SINGLE batched
decode. The result is a `ForkingTrajectory` (one `ForkPosition` per base-path
token, each carrying its alternates' rollouts, `o_{t,w}`, and `o_t`), plus the
prior `o_0` (an `N`-sample full-resample) and the final `o_T`.

`--max-positions` caps how many leading tokens are branched (cost control for the
local pilot); the cloud run sets it to 0.

## Stage 4 — analysis (`analyze_forking_dynamics.py`)

`analyze_forking_trajectory` runs the four analyses over `{O_t}`:
1. **change point** — reduce to `y_t = L2(o_0, o_t)`, add variance-0.03 noise, run
   the Bayesian RJ-MCMC CPD, report `p(τ=t|y)`, `p(m|y)`, the Bayes factor, and the
   forking-token index;
2. **dynamic states** — pull / drift / potential / forking-magnitude `Δ_t`;
3. **diversity** — balance `H(O_t)`, disruption, `E[∂]`, `Var[∂]`;
4. **survival** — hazard `h(t)` and survival `S(t)`.
Written to `forking_analysis.json`.

## Stage 5 — plotting (`plot_forking_dynamics.py`)

The headline figure is the **stacked-area `O_t` vs token position** (one colored
band per outcome category, `y ∈ [0, 1]`), with the detected change-point token
marked by a red dashed line and the base-path tokens drawn beneath, each shaded by
`p(τ=t|y)` on a yellow→red heatmap (the forking token gets a red box). Companion
panels show the dynamic states, the diversity series, and survival with the
`p(τ=t|y)` curve overlaid. The band widths show how the outcome distribution
"collapses" as the model commits.

## Verification status

The full five-stage pipeline was run end-to-end on a TINY local pilot (Qwen3-0.6B,
MPS, 6 branched positions, `S = 4`, `N = 4`): selection found a genuine
high-entropy item (1.099 nats), capture produced `o_0 = [0, .25, .25, .5]` →
`o_T = [0, 1, 0, 0]` (a clear outcome collapse onto OTHER), analysis ran the CPD +
all series, and plotting produced `forking_dynamics.png`. With so few positions the
CPD correctly finds no *significant* change point; the `most_forking` index still
localizes the largest barycenter jump. The cloud run is the identical commands at
paper scale.
