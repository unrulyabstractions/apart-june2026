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

## Stage 5b — branching tree (`render_branching_tree.py`)

The same `plot_forking_dynamics.py` run also emits a **branching tree** in the
style of the homogenization paper's Fig. 11. It is built by `build_branching_tree`
(in `src/dynamics/forking_paths/`) directly from the captured `{O_t}`:
- **root** — an earlier base-path position (or the prior `o_0`), the shared frame;
- **trunk** — the detected forking token, its barycenter `o_t` (Eq. 2);
- **branches** — the trunk's top alternate continuations `w` (the realized base
  token always kept), each with its own conditional outcome `o_{t,w}` (Eq. 1) and
  the next-token mass `p(x_t=w)` on its edge.

The renderer draws it left-to-right: each NODE is a horizontal STACKED BAR over the
outcome categories (Okabe-Ito palette), each EDGE is a curved connector whose width
encodes the branch mass and which carries its opening token label, and one legend
names the outcome categories. The same renderer is imported by the divergence study
(`sesgo/divergence/visualize_divergence_samples.py`), where the branches are the two
NAMED identities of a representative ambiguous item — so both studies present the
fork the same way the paper does.

## Verification status

The full pipeline was run end-to-end LOCALLY on Qwen3-0.6B (MPS, HuggingFace) at a
real scale: 60 branched base-path positions, `S = 40` continuations per `(t, w)`,
`N = 60`-sample prior, 384-token rollouts — ~60 min, 92 total branches. Selection
chose a gender gossip-stereotype item (entropy 0.703); capture produced
`o_0 = [0, .32, .13, .55]` evolving to `o_T = [.03, .65, .30, .03]` with a clear
"target" excursion around t=40 and a swing to "unknown" near t=50.

On this tiny model the per-token `o_t` is noisy, so the Bayesian CPD correctly
returns **no significant single change point** (Bayes factor 0.02, `p(m=0)=0.98`):
the drift series is genuinely jittery rather than a clean piecewise step. That is an
honest negative — we do NOT fabricate a change point. The branching tree therefore
sits on the **most-divergent-branch** position (`most_divergent_branch_index`), the
model-agnostic forking signature: the token where re-sampling a different next token
most diverts the outcome. For this item that is `t = 42` (the token `","`), whose
base continuation leans `[target .45, other .15, unknown .35]` while the alternate
`" and"` (p = 0.12) swings to `[target .12, other .82]` — a real fork. The cloud run
is the identical commands with `--max-positions 0` (all tokens) at paper sample
counts, where the larger model's cleaner `o_t` is expected to clear the CPD's BF>9.
