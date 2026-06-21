# SESGO experiments — commands

Spanish ambiguous/disambiguated social-bias study: does a debiasing **scaffold**
make a model abstain (ambiguous gold = *unknown*) or answer correctly
(disambiguated gold = the labelled group)? Five studies, two readouts
(non-thinking teacher-forced + sampled/greedy thinking), plus residual-stream
geometry. Outputs land in `out/sesgo/<study>/<MODEL>/` (`response_samples.json`,
`plots/`, geometry also `activations/` + `analysis/`).

Always run with `uv run python` (or `.venv/bin/python` on the cloud boxes).

## 0. Generate the prompt datasets (es + original; ambig + disambig)
```bash
uv run python sesgo/generate/generate_prompt_dataset.py
```
Writes `out/sesgo/<study>/prompt_dataset.json` for all five studies.

## 1. Run a study  (collect → visualize)
`<RS>` = `out/sesgo/<study>/<MODEL>/response_samples.json` (MODEL bare name, e.g. `Qwen3-0.6B`).
```bash
# baseline — non-thinking + greedy-non-thinking + greedy-thinking + 2-option
uv run python sesgo/baseline/collect_baseline_samples.py   [--subsample 0.05] [--batch-size 8]
uv run python sesgo/baseline/visualize_baseline_samples.py <RS>

# stability — consistency across surface variants (by-item subsample)
uv run python sesgo/stability/collect_stability_samples.py --items 12
uv run python sesgo/stability/visualize_stability_samples.py <RS>

# selection — per-scaffold accuracy, non-thinking + thinking (all 5 scaffold conditions)
uv run python sesgo/selection/collect_selection_samples.py  [--subsample 0.05 --n-thinking 4]
# when run sharded across cloud boxes, fold the shard slices back into one dataset first:
uv run python sesgo/selection/combine_selection_shards.py Qwen3-0.6B  [--sync-dir sync --out-dir out]
uv run python sesgo/selection/visualize_selection_samples.py <RS>

# divergence — system-default distribution over hot-sampled thinking draws
uv run python sesgo/divergence/collect_divergence_samples.py [--subsample 0.05 --n-thinking 8]
uv run python sesgo/divergence/visualize_divergence_samples.py <RS>

# geometry — residual-stream capture along the greedy path (forces the HF backend)
# Captures PER (position, layer): 8 chat-template positions (im_end, newline,
# im_start, assistant, think_open, think_close, answer_prefix, label) x each
# MIDDLE->LAST transformer layer (floor(L/2)..L-1; override with --layers). The
# grid spans BOTH no-scaffold and scaffolded prompts (scaffold-vs-none axis).
uv run python sesgo/geometry/collect_geometry_samples.py    [--subsample 0.1 --n-thinking 0 --layers 14,20,27]
# when run sharded across cloud boxes, fold the shard slices (samples + .pt) into one set first.
# (merge_sync.sh's --ignore-existing is WRONG for shards: it keeps only the first shard's
#  response_samples.json. This concatenates samples de-duped by sample_idx and unions the
#  per-shard activations/.pt — disjoint by global sample_idx — into one model dir.)
uv run python sesgo/geometry/combine_geometry_shards.py Qwen3-0.6B  [--sync-dir sync --out-dir out]
uv run python sesgo/geometry/analyze_geometry.py            <RS>   # PCA per (layer x position) + per-axis separation
uv run python sesgo/geometry/visualize_geometry_samples.py  <RS>   # PCA grid (every colour axis) + depth heatmap/sweep
PORT=8002 bash sesgo/geometry/visualize_geometry.sh         <RS>   # interactive viz server
```
`analyze_geometry.py` defaults to `--layer all`: it runs a SEPARATE PCA for EVERY
captured mid->last layer (plus a `mean` layer-averaged cell), so depth is never
collapsed. Alongside the per-cell projections it writes a top-level
`layer_axis_silhouette` table — silhouette separability per (layer x colour-axis)
at a representative late position — so one can see at what depth each axis becomes
separable. `visualize_geometry_samples.py` renders the 2D PCA scatters
(`pca_scatter_<position>.png`, `pca_by_<axis>.png`, `pca_axes_grid.png`) at an
ADAPTIVE feature layer chosen per model — the captured mid->last layer that
MAXIMISES the scaffold silhouette at the answer-`label` position (where the
structure is clearest), with a deeper-layer tie-break (within ε=0.01 of the max)
and a last-layer fallback (see `geometry_feature_layer.py`). The title carries that
layer + its model-relative depth (e.g. `layer 14/28 (0.50 depth)`), so the chosen
depth is comparable ACROSS model sizes by structure rather than absolute index.
The per-axis panels come from the shared `geometry_color_axes` registry
(categorical axes -> discrete legend; the continuous answer-distribution signals —
top-choice prob/logit, entropy, diversity, inverse perplexity -> sequential
colormap + colorbar). Two layer-aware depth views intentionally show ALL depths:
`silhouette_by_layer_axis.png` (the layer x axis heatmap) and
`silhouette_layer_sweep.png` (silhouette-vs-layer for the key axes accuracy /
context_condition / selected_role / scaffold). To add a colour-by axis, add ONE
`ColorAxis` row to `sesgo/geometry/geometry_color_axes.py` — analysis separation
AND every viz panel pick it up automatically.
The viz server is **multi-model**: it discovers every model under
`out/sesgo/geometry/*/` that has both `response_samples.json` and
`analysis/projections.json`, and exposes a **Model** selector. Switching models
reloads that model's PCA geometry (all layers/positions/axes) in place; `<RS>`
only fixes which model the page boots into. Run `collect → analyze` for each
model you want to compare; one model is enough for the server to run.
Common flags: `--model <hf-id>` (default `Qwen/Qwen3-0.6B`), `--subsample 0..1`,
`--batch-size N` (batched forward), `--shard-index/--shard-count` (split the grid).
Collection checkpoints to `response_samples.json` as it goes — re-run the same
command to **resume** after a crash.

### forking — per-token forking-paths dynamics on ONE ambiguous item
A different shape from the five batch studies above: instead of one readout per
item, this tracks how a SINGLE ambiguous item's outcome distribution `O_t`
evolves token-by-token along the thinking trajectory, locates the **forking
token** (Bayesian change point) where the committed answer locks in, and derives
the pull/drift/potential states, outcome diversity, and survival series. Run the
four drivers in order (each persists its artifact under
`out/sesgo/forking/<MODEL>/`); see `sesgo/forking/README.md` for the full method.
```bash
# 0. pick the highest-outcome-entropy ambiguous item (most likely to FLIP)
uv run python sesgo/forking/select_forking_item.py     --model Qwen/Qwen3-0.6B --categories gender --n-pilot 12 --max-new-tokens 600
# 1-3. capture {O_t}: greedy base path + batched branch sampling at every token
uv run python sesgo/forking/collect_forking_rollouts.py --model Qwen/Qwen3-0.6B --n-samples 40 --n-prior 300 --max-new-tokens 512
# 4. analyze: change-point + pull/drift/potential + diversity + survival
uv run python sesgo/forking/analyze_forking_dynamics.py --model Qwen/Qwen3-0.6B
# 5. plot: stacked-area O_t + token strip + companion dynamics panels
uv run python sesgo/forking/plot_forking_dynamics.py    --model Qwen/Qwen3-0.6B
```
Writes `selected_item.json`, `forking_trajectory.json`, `forking_analysis.json`,
and the headline `forking_dynamics.png`. The cloud run is the SAME commands with
`--max-positions 0` (every token, not a pilot subset); the branch decode rides
vLLM continuous batching on CUDA, and the HF backend micro-batches the forking
set (`HF_GEN_MICRO_BATCH`, default 64) so a 32B model fits an 80 GB GPU.

### steer — causal abstention test (invert the geometry into an add-mode hook)
The geometry study shows *where* the debiasing scaffold moves the residual stream;
this study adds that movement back in with the existing add-mode `resid_post`
intervention (`src/inference/interventions/`; nothing here re-implements hooks) and
asks whether `+alpha * v[L]` **causes** abstention (UNKNOWN mass) on **held-out**
ambiguous items the vector never saw. The direction is the diff-of-means
`v[L] = mean(resid_scaffold - resid_noscaffold)` over a seeded TRAIN split of the
231 scaffold-vs-none contrastive pairs. Requires the geometry artifact
(`out/sesgo/geometry/<MODEL>/`); see `sesgo/steer/README.md` for the full method.
```bash
# 1. fit per-layer diff-of-means vectors + persist the seeded 70/30 pair split
uv run python sesgo/steer/calculate_steering_vectors.py out/sesgo/geometry/<MODEL>/response_samples.json   [--seed 42 --train-fraction 0.7]
# 2. alpha sweep on held-out TEST ambiguous items (incl. alpha=0 + negative control)
uv run python sesgo/steer/run_steering_test.py out/sesgo/steer/<MODEL>/steering_vectors.json out/sesgo/geometry/<MODEL>/response_samples.json   [--layer 14 --alphas="-2,0,0.5,1,2,4" --normalize --limit 0]
# 3. cross-model figure: held-out TEST vs in-sample TRAIN abstention vs alpha
uv run python sesgo/steer/plot_steering_test.py out/sesgo/steer/Qwen3-0.6B/steering_test.json out/sesgo/steer/Qwen3-1.7B/steering_test.json out/sesgo/steer/Qwen3-4B/steering_test.json   [--metric abstain_rate --out out/sesgo/steer/figures/abstention_vs_alpha.png]
```
Writes `steering_vectors.json` (vectors + split), `steering_test.json` (the alpha
`sweep` on TEST + `train_sweep` + `scaffold_reference`), and the headline
`abstention_vs_alpha.png`. The causal claim holds when abstention rises with
positive alpha and the negative control drops it — on items `v` was never fit on.
Forces the HuggingFace backend (the only backend the intervention path supports).

## 2. Cross-model size sweep (headline figure)
```bash
uv run python sesgo/baseline/visualize_baseline_cross_model.py   # accuracy vs model size
```
Scans every `out/sesgo/baseline/<model>/` and plots the size trend per family.

## 3. Run the sweep on the cloud (Vast.ai, parallel, one right-sized GPU/model)
```bash
FLEET_CONFIRM=1 bash cloud/fleet_launch.sh                 # create all boxes concurrently
STUDIES=baseline BATCH_SIZE=32 bash cloud/fleet_run.sh     # run → sync-back → self-destruct
bash cloud/merge_sync.sh                                   # sync/ quarantine → out/
```
Models + GPU map live in `cloud/fleet_sizing.py`. Needs `HF_TOKEN` for gated
models. See `cloud/README.md` for the full flow + safety guarantees.
