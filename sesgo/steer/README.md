# SESGO causal steering — commands

Invert the SESGO geometry into a **causal** test. The geometry study showed *where*
a debiasing scaffold moves the residual stream; this study adds that movement back
in with a forward hook and asks whether it **causes** abstention.

The steering direction at a layer is the diff-of-means
`v[L] = mean over (train pairs x change-of-turn positions) of (resid_scaffold - resid_noscaffold)`,
fitted on a **seeded train split** of the 231 contrastive (scaffold vs no-scaffold)
pairs. The causal claim: adding `+alpha * v[L]` to `blocks.L.hook_resid_post` raises
abstention (UNKNOWN mass) on **held-out** ambiguous items the vector never saw.

All hooks are the existing inference-stack add-mode `resid_post` intervention
(`src/inference/interventions/`); nothing here re-implements hooks. The HuggingFace
backend is forced (the only backend the intervention path supports). Always run with
`uv run python`. Inputs/outputs live under `out/sesgo/`.

`<MODEL>` = bare model name (e.g. `Qwen3-0.6B`). The geometry artifact
(`response_samples.json` + `activations/` + `analysis/projections.json`) must
already exist (see `sesgo/geometry/`).

## 1. Calculate steering vectors (fit + seeded split)
```bash
uv run python sesgo/steer/calculate_steering_vectors.py \
    out/sesgo/geometry/<MODEL>/response_samples.json \
    [--seed 42 --train-fraction 0.7]
```
Pairs scaffold vs no-scaffold by `question_id`, makes a seeded 70/30 split over the
pairs, and on the TRAIN pairs only computes a per-layer diff-of-means vector for
**every** captured layer (14..27 for Qwen3-0.6B). Writes one BaseSchema bundle —
the vectors **and** the split — to `out/sesgo/steer/<MODEL>/steering_vectors.json`.
The primary steering layer defaults to the mid-depth layer where the scaffold
silhouette peaks (read from `analysis/projections.json`; ~0.50 relative depth).

## 2. Run the held-out steering test (alpha sweep)
```bash
uv run python sesgo/steer/run_steering_test.py \
    out/sesgo/steer/<MODEL>/steering_vectors.json \
    out/sesgo/geometry/<MODEL>/response_samples.json \
    [--layer 14 --alphas="-2,0,0.5,1,2,4" --normalize --limit 0]
```
On the held-out TEST **ambiguous** items (UNKNOWN is the unbiased gold) with **no
scaffold** in the prompt, sweeps the steering strength alpha — including `0` (the
unsteered baseline) and a **negative control** — and measures abstention under the
hook. Also scores the **actual-scaffold** prompt unsteered (the behaviour `+v` aims
to reproduce). Writes `out/sesgo/steer/<MODEL>/steering_test.json`.

- `--normalize` makes `v` a unit vector, so alpha is the absolute steer magnitude;
  the default scales the **raw** diff-of-means vector (alpha multiplies its captured
  norm), so very large alpha eventually over-steers off the residual manifold.
- `--limit N` caps the held-out items (for a quick pilot).

On the held-out TEST **ambiguous** items the sweep runs, plus the SAME sweep on
the in-sample TRAIN items (saved as `train_sweep`) so the figure can show
generalization. Writes `out/sesgo/steer/<MODEL>/steering_test.json`.

## 3. Plot the steering test (cross-model figure)
```bash
uv run python sesgo/steer/plot_steering_test.py \
    out/sesgo/steer/Qwen3-0.6B/steering_test.json \
    out/sesgo/steer/Qwen3-1.7B/steering_test.json \
    out/sesgo/steer/Qwen3-4B/steering_test.json \
    [--metric abstain_rate|mean_unknown_prob \
     --out out/sesgo/steer/figures/abstention_vs_alpha.png]
```
One row of per-model panels: held-out **TEST** abstention vs alpha (the causal
claim) overlaid with the in-sample **TRAIN** curve, marking the alpha=0 unsteered
baseline, the negative-alpha control region, and the real-scaffold reference.
Rendered labels are minimal plain-language (no pipeline jargon, no how-to-read
gloss): the TEST curve legend reads "Held-out", the TRAIN curve reads "Fit set",
alpha reads "Steering strength", and the abstention metric carries Wilson 95%
whiskers. With no args it defaults to the three Qwen3 bundles. Writes
`out/sesgo/steer/figures/abstention_vs_alpha.png`.

## Reading the output
`steering_test.json` carries the alpha `sweep` (TEST split) and `train_sweep`
(TRAIN split) — each point: `mean_unknown_prob`, `abstain_rate`,
`delta_unknown_prob` vs the alpha=0 baseline — plus the `scaffold_reference`. The
causal claim holds when `delta_unknown_prob` / `abstain_rate` rise with positive
alpha and the negative control drops them — on items the vector was never trained
on.

See [EXPLANATION.md](./EXPLANATION.md) for the full pipeline / module map.
