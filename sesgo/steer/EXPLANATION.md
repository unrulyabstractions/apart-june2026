# SESGO causal steering — explanation

## What this pipeline proves

The geometry study is *correlational*: it shows the scaffold moves the residual
stream and where. This study closes the causal loop. It **inverts** the geometry
capture — instead of reading the residual, it writes a scaffold-shaped direction
back in with a forward hook — and tests whether that single linear edit **causes**
the model to abstain on ambiguous items, generalising to **held-out** questions the
direction was never fitted on.

Abstention = the model preferring the UNKNOWN ("not enough information") option. On
ambiguous SESGO items UNKNOWN is the unbiased gold, so raising its probability is
exactly the debiasing effect the scaffold produces.

## The direction (the inverse of the geometry capture)

The geometry capture saved, per (sample, structural position, layer), the
`[d_model]` residual at `blocks.{layer}.hook_resid_post`. For each captured layer L
the steering vector is the **diff-of-means**:

```
v[L] = mean over (TRAIN pairs x {im_end, newline, im_start, assistant}) of
       ( resid_scaffold - resid_noscaffold )
```

- **Pairing**: by `question_id` (each id appears twice — scaffold / no-scaffold),
  split on `has_scaffold`. `scaffold_id` is the constant `interpretive_direction`
  whenever present, so it is **not** a pairing key.
- **Positions**: the four change-of-turn chat-template boundary tokens (the
  positions the scaffold reshapes), not the think/answer positions.
- **Train only**: a seeded 70/30 split over the 231 pairs; the direction sees only
  TRAIN, the causal test runs only on TEST. The split is saved in the bundle so Run
  and Verify reuse the exact held-out items.

Adding `alpha * v[L]` to `blocks.L.hook_resid_post` is the literal inverse of the
capture (same hook point, opposite operation).

## Module map (library `src/steer/`, drivers `sesgo/steer/`)

| File | Role |
|------|------|
| `contrastive_pair_index.py` | group GeometrySamples into (scaffold / no-scaffold) pairs by `question_id` |
| `geometry_residual_loader.py` | load one `[d_model]` residual (`torch.load(root / a.path)`, the canonical loader) |
| `diff_of_means_vectors.py` | per-layer `mean(scaffold - no_scaffold)` over train pairs x change-of-turn positions |
| `seeded_pair_split.py` | seeded, reproducible 70/30 split over question_ids |
| `captured_layer_index.py` | enumerate captured layers; pick the silhouette-peak primary layer (reuses `select_feature_layer`) |
| `steering_vector_schema.py` | `SteeringVectorBundle` (per-layer vectors + the split) — BaseSchema, flat |
| `sesgo_abstention_readout.py` | rebuild the `choose3` (prompt, prefix, labels) triple from a GeometrySample; read UNKNOWN mass / abstain flag |
| `steered_ternary_runner.py` | `TernaryChoiceRunner` subclass: `_run_triple` -> `compute_trajectories_batch_with_intervention` |
| `steering_intervention_runner.py` | build the `steering(...)` intervention per alpha; sweep + aggregate abstention |
| `steering_test_schema.py` | `SteeringTestResult` / `SweepPoint` / `ScaffoldReference` — BaseSchema, flat |
| `calculate_steering_vectors.py` | **driver**: fit + split + save the bundle |
| `run_steering_test.py` | **driver**: held-out (TEST) + in-sample (TRAIN) alpha sweep + scaffold reference -> `steering_test.json` |
| `steering_plot_styles.py` | presentation-only: palette + per-model abstention-vs-alpha panel drawer |
| `plot_steering_test.py` | **driver**: cross-model figure (TEST vs TRAIN curves, baseline/control/scaffold marks) -> `figures/abstention_vs_alpha.png` |

## Reuse (no re-implementation)

- The add-mode `resid_post` hook is **fully built** in `src/inference/interventions/`.
  We only call `steering(layer, direction, strength=alpha, normalize=...)` and route
  `choose3` through the existing `compute_trajectories_batch_with_intervention`.
- The 3-option readout (`choose3` + `_divergent_scores`) and the position->role
  remap (`SesgoNonThinking.from_ternary`) are reused verbatim; the steered runner
  swaps only the single batched-forward call.
- Residual loading mirrors `risk_geometry_analysis` (`root / a.path`); the primary
  layer reuses geometry's `select_feature_layer`.

## Hook semantics

The HuggingFace `resid_post` add-hook is a forward hook on `model.model.layers[L]`
that does `out[0] += alpha * v` over `[batch, seq, d_model]` — applied at **every**
position (the teacher-forced label position included), which is what the
teacher-forced abstention readout reads. `normalize=True` makes alpha the absolute
magnitude of a unit steer; the default (`normalize=False`) scales the raw captured
diff vector, whose norm grows with depth, so large alpha eventually over-steers.

## Result — held-out (TEST) abstention vs alpha, raw-scale v

Figure: `out/sesgo/steer/figures/abstention_vs_alpha.png` (one panel per model;
TEST = held-out causal curve, TRAIN = in-sample echo; alpha=0 baseline, control
region, and real-scaffold reference all marked). Numbers are the TEST split
(items the vector was NEVER fit on); the TRAIN curve tracks it closely.

**Qwen3-0.6B** (24 held-out ambiguous items, layer 14):

| alpha | mean UNKNOWN prob | abstain rate | Δ vs baseline |
|------:|------------------:|-------------:|--------------:|
| -2 (control) | 0.175 | 0.167 | -0.317 |
| 0 (baseline) | 0.492 | 0.542 | — |
| +0.5 | 0.558 | 0.625 | +0.066 |
| +1 | 0.601 | 0.708 | +0.109 |
| +2 | 0.617 | 0.750 | +0.125 |
| +4 | 0.481 | 0.542 | -0.011 |
| scaffold ref | 0.994 | 1.000 | (target) |

**Qwen3-1.7B** (12 held-out items, layer 14): abstain 0.667 (a=0) -> 0.917 (a=+2)
-> **1.000 (a=+4)**, hitting the scaffold's 1.0; control (a=-2) drops to 0.417.

**Qwen3-4B** (12 held-out items, layer 23): **null** — baseline already 0.75 and
the chosen layer-23 direction is flat (Δ ≈ 0 across +alpha). A single linear add
at this layer does not move abstention; the effect is model- and layer-specific.

`+v` monotonically raises held-out abstention through alpha=+2 on 0.6B/1.7B; the
negative control collapses it; raw-`v` `+4` over-steers off-manifold on 0.6B (but
on 1.7B reaches the scaffold). The causal claim holds on items the vector never
saw, for 2 of 3 models; Qwen3-4B (layer 23) is an honest null.

## Limitations / dual-use

- A single linear add at one layer reaches the scaffold's near-total abstention on
  Qwen3-1.7B (1.0 at a=+4) but only ~75% of it on Qwen3-0.6B and **nothing** on
  Qwen3-4B (layer 23) — the effect is model- and layer-specific, and the richer
  multi-layer scaffold is not always reproducible by one direction.
- Steering toward abstention is a *safety* lever (avoid biased commitments) but the
  same hook with `-alpha` **suppresses** abstention, i.e. pushes the model to commit
  to a (possibly biased) group — a dual-use direction to handle carefully.
- Validated on Qwen3-0.6B only; magnitudes are model- and layer-specific.
