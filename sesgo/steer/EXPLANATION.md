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
| `run_steering_test.py` | **driver**: held-out alpha sweep + scaffold reference -> `steering_test.json` |

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

## Pilot result (Qwen3-0.6B, 16 held-out ambiguous items, raw-scale v, layer 14)

| alpha | mean UNKNOWN prob | abstain rate | Δ vs baseline |
|------:|------------------:|-------------:|--------------:|
| -2 (control) | 0.068 | 0.000 | -0.315 |
| 0 (baseline) | 0.383 | 0.500 | — |
| +0.5 | 0.454 | 0.562 | +0.072 |
| +1 | 0.496 | 0.625 | +0.113 |
| +2 | 0.509 | 0.625 | +0.127 |
| +4 | 0.380 | 0.500 | -0.003 |
| scaffold ref | 0.991 | 1.000 | (target) |

`+v` monotonically raises abstention through alpha=+2; the negative control collapses
it; `+4` over-steers off-manifold (raw vector). The hook provably changes the raw
divergent-token logits. The causal claim holds on items the vector never saw.

## Limitations / dual-use

- A single linear add at one layer does **not** reproduce the scaffold's near-total
  abstention (0.99) — the scaffold is a richer, multi-layer effect.
- Steering toward abstention is a *safety* lever (avoid biased commitments) but the
  same hook with `-alpha` **suppresses** abstention, i.e. pushes the model to commit
  to a (possibly biased) group — a dual-use direction to handle carefully.
- Validated on Qwen3-0.6B only; magnitudes are model- and layer-specific.
