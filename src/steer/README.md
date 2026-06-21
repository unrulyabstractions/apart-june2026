# src/steer/ — SESGO causal-steering library

Reusable engine behind the `sesgo/steer/` drivers. It inverts the SESGO geometry
capture into an add-mode `resid_post` steering vector and runs the abstention
readout under that hook on held-out items.

## Contents

| File | Exports | Purpose |
|------|---------|---------|
| `steering_vector_schema.py` | `LayerSteeringVector`, `SteeringVectorBundle` | per-layer vectors + the seeded split (BaseSchema, flat) |
| `steering_test_schema.py` | `SteeringTestResult`, `SweepPoint`, `ScaffoldReference` | alpha-sweep + scaffold-reference result (BaseSchema, flat) |
| `seeded_pair_split.py` | `split_question_ids` | seeded, reproducible 70/30 split over question_ids |
| `contrastive_pair_index.py` | `ContrastivePair`, `build_pairs` | group GeometrySamples into scaffold/no-scaffold pairs by `question_id` |
| `geometry_residual_loader.py` | `load_residual` | load one `[d_model]` residual (`root / a.path`, canonical) |
| `diff_of_means_vectors.py` | `steering_vector_for_layer`, `steering_vectors_all_layers`, `CHANGE_OF_TURN_POSITIONS` | per-layer diff-of-means over train pairs x change-of-turn positions |
| `captured_layer_index.py` | `captured_layers`, `primary_layer` | enumerate captured layers; pick the silhouette-peak primary layer |
| `sesgo_abstention_readout.py` | `AbstentionReadout`, `build_readout`, `unknown_probability`, `is_abstained` | rebuild the `choose3` triple from a sample; read UNKNOWN mass |
| `steered_ternary_runner.py` | `SteeredTernaryChoiceRunner` | runs `choose3` under an intervention (swaps one batched-forward call) |
| `steering_intervention_runner.py` | `measure_abstention`, `run_alpha_sweep`, `unsteered_reference` | build `steering(...)` per alpha; sweep + aggregate abstention |

## Reuse, not reimplementation

The add-mode `resid_post` hook lives in `src/inference/interventions/` and is used
via `steering(...)` + `compute_trajectories_batch_with_intervention`. The 3-option
readout and the position->role remap are reused from `src/ternary_choice/` and
`src/datasets/sesgo_eval/`. The primary-layer pick reuses geometry's
`select_feature_layer`. No new hooks are written here.

See `sesgo/steer/EXPLANATION.md` for the full pipeline and pilot result.
