# `src/datasets/risk_geometry`

Residual-stream geometry capture for the **mental_risk** experiment — the
geometry-half analogue of the bias-role `src/datasets/sesgo_eval` geometry trio
(`geometry_activation.py` / `geometry_sample.py` / `geometry_dataset.py`), kept in
its own package with `_risk`-suffixed names so every filename stays globally
unique.

## Files

| File | Defines | Mirrors (sesgo) |
| --- | --- | --- |
| `risk_geometry_activation.py` | `RiskGeometryActivation` — a pointer (relative path) to one saved `[n_layers, d_model]` residual tensor at a structural token position. | `GeometryActivation` |
| `risk_geometry_sample.py` | `RiskGeometrySample` — one prompt's two-level risk readout (`NonThinkingResult` + `ScoreSummary`) plus its `activations`. | `GeometrySample` |
| `risk_geometry_dataset.py` | `RiskGeometryDataset` — `(prompt_dataset_id, model, config)` header + a list of `RiskGeometrySample`. | `GeometryDataset` |
| `risk_activation_capture.py` | `capture_activations(...)` + `find_positions(...)` + `POSITION_TYPES` — the reusable engine that snapshots residuals along the greedy non-thinking answer path. | the in-driver `capture_activations` of `collect_geometry_samples.py` |

## Key risk-vs-bias differences

- **No abstention gold.** SESGO's gold is always `UNKNOWN`, so its sample exposes
  `correct_non_thinking`. Risk gold is a continuous `gold_risk`, so the sample
  exposes `predicted_risk_non_thinking` / `predicted_risk_thinking` instead.
- **Binary, not ternary.** The non-thinking readout is `NonThinkingResult`
  (calibrated P(at risk) over two labels), captured via `BinaryChoiceRunner`, not
  the 3-way `SesgoNonThinking`.
- **Color-by axis is `framing`** (the intervention axis), with `disorder` /
  `language` as the other flat axes (cf. SESGO's `scaffold_id` /
  `bias_category` / `question_polarity` / `language`).

The capture itself is task-agnostic: it forces the HuggingFace backend (the only
one exposing `run_with_cache`), greedily decodes past an empty `<think></think>`
block, and snapshots the residual stream at `turn` / `think_open` / `think_close`
/ `answer`.

All classes inherit `BaseSchema`; the tensors live on disk and are referenced by
relative path only, so the dataset JSON stays small.
