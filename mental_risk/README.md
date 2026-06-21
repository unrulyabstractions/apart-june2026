# `mental_risk/` — the MentalRiskES risk-assessment experiment

Run-by-path drivers for the **mental_risk** experiment, mirroring the five-study
structure of [`sesgo/`](../sesgo/). Where SESGO probes a *bias-role
classification* (`LLM(x)` vs `LLM(x ⊕ scaffold)`, gold always `UNKNOWN`),
mental_risk probes a *continuous risk assessment*: given a subject's transcript,
the model judges their mental-health risk in `[0, 1]`, and we compare that against
the corpus gold risk.

Every driver bootstraps the repo root onto `sys.path`
(`sys.path.insert(0, parents[2])`) and is invoked **by path**, e.g.
`uv run python mental_risk/baseline/collect_baseline_risk.py`.

## SESGO → mental_risk mapping

| SESGO concept | mental_risk analogue | Why / how it differs |
| --- | --- | --- |
| **scaffold** (debiasing preamble, the intervention axis) | **framing** (`at_risk_of` / `suffering` / `safe` / `intervene`) | The framing reposes the same transcript as a different yes/no risk question. There is **no no-op baseline framing** — every framing is a real intervention — so selection ranks framings against *gold*, not against a baseline. Lives in [`scaffolds_risk.py`](./scaffolds_risk.py). |
| format axes (label style × permutation) | format axes (label style × order flip × scale direction × task type) | Stability varies these for one framing. |
| gold = `UNKNOWN`, accuracy = abstention | gold = continuous `gold_risk`, fit = Pearson r / MAE | Risk has no abstention notion; "does the readout track gold" replaces "does it abstain". |
| 3-way `TernaryChoiceRunner` readout | 2-way `BinaryChoiceRunner` readout (calibrated `P(at risk)`) | CATEGORIZE only; SCORE prompts (a free number) carry the thinking level alone. |
| thinking role-distribution entropy / JS | thinking **score-cloud** entropy / std / abs-error-from-gold (`ScoreSummary`) | Divergence works on the continuous sampled-score cloud. |

## The five studies

| Study | Grid (per subject) | Collector | Visualizer |
| --- | --- | --- | --- |
| **baseline** | 1 prompt: canonical framing × one CATEGORIZE format | `baseline/collect_baseline_risk.py` | `baseline/visualize_baseline_risk.py` |
| **stability** | all format variation, one framing (label style × order flip × scale dir × task) | `stability/collect_stability_risk.py` | `stability/visualize_stability_risk.py` |
| **selection** | all framings, canonical CATEGORIZE format | `selection/collect_selection_risk.py` | `selection/visualize_selection_risk.py` |
| **divergence** | one framing, canonical format, many thinking draws | `divergence/collect_divergence_risk.py` | `divergence/visualize_divergence_risk.py` |
| **geometry** | all framings, canonical CATEGORIZE format + residual capture | `geometry/collect_geometry_risk.py` | `geometry/{visualize,analyze}_geometry_risk.py` + server |

All five prompt grids are generated in one run by
[`generate/generate_risk_prompt_datasets.py`](./generate/generate_risk_prompt_datasets.py),
which writes `out/mental_risk/<study>/prompt_dataset.json`.

## Workflow

```bash
# 1) Generate all five prompt datasets (needs the MentalRiskES corpus; see below).
uv run python mental_risk/generate/generate_risk_prompt_datasets.py \
    --corpus-dir datasets/corpusMentalRiskES --password-file secret.txt

# 2) Collect model readouts for a study (model: Qwen/Qwen3-0.6B by default).
uv run python mental_risk/baseline/collect_baseline_risk.py
uv run python mental_risk/stability/collect_stability_risk.py
uv run python mental_risk/selection/collect_selection_risk.py
uv run python mental_risk/divergence/collect_divergence_risk.py
uv run python mental_risk/geometry/collect_geometry_risk.py   # forces HF backend

# 3) Visualize.
uv run python mental_risk/baseline/visualize_baseline_risk.py
uv run python mental_risk/selection/visualize_selection_risk.py
# ... (one visualizer per study)

# Geometry: PCA analysis + interactive Plotly viz.
uv run python mental_risk/geometry/analyze_geometry_risk.py \
    out/mental_risk/geometry/Qwen3-0.6B/samples.json
mental_risk/geometry/visualize_geometry_risk.sh           # analyze + serve + open
```

Output layout (per study, keyed by bare model name):
`out/mental_risk/<study>/<MODEL>/{samples.json, plots/, activations/, analysis/}`.

## Shared helpers (top-level modules, reused by the drivers)

| Module | Purpose |
| --- | --- |
| `scaffolds_risk.py` | the canonical framing set (`get_risk_framings`, `framing_keys`). |
| `subject_resolution.py` | shared subject-source CLI flags + decrypt-or-load (`resolve_subjects`). |
| `risk_sample_io.py` | subsample-aware `RiskPromptDataset` loader (raw-json stride fast path). |
| `risk_prediction.py` | `effective_risk` — the comparable per-sample risk (non-thinking, else thinking). |
| `framing_ranking.py` | `score_framings` / `best_framing` — rank framings by gold-tracking. |

## Library code

- [`src/datasets/mental_risk/`](../src/datasets/mental_risk/) — corpus loader (subjects, gold collapse, archive decrypt).
- [`src/datasets/prompt/`](../src/datasets/prompt/) — `RiskPromptGenerator` and the framing / task / label-style content.
- [`src/datasets/risk/`](../src/datasets/risk/) — `RiskQuerier` + the two-level readout schemas.
- [`src/datasets/risk_geometry/`](../src/datasets/risk_geometry/) — geometry schemas + `capture_activations` + the PCA/projection analysis engine.

## Data blocker

The MentalRiskES corpus is **encrypted and absent** from this checkout (it is
gitignored and needs `MENTALRISK_ZIP_PASSWORD`). A real model run therefore
requires the decrypted corpus in the main checkout. The generator + every
collector/visualizer/analyzer is fully implemented and verified against synthetic
fixtures; only the live model+data run remains.
