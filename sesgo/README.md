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

# selection — per-scaffold accuracy, non-thinking + thinking
uv run python sesgo/selection/collect_selection_samples.py  [--subsample 0.05 --n-thinking 4]
uv run python sesgo/selection/visualize_selection_samples.py <RS>

# divergence — system-default distribution over hot-sampled thinking draws
uv run python sesgo/divergence/collect_divergence_samples.py [--subsample 0.05 --n-thinking 8]
uv run python sesgo/divergence/visualize_divergence_samples.py <RS>

# geometry — residual-stream capture along the greedy path (forces the HF backend)
uv run python sesgo/geometry/collect_geometry_samples.py    [--subsample 0.1 --n-thinking 0]
uv run python sesgo/geometry/analyze_geometry.py            <RS>   # PCA + per-axis separation
uv run python sesgo/geometry/visualize_geometry_samples.py  <RS>   # PCA grid (every label axis)
PORT=8002 bash sesgo/geometry/visualize_geometry.sh         <RS>   # interactive viz server
```
Common flags: `--model <hf-id>` (default `Qwen/Qwen3-0.6B`), `--subsample 0..1`,
`--batch-size N` (batched forward), `--shard-index/--shard-count` (split the grid).
Collection checkpoints to `response_samples.json` as it goes — re-run the same
command to **resume** after a crash.

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
