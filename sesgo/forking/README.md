# `sesgo/forking/` — Forking-paths dynamics study (run-by-path drivers)

Four run-by-path drivers that take ONE ambiguous SESGO item through the full
forking-paths pipeline: pick the item, capture its per-token outcome distribution
`{O_t}`, analyze it (change point + dynamic states + diversity + survival), and
plot it. All computation lives in `src/dynamics/forking_paths/`; these scripts are
thin model + I/O orchestration.

The study answers: *as the model reasons token-by-token on an ambiguous bias item,
where (which token) does its committed answer become locked in, and how does the
outcome distribution collapse from "not enough information" toward a group?*

## Pipeline (run in order)

```bash
# 0. Pick the highest-outcome-entropy ambiguous item (most likely to FLIP).
uv run python sesgo/forking/select_forking_item.py \
    --model Qwen/Qwen3-0.6B --categories gender --n-pilot 12 --max-new-tokens 600

# 1-3. Capture {O_t}: greedy base path + batched branch sampling at every token.
uv run python sesgo/forking/collect_forking_rollouts.py \
    --model Qwen/Qwen3-0.6B --n-samples 40 --n-prior 300 --max-new-tokens 512

# 4. Analyze: change-point detection + pull/drift/potential + diversity + survival.
uv run python sesgo/forking/analyze_forking_dynamics.py --model Qwen/Qwen3-0.6B

# 5. Plot: stacked-area O_t + token strip + companion panels.
uv run python sesgo/forking/plot_forking_dynamics.py --model Qwen/Qwen3-0.6B
```

Outputs land under `out/sesgo/forking/<MODEL>/`:
`selected_item.json`, `forking_trajectory.json`, `forking_analysis.json`,
`forking_dynamics.png`.

## Files

| File | Role |
|------|------|
| `select_forking_item.py` | Stage 0: pilot decodes, rank ambiguous items by outcome entropy, persist the top item. |
| `collect_forking_rollouts.py` | Stages 1-3: capture `{O_t}` for the selected item, ONE batched decode over every branch. |
| `analyze_forking_dynamics.py` | Stage 4: change-point + dynamic-state + diversity + survival analysis. |
| `plot_forking_dynamics.py` | Stage 5: the stacked-area figure + companion panels. |
| `forking_plot_styles.py` | shared palette / token-strip / saver (presentation only). |
| `forking_item_io.py` | serialize/reload the selected `SesgoPromptSample` between drivers. |

## Tiny local pilot (verified, Qwen3-0.6B on MPS)

```bash
uv run python sesgo/forking/select_forking_item.py   --max-items 4 --n-pilot 6 --max-new-tokens 600
uv run python sesgo/forking/collect_forking_rollouts.py --n-samples 4 --n-prior 4 --max-positions 6 --max-new-tokens 400 --base-max-new-tokens 500
uv run python sesgo/forking/analyze_forking_dynamics.py
uv run python sesgo/forking/plot_forking_dynamics.py
```

The cloud run is the SAME commands with `--max-positions 0` (all tokens),
`--n-samples 30-40`, `--n-prior 300`, `--max-new-tokens 512`, on a vLLM CUDA box
(the capture's batched decode rides vLLM continuous batching unchanged).

## Cloud scale-up

The expensive call is `collect_forking_rollouts.py`. Its branch decode goes through
`ModelRunner.continue_from_text_batch`, which uses the backend `generate_batch`
(vLLM continuous batching on CUDA, HF left-padded batch locally). Set
`--max-positions 0` and the paper sample counts; everything else is unchanged.
