# `risk_geometry` — detailed flow

The geometry study asks: *where, in the model's representation, does the FRAMING
move a risk judgement?* This package supplies (1) the schemas that record the
answer and (2) the engines that capture and analyze it. It is the risk twin of
the bias-role geometry trio in `src/datasets/sesgo_eval/`.

## Capture (`risk_activation_capture.py`)

`capture_activations(runner, prompt, layers, sample_dir, rel_root)`:

1. Build the **greedy non-thinking answer path**: chat-templated prompt +
   `skip_thinking_prefix` (the empty `<think></think>` block) + the prompt's
   `choice_prefix` + the model's own temperature-0 continuation. The first
   diverging token index (vs the prefix-only ids) is the `answer` position —
   robust to leading-space BPE merges.
2. One forward pass with a `resid_post` `names_filter` captures the per-layer
   residual stream.
3. `find_positions` locates four structural tokens: the last `<|im_start|>`
   (`turn`), the `<think>` / `</think>` tokens (`think_open` / `think_close`),
   and the `answer` index. Each found position's `[n_layers, d_model]` slice is
   `torch.save`'d; the `RiskGeometryActivation` keeps only the relative path.

Only the **HuggingFace** backend exposes `run_with_cache`, so the geometry
collector constructs `BinaryChoiceRunner(model_name=..., backend=HUGGINGFACE)`
exactly as SESGO forces it on `TernaryChoiceRunner`.

## Analyze (`risk_geometry_analysis.py`)

`analyze_position(dataset, root, ptype, layer, n_components, seed)`:

1. `build_matrix` stacks the per-sample residual at one position (reduced over
   layers via `last` / `mean` / an int index) into `[n_valid, d_model]`.
2. `run_pca` fits a mean-centering PCA, `k` clamped to `min(n_components, n, d)`.
3. `framing_stats` computes, in **full PCA space**, per-framing centroids, the
   shift vector + L2 magnitude of each framing from the **anchor framing** (the
   alphabetically-first one — there is no no-op baseline), the pairwise centroid
   distance matrix, a framing silhouette, and the between/within scatter ratio.
4. `axis_separation` reports silhouette + between/within for `framing`,
   `disorder`, and `language`.

Positions with fewer than `MIN_SAMPLES` valid rows are skipped, so the resulting
`projections.json` always loads even on subsampled or degenerate data. The driver
`mental_risk/geometry/analyze_geometry_risk.py` is a thin orchestrator over these
functions, and the FastAPI server + static visualizer consume the JSON they emit.

## Risk-vs-bias differences

- **Continuous gold.** Samples expose `predicted_risk_non_thinking` /
  `predicted_risk_thinking`, not an abstention-correctness flag.
- **Binary readout.** `NonThinkingResult` (calibrated `P(at risk)`), not the
  3-way `SesgoNonThinking`.
- **Anchor, not baseline.** Shift vectors are anchored on the first framing,
  because no framing is a no-op.
