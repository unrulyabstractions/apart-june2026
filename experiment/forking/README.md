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
`forking_dynamics.png`, `forking_branching_tree.png`, and a
`forking_positions/` directory with one `pos_<NNN>.json` per base-path token
position (the per-position RAW rollout dump — every sampled alternate's raw
continuation text + parsed outcome label + token info, written incrementally so a
crash keeps every completed position; use it to audit `unparseable` draws).

The **branching tree** (`forking_branching_tree.png`) renders the fork the way the
paper's Fig. 11 does: a left-to-right `root → trunk → branches` tree whose NODES
are horizontal-bar outcome distributions over `[target, other, unknown,
unparseable]` and whose EDGES are the alternate continuation tokens (width ∝
`p(x_t=w)`). The trunk is the detected forking token; the branches are its top
alternate continuations, each carrying its own `o_{t,w}`. The same renderer
(`render_branching_tree.py`) is reused by the divergence study.

## Files

| File | Role |
|------|------|
| `select_forking_item.py` | Stage 0: pilot decodes, rank ambiguous items by outcome entropy, persist the top item. |
| `collect_forking_rollouts.py` | Stages 1-3 (SINGLE box): capture `{O_t}` for the selected item, ONE batched decode over every branch. |
| `decode_forking_base_path.py` | SHARDED phase 1: decode the base path ONCE, enumerate every branch prefix, write `base_path.json`. |
| `collect_forking_shard.py` | SHARDED phase 2 (run on each of N boxes): fork ONLY positions `[t for t in range(P) if t%N==k]`, write `forking_shard_<k>_of_<N>.json`. Shard 0 also computes o_0. |
| `merge_forking_shards.py` | SHARDED phase 3 (LOCAL): reassemble every shard by REAL position index into one full-length `forking_trajectory.json` — any uncovered index is PADDED with a gap sentinel (loudly warned) so a 4-of-5 fleet stays aligned; a missing shard 0 / prior fails loudly. A drop-in for analyze/plot. |
| `analyze_forking_dynamics.py` | Stage 4: change-point + dynamic-state + diversity + survival analysis. |
| `plot_forking_dynamics.py` | Stage 5: the stacked-area figure + companion panels + the branching tree. |
| `plot_forking_comparison.py` | Scaffold-vs-baseline: load TWO trajectories of the SAME item (no-scaffold + scaffold) and render one figure — two stacked-area `O_t` rows + an overlaid `H(O_t)` / abstention-mass row. |
| `plot_forking_commit_dynamics.py` | Paper F6: a standalone minimal two-panel figure — stacked-area `O_t` over the full token index (top, Wilson 95% CI on the leading share) + a single answer-diversity curve `H(O_t)` (bottom). Same minimal styling as `forking_dynamics.png` (no suptitle / panel prose), single legend OUTSIDE each axis; reads only `forking_trajectory.json`, scales to any reasoning length. |
| `render_branching_tree.py` | Shared left-to-right branching-tree renderer (Fig. 11 style); reused by the divergence study. |
| `forking_plot_styles.py` | shared palette / token-strip / saver (presentation only). |
| `forking_item_io.py` | serialize/reload the selected `SesgoPromptSample` between drivers. |

The branching tree's data model (`BranchingTree` / `BranchingTreeNode`) and the
trajectory→tree builder (`build_branching_tree`) live with the analysis logic in
`src/dynamics/forking_paths/` (`forking_tree_model.py`, `forking_tree_builder_logic.py`).

## Tiny local pilot (verified, Qwen3-0.6B on MPS)

```bash
uv run python sesgo/forking/select_forking_item.py   --max-items 4 --n-pilot 6 --max-new-tokens 600
uv run python sesgo/forking/collect_forking_rollouts.py --n-samples 4 --n-prior 4 --max-positions 6 --max-new-tokens 400 --base-max-new-tokens 500
uv run python sesgo/forking/analyze_forking_dynamics.py
uv run python sesgo/forking/plot_forking_dynamics.py
```

## Local rich run (this deliverable, Qwen3-0.6B on MPS, ~100 min)

```bash
uv run python sesgo/forking/select_forking_item.py \
    --categories gender,racism,xenophobia --n-pilot 16 --max-items 16
HF_GEN_MICRO_BATCH=96 uv run python sesgo/forking/collect_forking_rollouts.py \
    --n-samples 40 --n-prior 60 --max-positions 60 \
    --max-new-tokens 384 --base-max-new-tokens 512
uv run python sesgo/forking/analyze_forking_dynamics.py
uv run python sesgo/forking/plot_forking_dynamics.py
```

`--max-positions` caps the branched leading base-path tokens (the expensive part)
so MPS finishes in ~100 min while still covering a 60-token series for change-point
detection. The cloud run is the SAME commands with `--max-positions 0` (all tokens),
`--n-prior 300`, on a vLLM CUDA box (the batched decode rides vLLM continuous
batching unchanged).

## Cloud scale-up

The expensive call is `collect_forking_rollouts.py`. Its branch decode goes through
`ModelRunner.continue_from_text_batch`, which uses the backend `generate_batch`
(vLLM continuous batching on CUDA, HF left-padded batch locally). Set
`--max-positions 0` and the paper sample counts; everything else is unchanged.

## Sharded fleet (parallelize forking across N≥5 boxes by POSITION)

For long base paths the forking is split across `NUM_SHARDS+1` cloud boxes:

```bash
# Three sharded phases (the orchestrator runs all of them):
# 1. ONE base box: select item + decode the base path -> base_path.json
uv run python sesgo/forking/decode_forking_base_path.py --model Qwen/Qwen3-14B
# 2. N shard boxes (each forks positions[k::N]) -> forking_shard_<k>_of_<N>.json
uv run python sesgo/forking/collect_forking_shard.py --model Qwen/Qwen3-14B \
    --shard-index 0 --num-shards 5
# 3. LOCAL merge -> forking_trajectory.json (drop-in for analyze/plot)
uv run python sesgo/forking/merge_forking_shards.py \
    --in-dir sync/forkmerged/Qwen3-14B --out-dir out/sesgo/forking/Qwen3-14B
```

One command drives the whole cloud fleet (base box, then N parallel shard boxes,
then local merge + analyze + plot). A shard box dying is logged but never aborts the
fleet — the merge step LOUDLY reports any missing positions:

```bash
MODEL=Qwen/Qwen3-14B NUM_SHARDS=5 bash cloud/run_forking_sharded_fleet.sh
```

Cloud scripts: `cloud/run_one_forking_base_box.sh` (phase 1),
`cloud/run_one_forking_shard_box.sh` (phase 2, `SHARD_INDEX`/`NUM_SHARDS` env),
`cloud/run_forking_sharded_fleet.sh` (orchestrator). Each box destroys itself on an
EXIT trap (never left billing) and quarantines results under `sync/`.

## Scaffold vs no-scaffold comparison (SAME item, both conditions)

To measure whether a debiasing scaffold changes the *dynamics* of commitment, run
the SAME forced item through the pipeline twice — once with no scaffold and once with
the scaffold prepended — then overlay both trajectories. Three flags on
`select_forking_item.py` make this a fair, GPU-free selection:

- `--force-question-id <qid>` — skip the entropy pilot and select the single
  ambiguous prompt with that `question_id` (no model needed for selection);
- `--scaffold <scaffold_id>` — prepend that scaffold's bilingual preamble (matched
  against `sesgo.scaffolds`), so the written sample's `scaffold_id` carries it;
- `--run-tag <suffix>` — suffix the bare-model out subdir
  (`out/sesgo/forking/<bare-model><suffix>/`) so the two conditions never collide.
  `collect`/`analyze`/`plot` take the SAME `--run-tag` to read/write that subdir.

```bash
# No-scaffold condition (forced item) -> out/sesgo/forking/Qwen3-0.6B/
uv run python sesgo/forking/select_forking_item.py --model Qwen/Qwen3-0.6B \
    --sesgo-dir datasets/SESGO --force-question-id <QID> --run-tag ""
# Scaffold condition -> out/sesgo/forking/Qwen3-0.6B-interpretive_direction/
# (use --run-tag=VALUE form: a leading '-' suffix is mis-parsed as a flag otherwise)
uv run python sesgo/forking/select_forking_item.py --model Qwen/Qwen3-0.6B \
    --sesgo-dir datasets/SESGO --force-question-id <QID> \
    --scaffold interpretive_direction --run-tag=-interpretive_direction
# ...collect/analyze/plot each with the matching --run-tag, then:
uv run python sesgo/forking/plot_forking_comparison.py   # -> Qwen3-0.6B_scaffold_vs_baseline.png
```

On the cloud, `cloud/run_one_forking_box_32b.sh` forwards three env vars to the
on-box select/collect/analyze/plot stages: `SELECT_SCAFFOLD`, `SELECT_FORCE_QID`,
and `RUN_TAG` (all default empty == today's full-pilot single-box behaviour). Launch
the two conditions as two boxes with disjoint `RUN_TAG`s.
