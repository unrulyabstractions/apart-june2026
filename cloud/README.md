# cloud/ — run the FULL SESGO divergence + stability collections on a Vast.ai GPU

These scripts stand up a single CUDA GPU box on [Vast.ai](https://vast.ai), run the
**full** (no-subsample) divergence and stability SESGO collections there, and bring
the results back **safely**. They are adapted from the `constellation_takehome`
Vast workflow.

> **DRAFT — nothing here has been run.** Read the scripts, then a human launches
> after review. Only `vast_launch.sh` (and `vast_destroy.sh`) ever spend money.

## The safety guarantee (why this is safe)

The danger with cloud runs is the **sync-back** direction (cloud → local): a naive
`rsync` onto your `out/` can **overwrite or delete** local results. These scripts
make that impossible:

- The cloud's results are pulled into a **local, gitignored `sync/` quarantine
  directory** — **never** into `out/` and never onto code.
- The pull uses `rsync --ignore-existing` (an already-present local file is
  **never** overwritten) and has **no `--delete`** (nothing local is ever removed).
  The worst the cloud can do is **add new files under `sync/`.**
- Promoting results into `out/` is a **separate, purely-local** step
  (`merge_sync.sh`) that a human runs after inspecting `sync/`. It also uses
  `--ignore-existing` / no `--delete`, so it can only **add** to `out/`.

Flow:  **cloud `out/sesgo/` → local `sync/` (--ignore-existing) → human inspects →
`merge_sync.sh` → `out/`.**

`sync/` is gitignored, so copied-back results are never committed.

## Files

| Script             | Runs where | What it does |
|--------------------|------------|--------------|
| `vast_launch.sh`   | local      | Search offers, create a GPU instance, poll until running, write the id to `cloud/.vast_instance_id`. **Spends money.** |
| `at_vast.sh`       | local      | Resolve the SSH host fresh and run one command on the box. Used to invoke the remote `at_*` scripts. |
| `sync_up.sh`       | local      | rsync the repo **up** to the box, excluding `out/ datasets/ sync/ .git .venv __pycache__ *.pyc paper/build`. Plus a narrow extra sync of `datasets/SESGO/prompts/*.xlsx` (required generation input). Safe direction; no `--delete`. |
| `at_setup.sh`      | **remote** | `uv sync`, then **reinstall torch 2.6.0+cu124** (with its nvidia-cu12 runtime libs) over the cu130 wheel `uv sync` pulls, so CUDA initializes on the mixed cuda-12/cuda-13 fleet. All downstream steps use `.venv/bin/python` **directly** (NOT `uv run`, which re-syncs and reverts the pin). Picks up `HF_TOKEN`. Device check aborts the box if CUDA is unavailable. |
| `at_run.sh`        | **remote** | Regenerate the 5 prompt datasets, then the **FULL** divergence + stability collections (no `--subsample`). Writes only to remote `out/sesgo/{divergence,stability}/`. |
| `sync_back.sh`     | local      | **The safe pull.** rsync remote `out/sesgo/{divergence,stability}/` → local `sync/sesgo/...` with `--ignore-existing`, no `--delete`. Only ever adds new files to `sync/`. |
| `merge_sync.sh`    | local      | Promote NEW files from `sync/` into `out/` (`--ignore-existing`, no `--delete`). Purely local; run after inspecting `sync/`. `--move` clears promoted files from `sync/`. |
| `vast_destroy.sh`  | local      | Tear down the instance (or `--stop` to pause). Requires `--yes-i-am-really-sure`. |

The remote repo lives at `/root/apart`. Override with `REMOTE_ROOT=...` on any script.
The instance id is read from `cloud/.vast_instance_id` (or `INSTANCE=<id>` env override).

## Prerequisites

```bash
pip install vastai
vastai set api-key <YOUR_KEY>        # from the Vast Keys page
# Register your SSH public key (~/.ssh/id_ed25519.pub) on the Vast Keys page.
```

## The exact commands a human runs

```bash
# 1. LAUNCH a GPU box (spends money; asks y/N before creating)
bash cloud/vast_launch.sh

# 2. PUSH code + SESGO prompt sources up to the box
bash cloud/sync_up.sh

# 3. SET UP the env on the box (uv sync). Pass HF_TOKEN only if you need it:
bash cloud/at_vast.sh "bash cloud/at_setup.sh"
#   with a token:
#   HF_TOKEN=hf_xxx bash cloud/at_vast.sh "HF_TOKEN=$HF_TOKEN bash cloud/at_setup.sh"

# 4. RUN the FULL divergence + stability collections on the box
bash cloud/at_vast.sh "bash cloud/at_run.sh"          # both
#   or one at a time:
#   bash cloud/at_vast.sh "bash cloud/at_run.sh divergence"
#   bash cloud/at_vast.sh "bash cloud/at_run.sh stability"

# 5. SYNC BACK — SAFE. Pulls only NEW files into local sync/ (never out/, never overwrite)
bash cloud/sync_back.sh
#   preview first if you like: DRY_RUN=1 bash cloud/sync_back.sh

# 6. INSPECT the quarantined results by hand
find sync -type f

# 7. MERGE the inspected new files into out/ (local-only, never overwrites)
bash cloud/merge_sync.sh
#   preview: DRY_RUN=1 bash cloud/merge_sync.sh
#   or copy-then-clear sync/: bash cloud/merge_sync.sh --move

# 8. DESTROY the box (stops all billing). Dry-run without the flag:
bash cloud/vast_destroy.sh
bash cloud/vast_destroy.sh --yes-i-am-really-sure
```

## Notes

- **Model / backend.** Qwen3-0.6B is public (no HF token needed normally). On the
  Linux GPU box MLX is not installed, so `get_recommended_backend_inference()`
  returns the HuggingFace backend, which uses CUDA automatically. The custom image
  also ships the **vLLM** backend (continuous batching) for the generation fast path.
- **FULL runs.** `at_run.sh` passes no `--subsample`, so every prompt in the
  regenerated datasets is queried — the complete collections.
- **No `--delete` anywhere.** None of these rsyncs delete files, in either
  direction.

---

## Parallel per-model FLEET (run ALL models at once)

The single-box flow above runs one model. The **fleet** scripts run *every* model
**concurrently** — N models ⇒ N boxes running at the same time — so total
wall-clock ≈ the slowest single (model, shard) box, not the sum.

| Script                    | Runs where | What it does |
|---------------------------|------------|--------------|
| `fleet_sizing.py`         | local      | Single source of truth: model→GPU sizing map (`<=8B`→1× RTX 4090, `<=16B`→1× A100, `<=40B`→1× H100 80GB, else 2× H100 SXM) + shard plan. `python cloud/fleet_sizing.py plan` emits one TSV row per `(model, shard)` box; the trailing `num_gpus` column drives multi-GPU boxes (e.g. Llama-3.1-70B on 2× H100). |
| `fleet_launch.sh`         | local      | Fires every `vast create` **in parallel** (background), polls concurrently. Records ids under `cloud/.fleet/<tag>.id`. **Spends money.** |
| `fleet_run.sh`            | local      | Drives all boxes **concurrently**: per box → `sync_up` → `at_setup` → run its model+shard pipeline → safe sync-back to its own quarantine → **self-destruct**. Per-box log at `cloud/.fleet/<tag>.log`. Passes `STUDIES`/`BATCH_SIZE`/`N_THINKING`/`SUBSAMPLE`/`MAX_NEW_TOKENS`/`ITEMS`/`GENERATE_ALL_DATA` through to the on-box driver. |
| `fleet_model_run.sh`      | **remote** | On-box per-`(model, shard)` driver: runs the batched, sharded studies (`--batch-size` drives vLLM continuous batching). `SUBSAMPLE` (0-1, unset == full grid) threads `--subsample` into the thinking studies (divergence/selection/geometry) so a run queries a strided fraction of the grid; every shard subsamples the SAME slice before taking its shard, so the K shards still tile one subsampled grid. `GENERATE_ALL_DATA=1` with `STUDIES=baseline_full` builds + runs the FULL-DATA baseline grid (all langs × all origins × {none+3 scaffolds}, 24480 prompts) into its own `out/sesgo/baseline_full/` tree. |
| `fleet_destroy.sh`        | local      | Concurrent backstop teardown (in case a box hung). Needs `--yes-i-am-really-sure`. |

```bash
# Inspect the plan, launch the whole fleet concurrently, drive it, then merge.
python cloud/fleet_sizing.py plan
bash   cloud/fleet_launch.sh                 # all boxes at once (spends money)
bash   cloud/fleet_run.sh                    # setup+run+sync-back+self-destruct, in parallel
find   sync -type f                          # inspect the per-box quarantines
bash   cloud/merge_sync.sh                    # promote into out/ (--ignore-existing)
bash   cloud/fleet_destroy.sh --yes-i-am-really-sure   # backstop (usually a no-op)
```

### Full-data baseline (`baseline_full`): all languages × all origins × scaffolds

The full SESGO grid (es+en × original+BBQ-adapted × {none + 3 scaffolds} = 24480
prompts) runs as the `baseline_full` study on a heavily-sharded RTX 4090 fleet. It
writes to its OWN `out/sesgo/baseline_full/` tree, so it never clobbers the
es-original runs:

```bash
FLEET_DIR=cloud/.fleet_fulldata FLEET_MODELS=Qwen/Qwen3-0.6B FLEET_SHARDS=16 \
  bash cloud/fleet_launch.sh
FLEET_DIR=cloud/.fleet_fulldata STUDIES=baseline_full GENERATE_ALL_DATA=1 BATCH_SIZE=64 \
  bash cloud/fleet_run.sh    # generate full grid on-box, run, sync-back, self-destruct
```

### Optional dataset sharding (one model across K boxes)

For the largest/slowest model, set its `shards` in `fleet_sizing.py` to K (or
`FLEET_SHARDS=K`). The grid is split into **K disjoint, contiguous slices**
(`shard_slicing.py`); box k runs only shard k and writes to
`out/sesgo/<study>/<model>/shard_<k>_of_<K>/`. Default is **1 shard** (full grid,
plain path) for small models.

Each shard's `response_samples.json` is a **disjoint** slice of the same
(optionally subsampled) grid, so they must be recombined identity-aware, NOT via
`merge_sync.sh` (which `--ignore-existing` would keep only the first shard).
`sesgo/geometry/combine_geometry_shards.py` does this for geometry (concatenate +
copy activation `.pt`s); `sesgo/divergence/combine_divergence_shards.py` is the
divergence analogue — pure sample-level, no tensors — concatenating the per-shard
`SesgoSample`s de-duped by `sample_identity`:

```bash
uv run python sesgo/divergence/combine_divergence_shards.py Qwen3-32B
uv run python sesgo/divergence/visualize_divergence_samples.py \
    out/sesgo/divergence/Qwen3-32B/response_samples.json
```

### Safe concurrent sync-back — NEVER overwrite blindly (the guarantee, under concurrency)

Each box writes ONLY to its own **disjoint** slice
`out/sesgo/<study>/<bare-model>/[shard_k_of_K]/`, so concurrent boxes never target
the same file. Sync-back pulls into the gitignored `sync/` quarantine ONLY, via
`rsync --ignore-existing` (never `--delete`, never onto `out/`); each box lands in
its OWN `sync/box-<tag>/` subtree (`SYNC_SUBDIR`), so simultaneous pulls stay
disjoint. The `sync/ → out/` merge is also `--ignore-existing` and inspectable, so
**a re-run, a retry, or a parallel box can NEVER clobber an existing result** — an
existing local slice is **kept** (the cloud copy is ignored) and surfaced to the
human, never silently overwritten.

### Custom image + weight pre-cache (seconds-long box setup)

`Dockerfile` bakes torch + vLLM (the CUDA-only `cloud` extra) + repo deps **and**
pre-cached gated weights, so per-box setup is seconds, not a cold multi-GB pull.

```bash
# BUILD on an x86_64 CUDA host (HF_TOKEN as a build secret for the gated repos):
docker build -f cloud/Dockerfile --secret id=hf,env=HF_TOKEN -t <registry>/sesgo-vllm:latest .
docker push <registry>/sesgo-vllm:latest
# USE: vast pulls it on create when you point the fleet at the tag:
IMAGE=<registry>/sesgo-vllm:latest bash cloud/fleet_launch.sh
```

`prefetch_model_weights.py` (also baked into the image) pre-downloads
Llama-3.2-1B / gemma-2-2b / Mistral-7B (+ Qwen3-0.6B) with `HF_TOKEN`; the repo
list comes from the fleet plan so the two never drift.

> The fleet scripts are **drafts**: nothing is launched and the image is **not
> built or pushed** here. A human runs them after review. vLLM is **CUDA-only** and
> does not run on Apple Silicon — the batching logic is verified locally through the
> equivalent HuggingFace batched path.

### Known gotchas (hard-won, baked into the scripts)

- **torch CUDA build mismatch.** `uv sync` pulls the latest torch wheel
  (currently +cu130), which fails CUDA init on any host whose driver predates
  CUDA 13. `at_setup.sh` reinstalls **torch 2.6.0+cu124** (WITH its nvidia-cu12
  runtime libs — never `--no-deps`, or torch import dies with
  `libcudart.so.11.0: cannot open shared object file`). cu124 runs on every
  driver supporting CUDA ≥ 12.4, so the offer search filters `cuda_max_good>=12.4`.
- **`uv run` reverts the pin.** Every `uv run` re-syncs the venv to the lockfile
  first, silently reinstalling the cu130 wheel. `UV_NO_SYNC=1` did **not** prevent
  this in practice. So all on-box python steps invoke `.venv/bin/python` directly.
- **SSH-not-ready race.** A box reports `running` (v1 API) before sshd is
  reachable; pushing then dies mid-rsync and leaves a half-synced box (no prompt
  xlsx → generate writes 0 items → an EMPTY `response_samples.json`). `fleet_run`
  now probes real SSH (`wait_ssh`), retries `sync_up`+`at_setup`, and
  `fleet_model_run` aborts if a prompt dataset has 0 prompts.
- **Disk too small for big weights.** 24-32B bf16 weights + the HF xet cache's
  temp copy overflow a 60 GB box (`No space left on device` mid-download). Disk is
  sized per model in `fleet_sizing.py` (60/120/200/320 GB).
- **Models too big for one GPU (70B).** A 70B in bf16 (≈ 140 GB) does NOT fit a
  single 80 GB card. `fleet_sizing.py`'s huge tier requests **2× H100 SXM**
  (`num_gpus` column → `num_gpus>=2` in the offer query), and `fleet_run.sh`
  forwards **`HF_DEVICE_MAP=auto`** so the HuggingFace backend shards the weights
  across both GPUs (Accelerate device hooks). The SESGO querier forces the HF
  backend, so this is the multi-GPU path; the single-device `.to(device)` is
  skipped whenever `HF_DEVICE_MAP` is set (it would otherwise collapse the model
  onto one GPU and OOM). Use a small `BATCH_SIZE` (≈8) for the 70B — the cross-GPU
  activations leave little headroom.
- **OOM on undersized GPU.** Baseline's greedy-decode trajectory step at
  `BATCH_SIZE=32` OOM'd an 8B on a 24 GB RTX 4090. Put borderline 8B runs on a
  40 GB A100 and/or drop `BATCH_SIZE` — reliability over throughput.
- **Vast preemption.** Cheap interruptible offers get outbid mid-run
  (`intended_status` flips to `stopped`); the box stops and only a partial
  checkpoint survives. Prefer `reliability>=0.95` offers and just re-launch the
  affected models. Always sweep `vastai show instances-v1` at the end and destroy
  any stragglers (`cloud/fleet_destroy.sh --yes-i-am-really-sure`, or by id).
- **Box-replacement resume.** `collect` already checkpoints
  `response_samples.json` every 25 samples and, on restart, *resumes* from an
  existing one (the querier loads it and skips completed identities), so a box
  restarting its OWN collect resumes for free. To make *replacement* resumable —
  when a box dies and a fresh one takes its slice — `fleet_run.sh` runs a
  background **partial puller** (`cloud/sync_partial.sh pull`, every
  `PARTIAL_SYNC_EVERY=180`s) that mirrors the growing `response_samples.json` down
  to a throwaway `sync/partial-box-<tag>/` tree, and **pushes** the newest partial
  back up (`sync_partial.sh push`) *before* the run on each (re)launch so the fresh
  box's collect resumes instead of restarting from 0. The partial tree is never
  promoted (`merge_sync` globs `box-*/` only), so it can never reach `out/`. The
  thinking studies (divergence/selection/stability) resume fully; geometry only
  resumes samples whose side-car `.pt` activation tensors survive (it recomputes
  the rest), so its primary resume path is a same-box restart.

---

## Expected wall-clock + cost (full 2040-item grid)

Headline study = **divergence** (the heaviest: thinking-only, `n_thinking=8`
draws × ~512 generated tokens per draw). Per model the decode work is

  2040 items × 1 prompt × 8 draws × 512 tok ≈ **8.35 M generated tokens**

plus a cheap teacher-forced `choose3` prefill per item (negligible next to decode).

**Assumptions** (order-of-magnitude; tune to your offers):
- single-stream HF decode tok/s on an RTX 4090: ~40 (1–2 B), ~28 (3 B), ~20 (7 B).
- vLLM continuous-batching throughput tok/s (batch ~64–256) on the same GPU:
  ~2600 (1 B), ~2000 (2–3 B), ~950 (7 B) — ≈ 50–65× single-stream.
- vast RTX 4090 ≈ **$0.40/hr** (A100 ≈ $1.40, H100 ≈ $3.00). All four fleet
  models are ≤ 8 B ⇒ one RTX 4090 box each.
- Fixed per-box overhead (image pull warm + uv-sync no-op + prompt-gen) ≈ 3 min.

### Per model, divergence, RTX 4090 — WITHOUT vs WITH batching

| Model (params)         | tokens | single-stream HF | vLLM batched | box $ (vLLM) |
|------------------------|--------|------------------|--------------|--------------|
| Llama-3.2-1B  (1.2 B)  | 8.35 M | ~58 h            | ~57 min      | ~$0.38 |
| gemma-2-2b    (2.6 B)  | 8.35 M | ~70 h*           | ~73 min      | ~$0.49 |
| Mistral-7B    (7.2 B)  | 8.35 M | ~116 h           | ~150 min     | ~$1.00 |
| Qwen3-0.6B    (0.6 B)  | 8.35 M | ~52 h            | ~56 min      | ~$0.37 |

\* gemma-2-2b single-stream ~33 tok/s. Single-stream hours are why the original
single-box flow subsampled divergence (`DIV_SUB=0.5`); batching makes the FULL
grid finish in ~1–2.5 h, so no subsample is needed.

### Fleet total — parallel vs sequential

The fleet launches all four boxes **concurrently** and each self-destructs when
done, so the **total wall-clock ≈ the slowest single box**, not the sum:

| Mode                         | wall-clock | total $ |
|------------------------------|------------|---------|
| Sequential, single-stream HF | ~296 h (~12 days) | ~$118 |
| Sequential, vLLM batched     | ~5.6 h     | ~$2.24 |
| **Parallel fleet, vLLM batched** | **~2.5 h (= Mistral box)** | **~$2.24** |

Same dollar cost as sequential (you pay per box-hour either way), but ~2.3× less
wall-clock than running the batched boxes back-to-back, and ~120× less than the
single-stream sequential baseline.

### Optional sharding (split the slowest model)

If Mistral's ~2.5 h is the long pole, set its `shards=3` in `fleet_sizing.py`:
three 4090 boxes each run a disjoint ~680-item third (~50 min each), pulling the
fleet wall-clock down to ≈ **1.2 h** (the next-slowest box) for ~3× Mistral
box-cost (~$3 total). The other models stay at 1 shard. The lighter studies
(baseline / stability: one forward pass per prompt) are minutes per model and are
dominated by the fixed per-box overhead — batching still helps but the grid there
is cheap regardless.
