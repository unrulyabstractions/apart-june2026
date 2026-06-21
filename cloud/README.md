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
| `at_setup.sh`      | **remote** | `uv sync` (CUDA torch from PyPI; no MLX on Linux → HuggingFace/CUDA backend). Picks up `HF_TOKEN` from env if set. Device sanity check. |
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
  returns the HuggingFace backend, which uses CUDA automatically.
- **FULL runs.** `at_run.sh` passes no `--subsample`, so every prompt in the
  regenerated datasets is queried — the complete collections.
- **No `--delete` anywhere.** None of these rsyncs delete files, in either
  direction.
