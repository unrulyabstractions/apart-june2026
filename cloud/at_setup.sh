#!/usr/bin/env bash
#
# at_setup.sh — environment setup that runs ON the Vast.ai box.
#
# Pushed up by sync_up.sh, then invoked remotely, e.g.:
#   bash cloud/at_vast.sh "bash cloud/at_setup.sh"
#
# It installs uv (if missing), resolves the project's locked dependencies with
# `uv sync`, and prints a device sanity check. The apart pyproject pins
# torch>=2.6.0 / torchvision>=0.21.0 from PyPI; on a Linux+CUDA box that pulls a
# CUDA-enabled wheel. MLX is marked `sys_platform == 'darwin'` only, so it is NOT
# installed here — Qwen3-0.6B therefore auto-selects the HuggingFace/CUDA backend
# (src/inference/backends/backend_selection.py: not Apple Silicon -> HUGGINGFACE).
#
# HF_TOKEN: Qwen3-0.6B is a public model so a token is usually unnecessary, but if
# you need gated access export HF_TOKEN in your shell before calling at_vast.sh —
# it propagates over SSH only if you pass it through (see cloud/README.md).

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root on the remote (/root/apart)

# ── 1. Ensure uv is on PATH ────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  echo "[at_setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ── 2. Resolve + install locked deps (creates .venv on the box) ────────
# `uv sync` builds the venv from pyproject.toml + uv.lock for reproducibility.
echo "[at_setup] uv sync"
uv sync

# uv sync pulls the LATEST torch wheel (currently a +cu130 build) which fails CUDA
# init on Vast hosts whose driver predates CUDA 13 (the fleet draws a MIX of cuda-12
# and cuda-13 hosts). A CUDA 12.4 build runs on every driver that supports CUDA >=
# 12.4 -- i.e. ALL of them (the cheapest Vast 4090/A100/H100 hosts report
# cuda_max_good 12.4..13.x). We reinstall the cu124 wheel so the GPU is usable on
# every box.
#
# CRITICAL: do NOT pass --no-deps here. The torch wheel does NOT bundle the CUDA
# runtime; it depends on the nvidia-*-cu12 pip packages (libcudart, libcublas, ...).
# A --no-deps install left those out and torch import died with
# "libcudart.so.11.0: cannot open shared object file". Installing WITH deps pulls
# the matching nvidia-*-cu12 libs from the same pytorch index, so CUDA initializes.
echo "[at_setup] pinning torch 2.6.0+cu124 (+ its nvidia-cu12 runtime libs)"
uv pip install --reinstall "torch==2.6.0" "torchvision==0.21.0" \
  --index-url https://download.pytorch.org/whl/cu124

# FP8 checkpoints (Mistral Ministral-3 ship finegrained-fp8 weights) need the `kernels`
# package to dequant on the HF backend — BUT only for those boxes, and the version must
# match transformers (a too-new `kernels` breaks `transformers.integrations.hub_kernels`
# import entirely with "Either a revision or a version must be specified"). So it is opt-in
# via INSTALL_KERNELS=1 and PINNED, never installed unconditionally.
if [ "${INSTALL_KERNELS:-0}" = 1 ]; then
  echo "[at_setup] installing pinned kernels (FP8 dequant for Mistral FP8 checkpoints)"
  uv pip install "kernels==${KERNELS_VERSION:-0.4.4}"
fi

# CRITICAL: `uv run` ALWAYS re-syncs the venv to the lockfile first, which silently
# reinstalls the cu130 wheel and undoes the pin above (confirmed in fleet logs:
# even with UV_NO_SYNC=1 exported, `uv run python ...` re-downloaded torch 2.11+cu130
# before our script ran). The robust fix is to STOP using `uv run` for anything that
# touches torch and invoke the venv interpreter DIRECTLY (.venv/bin/python). The venv
# is already fully built by `uv sync`, so direct invocation is correct and never
# re-syncs. We expose it as PY for every downstream step.
PY="$PWD/.venv/bin/python"
[ -x "$PY" ] || { echo "[at_setup] FATAL: $PY missing after uv sync"; exit 1; }
echo "[at_setup] using venv interpreter directly: $PY"

# ── 3. Propagate HF_TOKEN into the env if the caller exported it ───────
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[at_setup] HF_TOKEN is set; huggingface-hub will pick it up from env."
else
  echo "[at_setup] HF_TOKEN not set (fine for the public Qwen3-0.6B)."
fi

# ── 4. Device sanity check (via the venv python directly, NOT `uv run`) ─
echo "[at_setup] device check"
"$PY" - <<'PYEOF'
import sys
import platform
import torch
print("torch:   ", torch.__version__)
print("platform:", platform.platform())
ok = torch.cuda.is_available()
print("cuda:    ", ok, "(", torch.cuda.get_device_name(0) if ok else "NONE", ")")
if not ok:
    print("[at_setup] FATAL: CUDA unavailable after the cu124 pin -- aborting so "
          "this box never runs the model on CPU.")
    sys.exit(1)
PYEOF

echo "[at_setup] done."
