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

# ── 3. Propagate HF_TOKEN into the env if the caller exported it ───────
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[at_setup] HF_TOKEN is set; huggingface-hub will pick it up from env."
else
  echo "[at_setup] HF_TOKEN not set (fine for the public Qwen3-0.6B)."
fi

# ── 4. Device sanity check ─────────────────────────────────────────────
echo "[at_setup] device check"
uv run python - <<'PY'
import platform
import torch
print("torch:   ", torch.__version__)
print("platform:", platform.platform())
print("cuda:    ", torch.cuda.is_available(),
      "(", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", ")")
try:
    import mlx.core  # noqa: F401
    print("mlx:      installed (UNEXPECTED on a Linux GPU box)")
except ImportError:
    print("mlx:      not installed (expected) -> HuggingFace/CUDA backend")
PY

echo "[at_setup] done."
