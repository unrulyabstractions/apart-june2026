"""Pre-download the SESGO fleet's model weights into the HF cache.

Baked into the cloud image (or run once on a box) so per-box startup is seconds,
not a cold multi-GB pull mid-run. Gated repos (Llama / Gemma / Mistral) need a
read token: set HF_TOKEN in the environment. Public repos (Qwen3-0.6B) need no
token. Models come from the fleet sizing plan so the two never drift apart.

Usage:
  HF_TOKEN=hf_xxx python cloud/prefetch_model_weights.py            # the whole fleet
  HF_TOKEN=hf_xxx python cloud/prefetch_model_weights.py MODEL ...  # explicit repos
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, __file__.rsplit("/cloud/", 1)[0])

from huggingface_hub import snapshot_download  # noqa: E402

from cloud.fleet_sizing import default_fleet  # noqa: E402

# Allow patterns: weights + tokenizer + config; skip redundant fp32/onnx variants
# that vLLM/transformers never load, to keep the cache (and image) lean.
_ALLOW = ["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"]


def _models_to_fetch() -> list[str]:
    """Explicit repos from argv, else every model in the default fleet plan."""
    if len(sys.argv) > 1:
        return sys.argv[1:]
    return [m.model for m in default_fleet()]


def prefetch(model: str, token: str | None) -> None:
    """Snapshot one repo into the local HF cache (resumable, allow-listed)."""
    print(f"[prefetch] {model}")
    snapshot_download(
        repo_id=model,
        allow_patterns=_ALLOW,
        token=token,
        # Default HF cache; the image bakes this layer so boxes start warm.
    )
    print(f"[prefetch] done: {model}")


def main() -> None:
    """Pre-download every requested model; gated repos use HF_TOKEN."""
    token = os.environ.get("HF_TOKEN")
    if token is None:
        print("[prefetch] HF_TOKEN unset — gated repos (Llama/Gemma/Mistral) will fail.")
    for model in _models_to_fetch():
        prefetch(model, token)
    print("[prefetch] all done.")


if __name__ == "__main__":
    main()
