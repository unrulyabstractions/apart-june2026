# Custom SESGO cloud image: torch + vLLM + repo deps baked in, weights pre-cached.
#
# WHY: a vanilla pytorch image makes every box `uv sync` (minutes) and cold-pull
# multi-GB weights mid-run. Baking the deps AND the gated model snapshots makes
# per-box setup seconds-long, so the parallel fleet's wall-clock is dominated by
# the actual collection, not startup.
#
# BUILD (on an x86_64 CUDA host; pass HF_TOKEN for the gated repos):
#   docker build -f cloud/Dockerfile \
#     --secret id=hf,env=HF_TOKEN \
#     -t <registry>/sesgo-vllm:latest .
#
# PUSH:
#   docker push <registry>/sesgo-vllm:latest
#
# USE FROM VAST: set IMAGE to the pushed tag for the fleet launch, e.g.
#   IMAGE=<registry>/sesgo-vllm:latest bash cloud/fleet_launch.sh
# vast pulls the image on instance create; because deps + weights are baked,
# at_setup.sh's `uv sync` is a near-instant no-op and the run starts immediately.
#
# NOTE: vLLM is CUDA-only. This image is x86_64/CUDA; it does NOT build on Apple
# Silicon and is NOT built locally here (per the hardware constraint).

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/root/.cache/huggingface

# System Python + git (for the `latents` git dependency) + curl for uv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible dependency resolution (matches the repo workflow).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /root/apart

# Dependency metadata first so the heavy install layer caches across code edits.
COPY pyproject.toml uv.lock ./

# Install the repo deps PLUS the CUDA-only `cloud` extra (vLLM). The extra's
# environment marker (linux/x86_64) is satisfied here, so vLLM is installed.
RUN uv sync --extra cloud --frozen || uv sync --extra cloud

# Bake the gated + public model snapshots into the image so boxes start warm.
# HF_TOKEN is provided as a build secret (never written to a layer).
COPY cloud/ ./cloud/
COPY src/ ./src/
RUN --mount=type=secret,id=hf \
    HF_TOKEN="$(cat /run/secrets/hf 2>/dev/null || true)" \
    uv run python cloud/prefetch_model_weights.py

# The rest of the repo (synced fresh per run by sync_up.sh, but present for any
# direct `docker run`). Code is cheap to overlay; weights/deps above are the bulk.
COPY . .

CMD ["bash"]
