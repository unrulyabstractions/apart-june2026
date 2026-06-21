"""Per-model GPU sizing + shard plan for the parallel cloud fleet.

A SINGLE source of truth, consumed two ways:
  * as a CLI (``python cloud/fleet_sizing.py plan``) that emits one TSV row per
    (model, shard) box, which the fleet shell scripts loop over to launch /
    sync / destroy boxes CONCURRENTLY;
  * as importable ``FleetMember`` / ``fleet_plan`` for tests and tooling.

Sizing rule (right-size the GPU to the model so every box runs cheaply):
  <= 8B params  -> 1× RTX 4090 (24 GB) — Llama-3.2-1B, gemma-2-2b, Mistral-7B, Qwen-0.6B
  <= 16B        -> 1× A100 40GB
  <= 40B        -> 1× H100 80GB (a 32B in bf16 fits one 80 GB card)
  else          -> 2× H100 SXM 80GB (≈160 GB) — Llama-3.1-70B (bf16 ≈ 140 GB) needs
                   MULTI-GPU; the HF backend shards it with device_map="auto"
                   (HF_DEVICE_MAP=auto), i.e. tensor/pipeline parallelism on one box.
``shards`` splits the 2040-item grid into K disjoint contiguous slices so even a
single big model parallelizes across K boxes; default 1 (small models).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# NOTE: plain dataclasses (NOT BaseSchema) on purpose — this planning helper runs
# under the SYSTEM python in fleet_launch.sh, and importing anything from src/
# triggers src/__init__.py's auto-export, which pulls in torch (absent there).


@dataclass
class FleetMember:
    """One model in the fleet: its repo id, GPU class, price ceiling, shard count."""

    model: str  # HF repo id, e.g. meta-llama/Llama-3.2-1B-Instruct
    params_b: float  # approximate parameter count in billions (drives GPU choice)
    gpu_name: str  # vast GPU_NAME, e.g. RTX_4090
    max_price: float  # $/hr ceiling for the offer search
    disk_gb: int  # disk to provision (must fit the HF weights download + xet cache)
    num_gpus: int = 1  # GPUs per box (>1 -> multi-GPU box, HF device_map="auto")
    shards: int = 1  # disjoint contiguous grid slices (boxes) for THIS model

    @property
    def bare_name(self) -> str:
        """Output-scoping name: last path segment of the repo id (e.g. Qwen3-0.6B)."""
        return self.model.split("/")[-1]


def _gpu_for(params_b: float) -> tuple[str, float, int, int]:
    """Right-size GPU class + price ceiling + DISK GB + #GPUs to a param count.

    Disk must hold the HF weights download (bf16 ≈ 2 GB/B params) PLUS the
    huggingface xet cache's temp copy, so it needs to be comfortably larger than
    the weights. A fixed 60 GB box ran the 24-32B downloads out of space mid-pull
    ("OSError: No space left on device"), so the big tier gets 200 GB.

    A 70B in bf16 (≈ 140 GB weights) does NOT fit one 80 GB card: the huge tier
    provisions 2× H100 SXM (≈ 160 GB) and a 320 GB disk (weights + xet temp copy),
    and the HF backend shards it with device_map="auto" (HF_DEVICE_MAP=auto).
    """
    # Price ceilings are GENEROUS on purpose: fleet_launch.sh selects offers
    # reliability-first (not cheapest-first), so the ceiling only needs to be high
    # enough that datacenter-grade, high-reliability hosts clear it. Uptime >> a few
    # cents/hr here. Returns (gpu_name, max_price, disk_gb, num_gpus).
    if params_b <= 8:
        return "RTX_4090", 1.20, 60, 1
    if params_b <= 16:
        return "A100_PCIE", 3.00, 120, 1
    # H100_PCIE is frequently sold out on Vast; H100_SXM is always 80 GB (fits a
    # 32B in bf16). Reliability-first selection needs price headroom to land a
    # solid host, so the big tier gets a generous ceiling.
    if params_b <= 40:
        return "H100_SXM", 7.00, 200, 1
    # HUGE tier (>40B): a 70B in bf16 needs ≈ 140 GB of VRAM, so split it across
    # 2× H100 SXM 80GB (≈ 160 GB). dph is PER-GPU on Vast, so a 2-GPU box at this
    # ceiling clears ~$14/hr of datacenter H100s — fine (money is not the
    # constraint here, landing a reliable multi-GPU host is). Disk 320 GB holds the
    # 140 GB weights plus the xet cache's temp copy with headroom.
    return "H100_SXM", 7.00, 320, 2


# The SESGO baseline size-sweep fleet: size ladders per family so we can read the
# size trend. _gpu_for right-sizes each (<=8B->4090, <=16B->A100, else H100). All
# single-shard (a 32B fits one H100 80GB); bump shards for a slow model if needed.
_DEFAULT_MODELS: list[tuple[str, float, int]] = [
    # Qwen3 dense ladder
    ("Qwen/Qwen3-0.6B", 0.6, 1),
    ("Qwen/Qwen3-1.7B", 1.7, 1),
    ("Qwen/Qwen3-4B", 4.0, 1),
    ("Qwen/Qwen3-8B", 8.0, 1),
    ("Qwen/Qwen3-14B", 14.0, 1),
    ("Qwen/Qwen3-32B", 32.0, 1),
    # Llama 3.x — the 70B is the big Llama: bf16 ≈ 140 GB, so the huge tier puts
    # it on 2× H100 SXM and the HF backend shards it (HF_DEVICE_MAP=auto).
    ("meta-llama/Llama-3.2-1B-Instruct", 1.2, 1),
    ("meta-llama/Llama-3.2-3B-Instruct", 3.2, 1),
    ("meta-llama/Llama-3.1-8B-Instruct", 8.0, 1),
    ("meta-llama/Llama-3.1-70B-Instruct", 70.0, 1),
    # Gemma 2
    ("google/gemma-2-2b-it", 2.6, 1),
    ("google/gemma-2-9b-it", 9.2, 1),
    ("google/gemma-2-27b-it", 27.0, 1),
    # Mistral
    ("mistralai/Mistral-7B-Instruct-v0.3", 7.2, 1),
    ("mistralai/Mistral-Small-24B-Instruct-2501", 24.0, 1),
]


def _parse_model_filter(spec: str | None) -> set[str] | None:
    """A comma list of repo ids OR bare names to keep (None == keep all)."""
    if not spec:
        return None
    return {s.strip() for s in spec.split(",") if s.strip()}


def _keep_member(model: str, keep: set[str] | None) -> bool:
    """Match a model against the filter by full repo id OR bare last segment."""
    if keep is None:
        return True
    return model in keep or model.split("/")[-1] in keep


def default_fleet() -> list[FleetMember]:
    """Build the fleet, right-sizing each model's GPU + price ceiling.

    Two env overrides let a SINGLE-study run target one model without editing the
    default ladder (so a selection-only Qwen run never disturbs the size sweep):
      * FLEET_MODELS — comma list of repo ids / bare names to keep (default: all).
      * FLEET_SHARDS — override the shard count for every kept model (e.g. split
        the heavier thinking grid across K disjoint boxes). 0/unset == per-model.
    """
    keep = _parse_model_filter(os.environ.get("FLEET_MODELS"))
    shard_override = int(os.environ.get("FLEET_SHARDS", "0") or "0")
    members: list[FleetMember] = []
    for model, params_b, shards in _DEFAULT_MODELS:
        if not _keep_member(model, keep):
            continue
        gpu, price, disk, ngpu = _gpu_for(params_b)
        members.append(
            FleetMember(
                model=model,
                params_b=params_b,
                gpu_name=gpu,
                max_price=price,
                disk_gb=disk,
                num_gpus=ngpu,
                shards=shard_override if shard_override > 0 else max(1, shards),
            )
        )
    return members


@dataclass
class FleetBox:
    """One concrete BOX = one (model, shard) unit of work the fleet launches."""

    model: str
    bare_name: str
    gpu_name: str
    max_price: float
    disk_gb: int  # disk to provision for this box
    shard_index: int  # 0-based shard this box owns
    shard_count: int  # total shards for this model
    num_gpus: int = 1  # GPUs to request for this box (>1 -> multi-GPU)


def fleet_plan(members: list[FleetMember]) -> list[FleetBox]:
    """Expand each model into one FleetBox PER shard (the launch unit)."""
    boxes: list[FleetBox] = []
    for m in members:
        for k in range(m.shards):
            boxes.append(
                FleetBox(
                    model=m.model,
                    bare_name=m.bare_name,
                    gpu_name=m.gpu_name,
                    max_price=m.max_price,
                    disk_gb=m.disk_gb,
                    shard_index=k,
                    shard_count=m.shards,
                    num_gpus=m.num_gpus,
                )
            )
    return boxes


def _print_tsv(boxes: list[FleetBox]) -> None:
    """Emit one TAB-separated row per box for the fleet shell loops to read.

    num_gpus is the LAST column so existing fleet shell readers (which read the
    first seven fields) keep working unchanged; the launcher reads the 8th field.
    """
    for b in boxes:
        print(
            f"{b.model}\t{b.bare_name}\t{b.gpu_name}\t{b.max_price}\t"
            f"{b.disk_gb}\t{b.shard_index}\t{b.shard_count}\t{b.num_gpus}"
        )


def main() -> None:
    """``plan`` emits the launch table; default also prints it."""
    _print_tsv(fleet_plan(default_fleet()))


if __name__ == "__main__":
    main()
