"""Per-model GPU sizing + shard plan for the parallel cloud fleet.

A SINGLE source of truth, consumed two ways:
  * as a CLI (``python cloud/fleet_sizing.py plan``) that emits one TSV row per
    (model, shard) box, which the fleet shell scripts loop over to launch /
    sync / destroy boxes CONCURRENTLY;
  * as importable ``FleetMember`` / ``fleet_plan`` for tests and tooling.

Sizing rule (right-size the GPU to the model so every box runs cheaply):
  <= 8B params  -> RTX 4090 (24 GB) — Llama-3.2-1B, gemma-2-2b, Mistral-7B, Qwen-0.6B
  <= 16B        -> A100 40GB
  else          -> H100 80GB
``shards`` splits the 2040-item grid into K disjoint contiguous slices so even a
single big model parallelizes across K boxes; default 1 (small models).
"""

from __future__ import annotations

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
    shards: int = 1  # disjoint contiguous grid slices (boxes) for THIS model

    @property
    def bare_name(self) -> str:
        """Output-scoping name: last path segment of the repo id (e.g. Qwen3-0.6B)."""
        return self.model.split("/")[-1]


def _gpu_for(params_b: float) -> tuple[str, float]:
    """Right-size GPU class + price ceiling to a model's parameter count."""
    if params_b <= 8:
        return "RTX_4090", 0.60
    if params_b <= 16:
        return "A100_PCIE", 1.60
    return "H100_PCIE", 3.20


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
    # Llama 3.x
    ("meta-llama/Llama-3.2-1B-Instruct", 1.2, 1),
    ("meta-llama/Llama-3.2-3B-Instruct", 3.2, 1),
    ("meta-llama/Llama-3.1-8B-Instruct", 8.0, 1),
    # Gemma 2
    ("google/gemma-2-2b-it", 2.6, 1),
    ("google/gemma-2-9b-it", 9.2, 1),
    ("google/gemma-2-27b-it", 27.0, 1),
    # Mistral
    ("mistralai/Mistral-7B-Instruct-v0.3", 7.2, 1),
    ("mistralai/Mistral-Small-24B-Instruct-2501", 24.0, 1),
]


def default_fleet() -> list[FleetMember]:
    """Build the default fleet, right-sizing each model's GPU + price ceiling."""
    members: list[FleetMember] = []
    for model, params_b, shards in _DEFAULT_MODELS:
        gpu, price = _gpu_for(params_b)
        members.append(
            FleetMember(
                model=model,
                params_b=params_b,
                gpu_name=gpu,
                max_price=price,
                shards=max(1, shards),
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
    shard_index: int  # 0-based shard this box owns
    shard_count: int  # total shards for this model


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
                    shard_index=k,
                    shard_count=m.shards,
                )
            )
    return boxes


def _print_tsv(boxes: list[FleetBox]) -> None:
    """Emit one TAB-separated row per box for the fleet shell loops to read."""
    for b in boxes:
        print(
            f"{b.model}\t{b.bare_name}\t{b.gpu_name}\t{b.max_price}\t"
            f"{b.shard_index}\t{b.shard_count}"
        )


def main() -> None:
    """``plan`` emits the launch table; default also prints it."""
    _print_tsv(fleet_plan(default_fleet()))


if __name__ == "__main__":
    main()
