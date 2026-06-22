"""Shard-aware prompt slicing + output paths shared by the collect scripts.

A box may own only shard k of K of the grid (cross-box parallelism). These two
helpers give every collect script the SAME behaviour:

  * slice the prompt dataset to this box's contiguous, disjoint shard, and
  * route output to a per-shard subdir so concurrent shard boxes for the SAME
    model write to DISJOINT paths and can never clobber each other on sync-back.

K == 1 (the default for small models) is a no-op: the full grid, the plain
``out/sesgo/<study>/<bare-model>/`` path — exactly the pre-sharding layout.
"""

from __future__ import annotations

from pathlib import Path

from src.common.shard_slicing import take_shard
from src.datasets.prompt import SesgoPromptDataset, SesgoPromptSample


def apply_shard(
    dataset: SesgoPromptDataset, shard_index: int, shard_count: int
) -> SesgoPromptDataset:
    """Keep only this box's contiguous shard of the prompts (1 shard == all)."""
    if shard_count <= 1:
        return dataset
    kept: list[SesgoPromptSample] = take_shard(
        dataset.samples, shard_index, shard_count
    )
    return SesgoPromptDataset(
        dataset_id=dataset.dataset_id,
        config=dataset.config,
        scaffold_ids=dataset.scaffold_ids,
        samples=kept,
    )


def shard_out_dir(
    base_out: Path, study: str, bare_model: str, shard_index: int, shard_count: int
) -> Path:
    """Per-(model, shard) output dir; disjoint across concurrent boxes.

    ``out/sesgo/<study>/<bare-model>/`` for a single shard, with a
    ``shard_<k>_of_<K>/`` leaf appended when K > 1 so shard boxes never collide.
    """
    root = base_out / "sesgo" / study / bare_model
    if shard_count <= 1:
        return root
    return root / f"shard_{shard_index}_of_{shard_count}"
