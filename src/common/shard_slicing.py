"""Split a sequence into K DISJOINT, CONTIGUOUS shards for cross-box parallelism.

A model whose grid is too slow for one box is split across K boxes; box k runs
ONLY shard k. The slices are contiguous and non-overlapping, and every item lands
in exactly one shard, so the union of all shards is the full grid with no dups —
the property that makes the concurrent sync-back safe (disjoint output slices).
"""

from __future__ import annotations


def shard_bounds(total: int, shard_index: int, shard_count: int) -> tuple[int, int]:
    """Half-open [start, end) bounds of shard ``shard_index`` over ``total`` items.

    The remainder is spread one-per-shard across the leading shards, so shard
    sizes differ by at most one and the K slices exactly tile [0, total).
    """
    if shard_count < 1 or not (0 <= shard_index < shard_count):
        raise ValueError(
            f"bad shard {shard_index}/{shard_count} (need 0 <= index < count, count>=1)"
        )
    base, extra = divmod(total, shard_count)
    start = shard_index * base + min(shard_index, extra)
    size = base + (1 if shard_index < extra else 0)
    return start, start + size


def take_shard(items: list, shard_index: int, shard_count: int) -> list:
    """Return the contiguous shard ``shard_index`` of ``items`` (1 shard == all)."""
    if shard_count <= 1:
        return items
    start, end = shard_bounds(len(items), shard_index, shard_count)
    return items[start:end]
