"""Merge sharded readout outputs back into one response_samples.json per (model, mode).

When a model's prompts are split across K boxes (--shard-index i --shard-count K), each box
writes out/<study>/<bare>-<mode>/shard_i_of_K/response_samples.json over a contiguous slice
of the dataset. This concatenates those shards (deduped by prompt_id) into the parent
out/<study>/<bare>-<mode>/response_samples.json so the figures see a single full slice.

Usage:
  uv run python -m experiment.stability.merge_shards --root out            # merge everything
  uv run python -m experiment.stability.merge_shards --root sync/<tag>     # one box's pull
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment.stability.greedy_readout_schema import GreedyReadoutDataset


def _merge_one(model_dir: Path) -> int | None:
    """Merge model_dir/shard_*_of_*/response_samples.json -> model_dir/response_samples.json.
    Returns the merged sample count, or None if there were no shard dirs."""
    shards = sorted(model_dir.glob("shard_*_of_*/response_samples.json"))
    if not shards:
        return None
    base = GreedyReadoutDataset.from_dict(json.load(shards[0].open()))
    seen, merged = set(), []
    for f in shards:
        ds = GreedyReadoutDataset.from_dict(json.load(f.open()))
        for s in ds.samples:
            if s.prompt_id not in seen:
                seen.add(s.prompt_id)
                merged.append(s)
    base.samples = merged
    out = model_dir / "response_samples.json"
    json.dump(base.to_dict(), out.open("w"), ensure_ascii=False, indent=2)
    return len(merged)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="out", help="root holding <study>/<model>/ trees")
    args = ap.parse_args()
    root = Path(args.root)
    n = 0
    # <root>/[sesgo?]/<study>/<model-mode>/shard_*  — just find any dir containing shard_* dirs
    for model_dir in sorted({p.parent.parent for p in root.rglob("shard_*_of_*/response_samples.json")}):
        cnt = _merge_one(model_dir)
        if cnt is not None:
            print(f"[merge] {model_dir}  ->  {cnt} samples (from "
                  f"{len(list(model_dir.glob('shard_*_of_*')))} shards)")
            n += 1
    print(f"[merge] merged {n} model slices under {root}")


if __name__ == "__main__":
    main()
