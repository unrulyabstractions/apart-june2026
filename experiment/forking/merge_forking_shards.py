"""PHASE 3 of the SHARDED forking-paths pipeline: merge shards -> one trajectory.

Run-by-path driver, run LOCALLY. Loads every ``forking_shard_*_of_*.json`` in
``--in-dir``, validates they share the same model / item / num_shards and cover
positions 0..P-1 exactly once (any gap or duplicate is logged LOUDLY — never
silently dropped), concatenates the positions by REAL base-path index, and rebuilds
a full ``ForkingTrajectory``. The output is a drop-in for ``plot_forking_dynamics``.

Output: <out_dir>/forking_trajectory.json (a ForkingTrajectory).

Usage:
  uv run python sesgo/forking/merge_forking_shards.py \
      --in-dir sync/forkshards --out-dir out/sesgo/forking/Qwen3-14B
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import load_json, save_json_atomic  # noqa: E402
from src.common.logging import log, log_header  # noqa: E402
from src.dynamics.forking_paths import ForkingShard, merge_forking_shards  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the shard merge."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, required=True, help="dir holding forking_shard_*_of_*.json")
    p.add_argument("--out-dir", type=Path, required=True, help="dir to write forking_trajectory.json")
    return p.parse_args()


def _load_shards(in_dir: Path) -> list[ForkingShard]:
    """Load every forking_shard_*_of_*.json under in_dir (recursively), sorted by index."""
    paths = sorted(in_dir.rglob("forking_shard_*_of_*.json"))
    shards = [ForkingShard.from_dict(load_json(p)) for p in paths]
    shards.sort(key=lambda s: s.shard_index)
    return shards


def main() -> None:
    """Load shards, merge by real position index, and persist the full trajectory."""
    args = parse_args()
    log_header("MERGE FORKING SHARDS")

    shards = _load_shards(args.in_dir)
    if not shards:
        log(f"[merge] FATAL: no forking_shard_*_of_*.json found under {args.in_dir}")
        sys.exit(1)
    log(f"[merge] loaded {len(shards)} shards: indices={[s.shard_index for s in shards]} "
        f"num_shards={shards[0].num_shards} model={shards[0].model}")

    trajectory, warnings = merge_forking_shards(shards)
    for w in warnings:
        log(f"[merge] WARNING: {w}")
    if not warnings:
        log("[merge] coverage OK: positions 0..P-1 covered exactly once, all shards agree")

    log(f"[merge] merged {len(trajectory.positions)} positions; "
        f"o_0={[round(x,2) for x in trajectory.prior_histogram]} "
        f"o_T={[round(x,2) for x in trajectory.final_histogram]}")

    out_path = args.out_dir / "forking_trajectory.json"
    save_json_atomic(trajectory.to_dict(), out_path)
    log(f"[merge] wrote {out_path}")


if __name__ == "__main__":
    main()
