"""Combine per-shard divergence slices of ONE model into a single dataset.

The cloud fleet runs the divergence grid for a model split across K boxes; box k
writes ONLY its disjoint contiguous shard under
``sync/box-<tag>/sesgo/divergence/<model>/shard_<k>_of_<K>/response_samples.json``.
``merge_sync.sh`` uses ``rsync --ignore-existing``, which is correct for disjoint
MODELS but WRONG for SHARDS of the SAME model: it would keep only the FIRST
shard's response_samples.json and silently drop the rest. So shards need a
dedicated, identity-aware combine instead — this script (the geometry analogue,
``combine_geometry_shards.py``, does the same for the geometry study, but
divergence is pure sample-level with NO activation .pt files, so this is simpler:
just concatenate the per-shard SesgoSamples, no tensor copying).

Why this is clobber-safe by construction: each shard owns a disjoint contiguous
slice of the SAME subsampled grid (subsample strides the raw json identically on
every box, THEN apply_shard takes this box's slice), so the per-shard samples are
DISJOINT by ``sample_identity``. We concatenate them, de-duping by identity so a
shard re-run can never double-count, and write a single combined
``response_samples.json``.

Verification (hard-fails): the combined sample count equals the de-duped union of
all shard samples, logged against the raw pre-dedup shard total so any unexpected
overlap is visible.

Usage:
  uv run python sesgo/divergence/combine_divergence_shards.py Qwen3-32B
  uv run python sesgo/divergence/combine_divergence_shards.py Qwen3-32B \
      --sync-dir sync --out-dir out
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/divergence/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample  # noqa: E402
from src.datasets.sesgo_eval.checkpoint_resume_helpers import (  # noqa: E402
    sample_identity,
)


def find_shard_dirs(sync_dir: Path, model: str) -> list[Path]:
    """Every ``shard_*_of_*/`` slice for ``model`` across all box-* quarantines.

    Sorted so the combine is deterministic. A model run on a single box (no
    shard leaf) is also picked up via its plain model dir, so this one entry
    point handles both sharded and single-box divergence runs.
    """
    roots = sorted(sync_dir.glob(f"box-*/sesgo/divergence/{model}"))
    shards: list[Path] = []
    for root in roots:
        leaves = sorted(root.glob("shard_*_of_*"))
        shards.extend(leaves if leaves else [root])
    return sorted(set(shards))


def merge_shard_datasets(shards: list[Path]) -> tuple[SesgoDataset, int]:
    """Concatenate every shard's SesgoSamples (de-duped by identity).

    Samples are keyed by ``sample_identity`` so a shard re-run never
    double-counts; the FIRST occurrence wins (later duplicates are dropped, never
    overwritten). Header (prompt_dataset_id / model / config) comes from the first
    shard. Returns the combined dataset and the raw pre-dedup shard sample total.
    """
    samples: list[SesgoSample] = []
    seen: set = set()
    raw_total = 0
    header: SesgoDataset | None = None
    for shard in shards:
        ds = SesgoDataset.from_json(shard / "response_samples.json")
        header = header or ds
        raw_total += len(ds.samples)
        for s in ds.samples:
            ident = sample_identity(s)
            if ident in seen:
                continue
            seen.add(ident)
            samples.append(s)
        log(f"[combine] {shard}: {len(ds.samples)} samples")
    if header is None:
        raise SystemExit("[combine] no shard datasets found")
    combined = SesgoDataset(
        prompt_dataset_id=header.prompt_dataset_id,
        model=header.model,
        config=header.config,
        samples=samples,
    )
    return combined, raw_total


def parse_args() -> argparse.Namespace:
    """Parse the model name plus optional sync/out roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Bare model name, e.g. Qwen3-32B")
    parser.add_argument("--sync-dir", type=Path, default=Path("sync"))
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    return parser.parse_args()


def main() -> None:
    """Combine all shard slices of one model into out/sesgo/divergence/<model>/."""
    args = parse_args()
    log_header(f"COMBINE DIVERGENCE SHARDS ({args.model})")

    shards = find_shard_dirs(args.sync_dir, args.model)
    if not shards:
        raise SystemExit(f"[combine] no shard slices under {args.sync_dir}/box-*/")
    log_section(f"{len(shards)} shard slices")

    combined, raw_total = merge_shard_datasets(shards)
    log(f"[combine] combined samples : {len(combined.samples)} "
        f"(raw shard sum {raw_total})")

    out_path = args.out_dir / "sesgo" / "divergence" / args.model / "response_samples.json"
    combined.save_as_json(out_path)
    log(f"[combine] wrote {out_path}")


if __name__ == "__main__":
    main()
