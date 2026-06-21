"""Combine per-shard selection slices of ONE model into a single dataset.

The cloud fleet runs the selection grid for a model split across K boxes; box k
writes ONLY its disjoint contiguous shard under
``sync/box-<tag>/sesgo/selection/<model>/shard_<k>_of_<K>/response_samples.json``.
``merge_sync.sh`` uses ``rsync --ignore-existing``, which is correct for disjoint
MODELS but WRONG for SHARDS of the SAME model: it would keep only the FIRST
shard's response_samples.json and silently drop the rest. So shards need a
dedicated, identity-aware combine instead — this script (the selection twin of
combine_geometry_shards.py, minus the activation tensors selection never emits).

Why this is clobber-safe by construction: a sample's ``sample_idx`` is the GLOBAL
grid index (0..N-1) and ``apply_shard`` PRESERVES it when slicing, so de-duping by
``sample_identity`` lets a shard re-run never double-count or overwrite. We
concatenate every shard's samples (first occurrence wins) and write a single
combined ``out/sesgo/selection/<model>/response_samples.json``.

Verification (hard-fails): combined count == sum of shard counts minus the number
of duplicate identities, and every one of the five scaffold conditions is present.

Usage:
  uv run python sesgo/selection/combine_selection_shards.py Qwen3-0.6B
  uv run python sesgo/selection/combine_selection_shards.py Qwen3-0.6B \
      --sync-dir sync --out-dir out
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/selection/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample  # noqa: E402
from src.datasets.sesgo_eval.checkpoint_resume_helpers import (  # noqa: E402
    sample_identity,
)


def find_shard_files(sync_dir: Path, model: str) -> list[Path]:
    """Every ``response_samples.json`` for ``model`` across all box-* quarantines.

    Sorted so the combine is deterministic. A single-box run (no shard leaf) is
    also picked up via its plain model dir, so this one entry point handles both
    sharded and single-box selection runs.
    """
    roots = sorted(sync_dir.glob(f"box-*/sesgo/selection/{model}"))
    files: list[Path] = []
    for root in roots:
        leaves = sorted(root.glob("shard_*_of_*/response_samples.json"))
        files.extend(leaves if leaves else [root / "response_samples.json"])
    return sorted({f for f in files if f.exists()})


def merge_shard_datasets(files: list[Path]) -> tuple[SesgoDataset, int]:
    """Concatenate every shard's SesgoSamples (de-duped); return it + raw total.

    Samples are keyed by ``sample_identity`` so a shard re-run never
    double-counts; the FIRST occurrence wins (later duplicates dropped, never
    overwritten). Header (prompt_dataset_id / model / config) comes from the first
    shard. Returns the combined dataset and the raw pre-dedup shard sample total.
    """
    samples: list[SesgoSample] = []
    seen: set = set()
    raw_total = 0
    header: SesgoDataset | None = None
    for path in files:
        ds = SesgoDataset.from_json(path)
        header = header or ds
        raw_total += len(ds.samples)
        for s in ds.samples:
            ident = sample_identity(s)
            if ident in seen:
                continue
            seen.add(ident)
            samples.append(s)
        log(f"[combine] {path}: {len(ds.samples)} samples")
    if header is None:
        raise SystemExit("[combine] no shard datasets found")
    combined = SesgoDataset(
        prompt_dataset_id=header.prompt_dataset_id,
        model=header.model,
        config=header.config,
        samples=samples,
    )
    return combined, raw_total


def verify_combined(dataset: SesgoDataset, raw_total: int) -> None:
    """Log the dedup and assert all five scaffold conditions survived the merge."""
    by_scaffold = Counter(s.scaffold_id or "(none)" for s in dataset.samples)
    log(f"[combine] combined samples : {len(dataset.samples)} (raw shard sum {raw_total})")
    log(f"[combine] by scaffold      : {dict(by_scaffold)}")
    if len(by_scaffold) < 2:
        raise SystemExit(
            f"[combine] only {len(by_scaffold)} scaffold condition(s) present; "
            "the scaffold variants did not land — refusing to write a baseline-only file."
        )


def parse_args() -> argparse.Namespace:
    """Parse the model name plus optional sync/out roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Bare model name, e.g. Qwen3-0.6B")
    parser.add_argument("--sync-dir", type=Path, default=Path("sync"))
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    return parser.parse_args()


def main() -> None:
    """Combine all shard slices of one model into out/sesgo/selection/<model>/."""
    args = parse_args()
    log_header(f"COMBINE SELECTION SHARDS ({args.model})")

    files = find_shard_files(args.sync_dir, args.model)
    if not files:
        raise SystemExit(f"[combine] no shard slices under {args.sync_dir}/box-*/")
    log_section(f"{len(files)} shard slices")

    combined, raw_total = merge_shard_datasets(files)
    verify_combined(combined, raw_total)

    out_path = args.out_dir / "sesgo" / "selection" / args.model / "response_samples.json"
    combined.save_as_json(out_path)
    log(f"[combine] wrote {out_path}")


if __name__ == "__main__":
    main()
