"""Combine per-shard baseline_full slices of ONE model into a single dataset.

The cloud fleet runs the full-data baseline grid for a model split across K boxes;
box k writes ONLY its disjoint contiguous shard under
``sync/box-<tag>/sesgo/baseline_full/<model>/shard_<k>_of_<K>/response_samples.json``.
``merge_sync.sh`` uses ``rsync --ignore-existing``, which is correct for disjoint
MODELS but WRONG for SHARDS of the SAME model (it would keep only the FIRST shard
and silently drop the rest). So shards need a dedicated, identity-aware combine —
this is the baseline_full twin of combine_selection_shards.py.

Why this is clobber-safe by construction: a sample's ``sample_idx`` is the GLOBAL
grid index (0..N-1) and ``apply_shard`` PRESERVES it when slicing, so de-duping by
``sample_identity`` lets a shard re-run never double-count or overwrite (first
occurrence wins). We concatenate every shard's samples and write a single combined
``out/sesgo/baseline_full/<model>/response_samples.json``.

Verification (hard-fails): all four scaffold conditions present (none + the 3
representative scaffolds), and BOTH languages + BOTH origins survived the merge.

Usage:
  uv run python sesgo/baseline/combine_baseline_full_shards.py Qwen3-0.6B
  uv run python sesgo/baseline/combine_baseline_full_shards.py Qwen3-0.6B \
      --sync-dir sync --out-dir out
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/baseline/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample  # noqa: E402
from src.datasets.sesgo_eval.checkpoint_resume_helpers import (  # noqa: E402
    sample_identity,
)

_STUDY = "baseline_full"


def find_shard_files(sync_dir: Path, model: str) -> list[Path]:
    """Every ``response_samples.json`` for ``model`` across all box-* quarantines.

    Sorted so the combine is deterministic. A single-box run (no shard leaf) is
    also picked up via its plain model dir, so this one entry point handles both
    sharded and single-box runs.
    """
    roots = sorted(sync_dir.glob(f"box-*/sesgo/{_STUDY}/{model}"))
    files: list[Path] = []
    for root in roots:
        leaves = sorted(root.glob("shard_*_of_*/response_samples.json"))
        files.extend(leaves if leaves else [root / "response_samples.json"])
    return sorted({f for f in files if f.exists()})


def merge_shard_datasets(files: list[Path]) -> tuple[SesgoDataset, int]:
    """Concatenate every shard's SesgoSamples (de-duped); return it + raw total.

    Samples are keyed by ``sample_identity`` so a shard re-run never double-counts;
    the FIRST occurrence wins. Header (prompt_dataset_id / model / config) comes
    from the first shard. Returns the combined dataset and the raw pre-dedup total.
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
    """Log the dedup and assert all axes (scaffold/language/origin) survived."""
    by_scaffold = Counter(s.scaffold_id or "(none)" for s in dataset.samples)
    by_language = Counter(s.language for s in dataset.samples)
    by_origin = Counter("BBQ-adapted" if s.bbq else "original" for s in dataset.samples)
    log(f"[combine] combined samples : {len(dataset.samples)} (raw shard sum {raw_total})")
    log(f"[combine] by scaffold      : {dict(by_scaffold)}")
    log(f"[combine] by language      : {dict(by_language)}")
    log(f"[combine] by origin        : {dict(by_origin)}")
    if len(by_scaffold) < 4:
        raise SystemExit(
            f"[combine] only {len(by_scaffold)} scaffold condition(s) present; the "
            "full-data grid needs none + 3 scaffolds — refusing to write a partial file."
        )
    if len(by_language) < 2 or len(by_origin) < 2:
        raise SystemExit(
            "[combine] the full grid must span BOTH languages and BOTH origins; "
            f"got languages={dict(by_language)} origins={dict(by_origin)} — refusing."
        )


def parse_args() -> argparse.Namespace:
    """Parse the model name plus optional sync/out roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Bare model name, e.g. Qwen3-0.6B")
    parser.add_argument("--sync-dir", type=Path, default=Path("sync"))
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    return parser.parse_args()


def main() -> None:
    """Combine all shard slices of one model into out/sesgo/baseline_full/<model>/."""
    args = parse_args()
    log_header(f"COMBINE BASELINE_FULL SHARDS ({args.model})")

    files = find_shard_files(args.sync_dir, args.model)
    if not files:
        raise SystemExit(f"[combine] no shard slices under {args.sync_dir}/box-*/")
    log_section(f"{len(files)} shard slices")

    combined, raw_total = merge_shard_datasets(files)
    verify_combined(combined, raw_total)

    out_path = args.out_dir / "sesgo" / _STUDY / args.model / "response_samples.json"
    combined.save_as_json(out_path)
    log(f"[combine] wrote {out_path}")


if __name__ == "__main__":
    main()
