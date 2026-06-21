"""Combine per-shard geometry slices of ONE model into a single dataset.

The cloud fleet runs the geometry grid for a model split across K boxes; box k
writes ONLY its disjoint contiguous shard under
``sync/box-<tag>/sesgo/geometry/<model>/shard_<k>_of_<K>/`` (its own
response_samples.json + activations/*.pt). ``merge_sync.sh`` uses
``rsync --ignore-existing``, which is correct for disjoint MODELS but WRONG for
SHARDS of the SAME model: it would keep only the FIRST shard's
response_samples.json and silently drop the rest. So shards need a dedicated,
identity-aware combine instead — this script.

Why this is clobber-safe by construction: a prompt's ``sample_idx`` is the GLOBAL
grid index (0..N-1) and ``apply_shard`` PRESERVES it when slicing, so the
activation filenames (``sample_<global_idx>_<pos>_L<layer>.pt``) are already
DISJOINT across shards. We therefore concatenate the per-shard samples (de-duped
by ``sample_identity`` so a re-run can never double-count or overwrite), copy
every shard's .pt into one ``out/sesgo/geometry/<model>/activations/`` tree (no
two shards name the same file), and write a single combined
``response_samples.json`` whose relative activation paths stay valid as-is.

Verification (hard-fails): combined sample count == sum of shard counts (after
de-dup), and every activation path the combined dataset references resolves to an
existing .pt on disk.

Usage:
  uv run python sesgo/geometry/combine_geometry_shards.py Qwen3-0.6B
  uv run python sesgo/geometry/combine_geometry_shards.py Qwen3-32B \
      --sync-dir sync --out-dir out
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/geometry/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import ensure_dir  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import GeometryDataset, GeometrySample  # noqa: E402
from src.datasets.sesgo_eval.checkpoint_resume_helpers import (  # noqa: E402
    sample_identity,
)


def find_shard_dirs(sync_dir: Path, model: str) -> list[Path]:
    """Every ``shard_*_of_*/`` slice for ``model`` across all box-* quarantines.

    Sorted so the combine is deterministic. A model run on a single box (no
    shard leaf) is also picked up via its plain model dir, so this one entry
    point handles both sharded and single-box geometry runs.
    """
    roots = sorted(sync_dir.glob(f"box-*/sesgo/geometry/{model}"))
    shards: list[Path] = []
    for root in roots:
        leaves = sorted(root.glob("shard_*_of_*"))
        shards.extend(leaves if leaves else [root])
    return sorted(set(shards))


def copy_activations(shard: Path, act_out: Path) -> int:
    """Copy a shard's activations/*.pt into the combined tree; return the count.

    Filenames are keyed by the GLOBAL sample_idx, so two DISTINCT shards never
    name the same file and nothing is clobbered. An existing dest therefore only
    ever means THIS exact global-idx file was already copied (a re-run of the same
    shard, or the same combine run twice) — so we skip it, keeping the combine
    idempotent. We never overwrite, so a genuine cross-shard collision (a sharding
    bug producing duplicate global indices) is still surfaced by verify_combined.
    """
    src = shard / "activations"
    if not src.is_dir():
        return 0
    copied = 0
    for pt in src.glob("*.pt"):
        dest = act_out / pt.name
        if dest.exists():
            continue  # same global-idx tensor already present: idempotent re-run
        shutil.copy2(pt, dest)
        copied += 1
    return copied


def merge_shard_datasets(
    shards: list[Path], act_out: Path
) -> tuple[GeometryDataset, int]:
    """Concatenate every shard's GeometrySamples (de-duped) + copy activations.

    Samples are keyed by ``sample_identity`` so a shard re-run never
    double-counts; the FIRST occurrence wins (later duplicates are dropped, never
    overwritten). Header (prompt_dataset_id / model / config) comes from the first
    shard. Returns the combined dataset and the raw pre-dedup shard sample total.
    """
    samples: list[GeometrySample] = []
    seen: set = set()
    raw_total = 0
    header: GeometryDataset | None = None
    for shard in shards:
        ds = GeometryDataset.from_json(shard / "response_samples.json")
        header = header or ds
        raw_total += len(ds.samples)
        copy_activations(shard, act_out)
        for s in ds.samples:
            ident = sample_identity(s)
            if ident in seen:
                continue
            seen.add(ident)
            samples.append(s)
        log(f"[combine] {shard}: {len(ds.samples)} samples")
    if header is None:
        raise SystemExit("[combine] no shard datasets found")
    combined = GeometryDataset(
        prompt_dataset_id=header.prompt_dataset_id,
        model=header.model,
        config=header.config,
        samples=samples,
    )
    return combined, raw_total


def verify_combined(dataset: GeometryDataset, out_root: Path, raw_total: int) -> None:
    """Hard-fail unless every .pt the combined dataset references exists.

    The de-duped count may be < raw_total only by exactly the number of duplicate
    identities; we log both so the operator sees the dedup. Then we assert every
    referenced activation tensor resolves on disk.
    """
    missing = [
        a.path
        for s in dataset.samples
        for a in s.activations
        if not (out_root / a.path).exists()
    ]
    if missing:
        raise SystemExit(
            f"[combine] {len(missing)} referenced .pt missing, e.g. {missing[:3]}"
        )
    log(f"[combine] combined samples : {len(dataset.samples)} (raw shard sum {raw_total})")
    log(f"[combine] all {sum(len(s.activations) for s in dataset.samples)} .pt present")


def parse_args() -> argparse.Namespace:
    """Parse the model name plus optional sync/out roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Bare model name, e.g. Qwen3-0.6B")
    parser.add_argument("--sync-dir", type=Path, default=Path("sync"))
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    return parser.parse_args()


def main() -> None:
    """Combine all shard slices of one model into out/sesgo/geometry/<model>/."""
    args = parse_args()
    log_header(f"COMBINE GEOMETRY SHARDS ({args.model})")

    shards = find_shard_dirs(args.sync_dir, args.model)
    if not shards:
        raise SystemExit(f"[combine] no shard slices under {args.sync_dir}/box-*/")
    log_section(f"{len(shards)} shard slices")

    out_root = args.out_dir / "sesgo" / "geometry" / args.model
    act_out = ensure_dir(out_root / "activations")
    combined, raw_total = merge_shard_datasets(shards, act_out)

    out_path = out_root / "response_samples.json"
    combined.save_as_json(out_path)
    verify_combined(combined, out_root, raw_total)
    log(f"[combine] wrote {out_path}")


if __name__ == "__main__":
    main()
