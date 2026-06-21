"""HEADLINE cross-model figure for the SESGO baseline SIZE SWEEP.

Run-by-path driver that scans ``out/sesgo/baseline/*/response_samples.json`` —
each subdir is ONE model named by its bare repo id (Qwen3-0.6B, gemma-2-27b-it,
…) — and plots how SESGO baseline accuracy moves with model SIZE, across the
Qwen / Llama / Gemma / Mistral families. A cloud fleet is filling in up to 14
such dirs; this scales to whatever exists, skipping models it can't size/place.

It answers three size-trend questions in one stacked figure (3-option /
2-option / greedy-thinking):
  - AMBIGUOUS abstention accuracy (gold=unknown) vs size;
  - DISAMBIGUATED accuracy split by gold into TARGET-gold vs OTHER-gold (the gap
    is the bias-by-size signal) vs size.
Every point carries a Wilson 95% CI and an annotated n.

Usage:
  uv run python sesgo/baseline/visualize_baseline_cross_model.py
  uv run python sesgo/baseline/visualize_baseline_cross_model.py --base-dir out/sesgo/baseline
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import seaborn as sns  # noqa: E402

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/baseline/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402

from sesgo.baseline.cross_model_aggregation import (  # noqa: E402
    CrossModelPoint,
    points_for_model,
)
from sesgo.baseline.cross_model_plotting import plot_size_sweep  # noqa: E402

_SAMPLES_FILE = "response_samples.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the cross-model size-sweep visualization."""
    parser = argparse.ArgumentParser(
        description="Headline cross-model SESGO baseline size-sweep figure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir", type=Path, default=Path("out/sesgo/baseline"),
        help="Dir whose immediate subdirs are per-model response_samples.json owners",
    )
    return parser.parse_args()


def _model_dirs(base_dir: Path) -> list[Path]:
    """Per-model sample files: every ``<base>/<model>/response_samples.json``."""
    return sorted(p for p in base_dir.glob(f"*/{_SAMPLES_FILE}") if p.is_file())


def _collect(sample_files: list[Path]) -> tuple[list[CrossModelPoint], list[str]]:
    """Load each model dataset and flatten to plottable points; report placed models."""
    points: list[CrossModelPoint] = []
    placed: list[str] = []
    for path in sample_files:
        dataset = SesgoDataset.from_json(path)
        model_points = points_for_model(dataset.model_name, dataset.samples)
        if model_points:
            points.extend(model_points)
            placed.append(dataset.model_name)
            log(f"  {dataset.model_name:<34} {len(dataset.samples):>5} samples"
                f"  -> {len(model_points)} point(s)")
        else:
            log(f"  {dataset.model_name:<34} skipped (unknown size/family)")
    return points, placed


def _log_points(points: list[CrossModelPoint]) -> None:
    """Emit the per-point accuracy table (model x condition x slice)."""
    log_section("CROSS-MODEL POINTS (accuracy, Wilson 95% CI, n)")
    for p in sorted(points, key=lambda q: (q.condition, q.family, q.params_b, q.slice_label)):
        _, lo, hi = p.wilson
        log(f"  {p.condition:<18} {p.slice_label:<16} {p.model:<24} "
            f"{p.params_b:>5}B  acc={p.accuracy:.2%} "
            f"[{lo:.2%},{hi:.2%}] n={p.total}")


def main() -> None:
    """Scan model dirs, aggregate accuracy points, plot the size sweep, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO BASELINE CROSS-MODEL SIZE SWEEP")

    sample_files = _model_dirs(args.base_dir)
    log(f"[viz] found {len(sample_files)} model dir(s) under {args.base_dir}")
    if not sample_files:
        log("[viz] nothing to plot — no per-model response_samples.json found")
        return

    points, placed = _collect(sample_files)
    if not points:
        log("[viz] no sizable/placeable models — nothing to plot")
        return
    _log_points(points)

    plots_dir = args.base_dir / "cross_model" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    out_path = plots_dir / "baseline_size_sweep.png"
    plot_size_sweep(points, out_path, n_models=len(placed))
    log(f"[viz] wrote size-sweep figure over {len(placed)} model(s):")
    log(f"  {out_path}")


if __name__ == "__main__":
    main()
