"""Visualize the FULL-DATA SESGO baseline sliced by the NEW axes.

Run-by-path driver for the full_data study. Loads a ``response_samples.json``
produced by ``collect_baseline_samples.py --study full_data`` (a SesgoDataset
spanning BOTH languages, BOTH origins, and the no-scaffold + three-scaffold grid) and
answers one question across the three axes the full-data study adds: how does the
ambiguous ABSTENTION rate move with language (es vs en), origin (original vs
BBQ-adapted), and scaffold (none vs each debiasing scaffold)?

The per-category baseline figure already covers bias_category; this driver is the
complementary view that surfaces the language/origin/scaffold structure the widened
grid unlocks. One grid figure (rows = the three readouts, columns = the three axes)
lands at out/sesgo/full_data/<MODEL>/plots/abstention_by_axis.png.

Usage:
  uv run python sesgo/baseline/visualize_full_data_samples.py \
      out/sesgo/full_data/Qwen3-0.6B/response_samples.json
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
from src.datasets.sesgo import SesgoLabel  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample  # noqa: E402

from sesgo.baseline.full_data_axis_plots import plot_full_axes  # noqa: E402
from sesgo.baseline.full_data_axis_slices import (  # noqa: E402
    AXES,
    abstention_cells,
    axis_values,
)
from sesgo.baseline.full_data_breakdown_plots import plot_abstention_breakdown  # noqa: E402
from sesgo.baseline.cross_model_aggregation import COND_TITLES  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for full-data baseline axis visualization."""
    parser = argparse.ArgumentParser(
        description="Plot full-data SESGO baseline abstention by language/origin/scaffold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "samples",
        type=Path,
        help="Path to response_samples.json (a SesgoDataset) from --study full_data",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; plots land at <out-dir>/sesgo/full_data/<MODEL>/plots/",
    )
    return parser.parse_args()


def _scored(dataset: SesgoDataset) -> list[SesgoSample]:
    """Samples carrying a 3-option non-thinking readout (the always-present base)."""
    return [s for s in dataset.samples if s.non_thinking is not None]


def _log_axis_stats(cells_by_axis: dict, axis_vals: dict) -> None:
    """Emit per-axis x per-readout abstention to the log (the report numbers)."""
    log_section("FULL-DATA ABSTENTION BY AXIS (accuracy = fraction predicted UNKNOWN)")
    for axis in AXES:
        log(f"  axis = {axis}:")
        lut = {(c.condition, c.slice_label): c for c in cells_by_axis[axis]}
        for cond, title in COND_TITLES.items():
            parts = []
            for value in axis_vals[axis]:
                cell = lut.get((cond, value))
                if cell is None or cell.total == 0:
                    parts.append(f"{value}=n/a")
                else:
                    parts.append(f"{value}={cell.accuracy:.0%}(n={cell.total})")
            log(f"    {title}: " + "  ".join(parts))


def main() -> None:
    """Load the SesgoDataset, slice abstention by the new axes, plot, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO FULL-DATA BASELINE (language / origin / scaffold)")

    dataset = SesgoDataset.from_json(args.samples)
    scored = _scored(dataset)
    ambig = [s for s in scored if s.gold_label is SesgoLabel.UNKNOWN]
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")
    log(f"[viz] {len(scored)} scored, {len(ambig)} ambiguous (abstention is scored there)")

    cells_by_axis = {axis: abstention_cells(scored, axis) for axis in AXES}
    axis_vals = {axis: axis_values(ambig, axis) for axis in AXES}
    _log_axis_stats(cells_by_axis, axis_vals)

    plots_dir = args.out_dir / "sesgo" / "full_data" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    out_path = plot_full_axes(
        cells_by_axis, axis_vals, dataset.model_name, len(ambig),
        plots_dir / "abstention_by_axis.png",
    )
    log(f"[viz] wrote {out_path}")

    # The hero breakdown: ambiguous abstention by thinking x scaffold x wording,
    # faceted per bias category -- the plain-language view of whether scaffolds work.
    breakdown_path = plot_abstention_breakdown(
        ambig, dataset.model_name, plots_dir / "abstention_breakdown.png",
    )
    log(f"[viz] wrote {breakdown_path}")


if __name__ == "__main__":
    main()
