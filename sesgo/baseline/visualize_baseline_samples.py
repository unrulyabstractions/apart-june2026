"""Compute and plot the SINGLE-MODEL SESGO baseline accuracy + role-prob figures.

Run-by-path driver for the baseline study. Loads a ``response_samples.json``
produced by ``collect_baseline_samples.py`` (a SesgoDataset of Spanish, original
items in BOTH context conditions) and answers, per ``bias_category``, the three
accuracy questions that define the baseline:

  1. AMBIGUOUS abstention accuracy (gold == UNKNOWN) — does the model abstain
     when there is no evidence?
  2. DISAMBIGUATED accuracy on items whose gold is the TARGET group.
  3. DISAMBIGUATED accuracy on items whose gold is the OTHER group.
The TARGET-vs-OTHER gap is a bias signal and is surfaced side by side.

All three are reported across THREE readouts — 3-option non-thinking, 2-option
forced choice (no UNKNOWN), and greedy-thinking — as STACKED subfigures in one
figure file, with Wilson 95% CIs and an annotated n on every bar. A companion
figure keeps the mean non-thinking role-prob [target/other/unknown] per category
with SEM whiskers. Plots land at out/sesgo/baseline/<MODEL>/plots/.

Usage:
  uv run python sesgo/baseline/visualize_baseline_samples.py \
      out/sesgo/baseline/Qwen3-0.6B/response_samples.json
  uv run python sesgo/baseline/visualize_baseline_samples.py SAMPLES.json --out-dir out
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
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample  # noqa: E402

from sesgo.baseline.baseline_accuracy_slices import (  # noqa: E402
    AccuracyCell,
    categories_of,
    cells_for,
)
from sesgo.baseline.baseline_role_prob_plot import plot_role_prob  # noqa: E402
from sesgo.baseline.baseline_sample_plots import plot_accuracy  # noqa: E402
from sesgo.baseline.cross_model_aggregation import CONDITIONS  # noqa: E402
from sesgo.common import READOUT_LABEL  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for single-model baseline visualization."""
    parser = argparse.ArgumentParser(
        description="Plot single-model SESGO baseline accuracy + role-prob figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "samples",
        type=Path,
        help="Path to response_samples.json (a SesgoDataset) from collect_baseline_samples.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; plots land at <out-dir>/sesgo/baseline/<MODEL>/plots/",
    )
    return parser.parse_args()


def _scored(dataset: SesgoDataset) -> list[SesgoSample]:
    """Samples carrying a 3-option non-thinking readout (the always-present base)."""
    return [s for s in dataset.samples if s.non_thinking is not None]


def _log_stats(cells: list[AccuracyCell], cats: list[str]) -> None:
    """Emit the per-condition x per-slice accuracy table to the log."""
    log_section("BASELINE ACCURACY BY CATEGORY (slice = ambig / disambig-target / -other)")
    lut = {(c.condition, c.category, c.slice_label): c for c in cells}
    for cond, _ in CONDITIONS:
        log(f"  {READOUT_LABEL.get(cond, cond)}:")
        for cat in cats:
            parts = []
            for slc in ("ambig", "disambig-target", "disambig-other"):
                cell = lut.get((cond, cat, slc))
                if cell is None or cell.total == 0:
                    parts.append(f"{slc}=n/a")
                else:
                    parts.append(f"{slc}={cell.accuracy:.0%}(n={cell.total})")
            log(f"    {cat:<12}: " + "  ".join(parts))


def main() -> None:
    """Load the SesgoDataset, compute accuracy slices, plot, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO BASELINE (single model)")

    dataset = SesgoDataset.from_json(args.samples)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")
    n_items = len({s.question_id for s in dataset.samples})
    log(f"[viz] {n_items} distinct question_id(s)")

    scored = _scored(dataset)
    cats = categories_of(scored)
    cells = cells_for(scored)
    _log_stats(cells, cats)

    plots_dir = args.out_dir / "sesgo" / "baseline" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    written = [
        plot_accuracy(cells, cats, plots_dir / "accuracy.png"),
        plot_role_prob(scored, cats, plots_dir / "role_prob.png"),
    ]
    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
