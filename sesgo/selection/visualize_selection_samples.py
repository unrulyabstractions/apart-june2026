"""Plot the SELECTION study: how often a model abstains on ambiguous SESGO items.

Run-by-path driver. Loads a response_samples.json produced by
collect_selection_samples.py (a SesgoDataset of ambiguous items where the only correct
answer is "unknown", read out at several answering styles) and renders three plain
figures that answer ONE question: how often does the model correctly refuse to guess a
social group? Higher abstention is better — guessing on an ambiguous question is the
biased move. The run is small (~35 items), so every bar shows its sample size and a
Wilson 95% CI; no figure implies more power than the data give.

  abstention_by_scaffold.png          - headline: abstention by answering style
  abstention_by_scaffold_category.png - abstention split by bias category
  accuracy_by_scaffold_ambig.png      - per-readout panels (the full small-n picture)

Usage:
  .venv/bin/python sesgo/selection/visualize_selection_samples.py \
      out/sesgo/selection/Qwen3-0.6B/response_samples.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import seaborn as sns  # noqa: E402

# Bootstrap repo root + this dir onto sys.path so src.* / sesgo.* and the sibling
# helper modules all resolve regardless of cwd (parents[2] is the repo root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.common.logging import log, log_header  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402
from selection_figure_helpers import (  # noqa: E402
    figure_abstention_by_category,
    figure_abstention_by_readout,
    figure_abstention_panels,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for selection visualization."""
    parser = argparse.ArgumentParser(
        description="Plot abstention on ambiguous SESGO items for a SesgoDataset",
    )
    parser.add_argument(
        "samples", type=Path,
        help="Path to response_samples.json (a SesgoDataset) from collect_selection_samples.py",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"),
        help="Base output dir; plots land at <out-dir>/sesgo/selection/<MODEL>/plots/",
    )
    return parser.parse_args()


def main() -> None:
    """Load the SesgoDataset, render the three abstention figures, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO SELECTION")

    dataset = SesgoDataset.from_json(args.samples)
    by_cond = Counter(s.context_condition for s in dataset.samples)
    log(f"[viz] {len(dataset.samples)} samples · conditions {dict(by_cond)} · "
        f"model={dataset.model_name}")

    plots_dir = args.out_dir / "sesgo" / "selection" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    model = dataset.model_name
    written = [
        figure_abstention_by_readout(
            dataset, model, plots_dir / "abstention_by_scaffold.png"),
        figure_abstention_by_category(
            dataset, model, plots_dir / "abstention_by_scaffold_category.png"),
        figure_abstention_panels(
            dataset, model, plots_dir / "accuracy_by_scaffold_ambig.png"),
    ]

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
