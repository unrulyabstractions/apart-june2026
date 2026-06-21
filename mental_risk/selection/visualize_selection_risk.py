"""Plot the SELECTION RiskDataset into PNGs under .../selection/<MODEL>/plots/.

Run-by-path driver (risk analogue of sesgo/selection/visualize_selection_samples.py).
SESGO ranked debiasing scaffolds by abstention accuracy and starred the best;
risk ranks FRAMINGS by how well their predicted risk tracks the continuous gold
(Pearson r primary, MAE tiebreak) and stars the SELECTED framing. Plots:

  framing_correlation.png  - per-framing Pearson r with gold (bars), SELECTED
                             framing starred. The headline selection plot.
  framing_mean_risk.png    - per-framing mean predicted risk + mean gold line, so
                             over/under-estimation per framing is visible.

Usage:
  uv run python mental_risk/selection/visualize_selection_risk.py
  uv run python mental_risk/selection/visualize_selection_risk.py \
      out/mental_risk/selection/Qwen3-0.6B/samples.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mental_risk.framing_ranking import (  # noqa: E402
    FramingScore,
    best_framing,
    score_framings,
)
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.risk import RiskDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for selection visualization."""
    parser = argparse.ArgumentParser(description="Plot a selection RiskDataset")
    parser.add_argument(
        "samples", type=Path, nargs="?",
        default=Path("out/mental_risk/selection/Qwen3-0.6B/samples.json"),
        help="Path to a selection samples.json (a RiskDataset)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    return parser.parse_args()


def plot_framing_correlation(
    scores: list[FramingScore], selected: str | None, model_name: str, out_path: Path
) -> Path:
    """Bar chart of per-framing Pearson r with gold; SELECTED framing starred."""
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(scores)), 5))
    if scores:
        names = [fs.framing for fs in scores]
        rs = [fs.pearson if fs.pearson is not None else 0.0 for fs in scores]
        colors = ["#48d597" if n == selected else "#30638e" for n in names]
        ax.bar(names, rs, color=colors, edgecolor="white")
        for i, fs in enumerate(scores):
            star = " *" if fs.framing == selected else ""
            txt = f"{fs.pearson:.2f}{star}" if fs.pearson is not None else "n/a"
            ax.text(i, rs[i], txt, ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no scored framings", ha="center", va="center",
                transform=ax.transAxes)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Pearson r (predicted risk vs gold)")
    ax.set_title(f"MentalRiskES framing selection ({model_name})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_framing_mean_risk(
    scores: list[FramingScore], dataset: RiskDataset, out_path: Path
) -> Path:
    """Per-framing mean predicted risk with the mean gold drawn as a reference."""
    golds = [s.gold_risk for s in dataset.samples if s.gold_risk is not None]
    mean_gold = float(np.mean(golds)) if golds else None
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(scores)), 5))
    if scores:
        names = [fs.framing for fs in scores]
        means = [fs.mean_risk for fs in scores]
        ax.bar(names, means, color="#9b7dff", edgecolor="white")
        if mean_gold is not None:
            ax.axhline(mean_gold, color="#d1495b", linestyle="--",
                       label=f"mean gold {mean_gold:.2f}")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "no scored framings", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean predicted risk")
    ax.set_title(f"MentalRiskES mean risk by framing ({dataset.model_name})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    """Load the selection RiskDataset, rank framings, and render the plots."""
    args = parse_args()
    log_header("VISUALIZE SELECTION RISK")
    dataset = RiskDataset.from_json(args.samples)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    scores = score_framings(dataset.samples)
    selected = best_framing(scores)
    log_section("framing ranking (best gold-tracking first)")
    for fs in scores:
        star = " <- SELECTED" if fs.framing == selected else ""
        log(f"  {fs.framing:<14} r={fs.pearson} mae={fs.mae} n={fs.n}{star}")

    plots_dir = args.out_dir / "mental_risk" / "selection" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    written = [
        plot_framing_correlation(scores, selected, dataset.model_name,
                                 plots_dir / "framing_correlation.png"),
        plot_framing_mean_risk(scores, dataset, plots_dir / "framing_mean_risk.png"),
    ]
    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
