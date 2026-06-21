"""Plot the DIVERGENCE RiskDataset into PNGs under .../divergence/<MODEL>/plots/.

Run-by-path driver (risk analogue of sesgo/divergence/visualize_divergence_samples.py).
SESGO characterized the spread of the 3-way thinking role distribution (entropy,
JS-from-gold); risk has a continuous sampled score cloud per prompt summarized by
ScoreSummary, so we plot the cloud's spread directly. Plots:

  thinking_entropy_hist.png   - histogram of per-prompt Shannon entropy of the
                                sampled risk-score distribution (+ mean line).
  thinking_std_hist.png       - histogram of per-prompt std of the sampled scores
                                (dispersion across draws).
  abs_error_from_gold_hist.png- histogram of |mean thinking risk - gold risk|, the
                                risk analogue of SESGO's JS-from-gold-UNKNOWN.
  entropy_by_task.png         - mean entropy split by task type (SCORE vs
                                CATEGORIZE), since a free number and a forced label
                                produce very different clouds.

Usage:
  uv run python mental_risk/divergence/visualize_divergence_risk.py
  uv run python mental_risk/divergence/visualize_divergence_risk.py \
      out/mental_risk/divergence/Qwen3-0.6B/samples.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header  # noqa: E402
from src.datasets.risk import RiskDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for divergence visualization."""
    parser = argparse.ArgumentParser(description="Plot a divergence RiskDataset")
    parser.add_argument(
        "samples", type=Path, nargs="?",
        default=Path("out/mental_risk/divergence/Qwen3-0.6B/samples.json"),
        help="Path to a divergence samples.json (a RiskDataset)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    return parser.parse_args()


def _hist(values: list[float], xlabel: str, title: str, out_path: Path, color: str) -> Path:
    """A small labelled histogram with a mean line (shared by the spread plots)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if values:
        ax.hist(values, bins=20, color=color, edgecolor="white")
        ax.axvline(float(np.mean(values)), color="#d1495b", linestyle="--",
                   label=f"mean {np.mean(values):.3f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no parsed thinking draws", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of prompts")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_entropy_by_task(dataset: RiskDataset, out_path: Path) -> Path:
    """Bar chart of mean thinking-entropy split by task type."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for s in dataset.samples:
        if s.thinking is not None and s.thinking.n > 0:
            by_task[s.task_type.value].append(s.thinking.entropy)
    tasks = sorted(by_task)
    means = [float(np.mean(by_task[t])) for t in tasks]
    fig, ax = plt.subplots(figsize=(6, 5))
    if tasks:
        ax.bar(tasks, means, color="#43c6e8", edgecolor="white")
        for i, h in enumerate(means):
            ax.text(i, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no parsed thinking draws", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel("mean thinking entropy (nats)")
    ax.set_title(f"MentalRiskES thinking entropy by task ({dataset.model_name})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    """Load the divergence RiskDataset and render the thinking-cloud plots."""
    args = parse_args()
    log_header("VISUALIZE DIVERGENCE RISK")
    dataset = RiskDataset.from_json(args.samples)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    parsed = [s for s in dataset.samples if s.thinking is not None and s.thinking.n > 0]
    entropies = [s.thinking.entropy for s in parsed]
    stds = [s.thinking.std for s in parsed]
    abs_err = [abs(s.thinking.mean - s.gold_risk) for s in parsed if s.gold_risk is not None]

    plots_dir = args.out_dir / "mental_risk" / "divergence" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    written = [
        _hist(entropies, "Shannon entropy of sampled risk scores (nats)",
              f"MentalRiskES thinking entropy ({dataset.model_name})",
              plots_dir / "thinking_entropy_hist.png", "#30638e"),
        _hist(stds, "std of sampled risk scores",
              f"MentalRiskES thinking dispersion ({dataset.model_name})",
              plots_dir / "thinking_std_hist.png", "#9b7dff"),
        _hist(abs_err, "|mean thinking risk - gold risk|",
              f"MentalRiskES thinking error from gold ({dataset.model_name})",
              plots_dir / "abs_error_from_gold_hist.png", "#ffb454"),
        plot_entropy_by_task(dataset, plots_dir / "entropy_by_task.png"),
    ]
    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
