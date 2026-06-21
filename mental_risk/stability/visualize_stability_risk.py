"""Plot the STABILITY RiskDataset into PNGs under .../stability/<MODEL>/plots/.

Run-by-path driver (risk analogue of sesgo/stability/visualize_stability_samples.py).
SESGO measured how consistently the abstention LABEL survives format-only
rewrites; risk has a continuous answer, so we measure how consistent the
predicted RISK is across the format variations of a fixed (subject, framing,
task) cell. Plots:

  risk_spread_hist.png   - histogram of per-(subject, framing, task) std of
                           predicted risk across format variations (lower == more
                           format-stable).
  spread_by_task.png     - mean within-cell spread split by task type, since
                           SCORE (free number) and CATEGORIZE (forced label)
                           react to format very differently.

NOTE on the risk-vs-bias difference: SESGO's per-axis flip-rate plot needs the
sample to carry its label_style / permutation, and SesgoSample does. The risk
RiskAssessmentSample deliberately keeps only (subject, framing, task) provenance
(format axes are dropped at collection), so we report spread per CELL and per
TASK rather than attributing it to a single named format axis — an honest read of
the metadata that survives collection rather than a fabricated one.

Usage:
  uv run python mental_risk/stability/visualize_stability_risk.py
  uv run python mental_risk/stability/visualize_stability_risk.py \
      out/mental_risk/stability/Qwen3-0.6B/response_samples.json
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

from mental_risk.risk_prediction import effective_risk  # noqa: E402
from src.common.logging import log, log_header  # noqa: E402
from src.datasets.risk import RiskAssessmentSample, RiskDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for stability visualization."""
    parser = argparse.ArgumentParser(description="Plot a stability RiskDataset")
    parser.add_argument(
        "samples", type=Path, nargs="?",
        default=Path("out/mental_risk/stability/Qwen3-0.6B/response_samples.json"),
        help="Path to a stability response_samples.json (a RiskDataset)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    return parser.parse_args()


def _cell_key(s: RiskAssessmentSample) -> tuple:
    """Identity held fixed across format variations: (subject, framing, task)."""
    return (s.subject_id, s.framing, s.task_type.value)


def _cell_risks(dataset: RiskDataset) -> dict[tuple, list[float]]:
    """Predicted risk per fixed-identity cell, dropping unparseable samples."""
    cells: dict[tuple, list[float]] = defaultdict(list)
    for s in dataset.samples:
        r = effective_risk(s)
        if r is not None:
            cells[_cell_key(s)].append(r)
    return cells


def plot_risk_spread(dataset: RiskDataset, out_path: Path) -> Path:
    """Histogram of per-cell std of predicted risk across format variations."""
    spreads = [float(np.std(v)) for v in _cell_risks(dataset).values() if len(v) > 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    if spreads:
        ax.hist(spreads, bins=20, color="#30638e", edgecolor="white")
        ax.axvline(float(np.mean(spreads)), color="#d1495b", linestyle="--",
                   label=f"mean {np.mean(spreads):.3f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no multi-variation cells", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("std of predicted risk across format variations")
    ax.set_ylabel("number of (subject, framing, task) cells")
    ax.set_title(f"MentalRiskES format spread of predicted risk ({dataset.model_name})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_spread_by_task(dataset: RiskDataset, out_path: Path) -> Path:
    """Bar chart of mean within-cell risk spread split by task type."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for (subject, framing, task), risks in _cell_risks(dataset).items():
        if len(risks) > 1:
            by_task[task].append(float(np.std(risks)))
    tasks = sorted(by_task)
    means = [float(np.mean(by_task[t])) for t in tasks]
    fig, ax = plt.subplots(figsize=(6, 5))
    if tasks:
        ax.bar(tasks, means, color="#9b7dff", edgecolor="white")
        for i, h in enumerate(means):
            ax.text(i, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no multi-variation cells", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel("mean within-cell std of predicted risk")
    ax.set_title(f"MentalRiskES format spread by task ({dataset.model_name})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    """Load the stability RiskDataset and render the stability plots."""
    args = parse_args()
    log_header("VISUALIZE STABILITY RISK")
    dataset = RiskDataset.from_json(args.samples)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    plots_dir = args.out_dir / "mental_risk" / "stability" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    written = [
        plot_risk_spread(dataset, plots_dir / "risk_spread_hist.png"),
        plot_spread_by_task(dataset, plots_dir / "spread_by_task.png"),
    ]
    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
