"""Plot a collected RiskDataset into PNGs under out/mental_risk/<MODEL>/plots/.

Run-by-path driver. Loads a responses.json produced by collect_llm_responses.py
(a RiskDataset) and renders predicted-vs-gold scatter plots (non-thinking and
thinking, colored by framing, annotated with Pearson r) plus mean-predicted-risk
bar charts by framing and by language. Thinking predictions with no parsed draw
(n == 0) have no usable score and are excluded.

Usage:
  uv run python mental_risk/baseline/visualize_llm_responses.py \
      out/mental_risk/Qwen3-0.6B/responses.json
  uv run python mental_risk/baseline/visualize_llm_responses.py RESPONSES.json --out-dir out
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

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/mental_risk/baseline/x.py, parents[2] is root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header  # noqa: E402
from src.datasets.risk import RiskDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for risk visualization."""
    parser = argparse.ArgumentParser(
        description="Plot a collected RiskDataset into PNGs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "responses",
        type=Path,
        help="Path to responses.json (a RiskDataset) from collect_llm_responses.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output directory; plots land at <out-dir>/mental_risk/<MODEL>/plots/",
    )
    return parser.parse_args()


def _pearson(preds: list[float], golds: list[float]) -> float | None:
    """Pearson r between paired predictions and golds, or None if degenerate."""
    if len(preds) < 2 or np.std(preds) == 0 or np.std(golds) == 0:
        return None
    return float(np.corrcoef(preds, golds)[0, 1])


def plot_pred_vs_gold(dataset: RiskDataset, level: str, out_path: Path) -> Path:
    """Scatter predicted risk vs gold, colored by framing; annotate Pearson r."""
    attr = (
        "predicted_risk_non_thinking"
        if level == "non_thinking"
        else "predicted_risk_thinking"
    )
    rows = [
        (getattr(s, attr), s.gold_risk, s.framing)
        for s in dataset.samples
        if getattr(s, attr) is not None and s.gold_risk is not None
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    pretty = "non-thinking" if level == "non_thinking" else "thinking"
    if rows:
        framings = sorted({f for _, _, f in rows})
        palette = dict(zip(framings, sns.color_palette("tab10", len(framings))))
        for f in framings:
            xs = [g for p, g, fr in rows if fr == f]
            ys = [p for p, g, fr in rows if fr == f]
            ax.scatter(xs, ys, label=f, color=palette[f], alpha=0.7, s=40)
        ax.legend(title="framing", fontsize=8)
        preds = [p for p, _, _ in rows]
        golds = [g for _, g, _ in rows]
        r = _pearson(preds, golds)
        r_txt = f"Pearson r = {r:.3f}" if r is not None else "Pearson r = n/a"
        log(f"[viz] {pretty} corr(pred, gold) = {r}")
        ax.text(
            0.02,
            0.98,
            r_txt,
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"},
        )
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="y = x")
    ax.set_xlabel("gold risk")
    ax.set_ylabel(f"predicted risk ({pretty})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"MentalRiskES {pretty} predicted vs gold risk ({dataset.model_name})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_mean_risk_by(dataset: RiskDataset, group_attr: str, out_path: Path) -> Path:
    """Grouped bars of mean predicted risk (both levels) by `group_attr`."""
    groups = sorted({getattr(s, group_attr) for s in dataset.samples})
    nt: dict[str, list[float]] = defaultdict(list)
    th: dict[str, list[float]] = defaultdict(list)
    for s in dataset.samples:
        g = getattr(s, group_attr)
        if s.predicted_risk_non_thinking is not None:
            nt[g].append(s.predicted_risk_non_thinking)
        if s.predicted_risk_thinking is not None:
            th[g].append(s.predicted_risk_thinking)

    def _mean(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    nt_means = [_mean(nt.get(g, [])) for g in groups]
    th_means = [_mean(th.get(g, [])) for g in groups]

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(groups)), 5))
    x = np.arange(len(groups))
    width = 0.38
    ax.bar(x - width / 2, nt_means, width, label="non-thinking", color="#30638e")
    ax.bar(x + width / 2, th_means, width, label="thinking", color="#d1495b")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean predicted risk")
    ax.set_title(
        f"MentalRiskES mean predicted risk by {group_attr} ({dataset.model_name})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    """Load the RiskDataset and render all plots into the plots/ dir."""
    args = parse_args()
    log_header("VISUALIZE MENTAL_RISK RESPONSES")

    dataset = RiskDataset.from_json(args.responses)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    plots_dir = args.out_dir / "mental_risk" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    written: list[Path] = []
    written.append(
        plot_pred_vs_gold(
            dataset, "non_thinking", plots_dir / "pred_vs_gold_non_thinking.png"
        )
    )
    if any(s.predicted_risk_thinking is not None for s in dataset.samples):
        written.append(
            plot_pred_vs_gold(
                dataset, "thinking", plots_dir / "pred_vs_gold_thinking.png"
            )
        )
    else:
        log("[viz] no parsed thinking draws; skipping thinking scatter")

    written.append(
        plot_mean_risk_by(dataset, "framing", plots_dir / "mean_risk_by_framing.png")
    )
    written.append(
        plot_mean_risk_by(dataset, "language", plots_dir / "mean_risk_by_language.png")
    )

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
