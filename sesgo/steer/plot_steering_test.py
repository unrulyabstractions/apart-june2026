"""Plot the held-out SESGO steering test: abstention vs steering strength alpha.

Reads one or more ``steering_test.json`` bundles (one per model) and draws a
single multi-panel figure — one panel per model — of abstention against the
steering strength alpha. Each panel overlays the held-out TEST curve (the causal
claim) and the in-sample TRAIN curve (generalization check), and marks the
alpha=0 unsteered baseline, the negative-alpha control region, and the real
debiasing-scaffold reference (the target behaviour +v aims to reproduce).

The figure is the visual statement of the causal finding: on items the steering
vector was NEVER fit on, does adding +alpha*v raise abstention (UNKNOWN mass)
monotonically, while the negative control drops it, approaching the scaffold?

Usage (defaults to all three Qwen3 models under out/sesgo/steer/):
  uv run python sesgo/steer/plot_steering_test.py \
      out/sesgo/steer/Qwen3-0.6B/steering_test.json \
      out/sesgo/steer/Qwen3-1.7B/steering_test.json \
      out/sesgo/steer/Qwen3-4B/steering_test.json \
      [--metric abstain_rate|mean_unknown_prob]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import ensure_dir  # noqa: E402
from src.common.logging import log, log_header  # noqa: E402
from src.steer import SteeringTestResult  # noqa: E402
from steering_plot_styles import draw_panel, save_fig  # noqa: E402

_METRIC_LABELS = {
    "abstain_rate": "abstention rate  (argmax = UNKNOWN)",
    "mean_unknown_prob": "mean UNKNOWN probability",
}
_DEFAULT_INPUTS = (
    "out/sesgo/steer/Qwen3-0.6B/steering_test.json",
    "out/sesgo/steer/Qwen3-1.7B/steering_test.json",
    "out/sesgo/steer/Qwen3-4B/steering_test.json",
)


def parse_args() -> argparse.Namespace:
    """Parse the input bundles, the metric to plot, and the output path."""
    parser = argparse.ArgumentParser(description="Plot the SESGO steering test")
    parser.add_argument(
        "results", type=Path, nargs="*", help="steering_test.json bundle(s), one per model"
    )
    parser.add_argument(
        "--metric", choices=tuple(_METRIC_LABELS), default="abstain_rate",
        help="abstention metric to plot (default: abstain_rate)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("out/sesgo/steer/figures/abstention_vs_alpha.png"),
        help="output figure path",
    )
    return parser.parse_args()


def _model_size_billions(model: str) -> float:
    """Parse the 'NB'/'NNB' parameter count out of a model name (for panel order)."""
    tag = model.split("/")[-1].split("-")[-1]  # e.g. "0.6B", "1.7B", "4B"
    return float(tag.rstrip("Bb")) if tag.rstrip("Bb").replace(".", "").isdigit() else 0.0


def _load_results(paths: list[Path]) -> list[SteeringTestResult]:
    """Load each bundle, smallest model first (panels read left->right by size)."""
    results = [SteeringTestResult.from_json(p) for p in paths]
    return sorted(results, key=lambda r: _model_size_billions(r.model))


def build_figure(results: list[SteeringTestResult], metric: str):
    """One row of per-model abstention-vs-alpha panels sharing a y-axis."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.6), sharey=True, squeeze=False)
    for ax, result in zip(axes[0], results):
        draw_panel(ax, result, metric)
    axes[0][0].set_ylabel(_METRIC_LABELS[metric])
    fig.suptitle(
        "SESGO causal steering: held-out abstention vs steering strength",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, 0.965,
        "+alpha*v added to resid_post on TEST ambiguous no-scaffold items the "
        "vector never saw; -alpha is the control",
        ha="center", fontsize=9.5, style="italic", color="#555555",
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Load every model's steering test and render the cross-model figure."""
    args = parse_args()
    log_header("PLOT SESGO STEERING TEST")
    paths = args.results or [Path(p) for p in _DEFAULT_INPUTS]
    results = _load_results([p for p in paths if p.exists()])
    log(f"[plot] models={[r.model.split('/')[-1] for r in results]} metric={args.metric}")

    fig = build_figure(results, args.metric)
    ensure_dir(args.out.parent)
    save_fig(fig, args.out)
    log(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
