"""Plot the held-out SESGO steering test: abstention vs steering strength.

Reads one or more ``steering_test.json`` bundles (one per model) and draws a
single multi-panel figure — one panel per model — of abstention against the
steering strength. Each panel overlays the held-out curve (the causal claim) and
the in-sample curve (the fitted items), and marks the no-steering point, the
push-the-opposite-way control region, and the debiasing scaffold's own level (the
behaviour the steering direction is trying to reproduce).

The figure is the visual statement of the causal finding: on questions the
steering direction was NEVER fit on, does pushing harder toward the scaffold
direction raise abstention (the model answering 'unknown') the way the real
scaffold does — while pushing the opposite way lowers it?

Usage (defaults to all three Qwen3 models under out/sesgo/steer/):
  .venv/bin/python sesgo/steer/plot_steering_test.py \
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

# Plain-language y-axis title per metric.
_METRIC_AXIS = {
    "abstain_rate": "Abstention rate",
    "mean_unknown_prob": "P(unknown)",
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
        "--metric", choices=tuple(_METRIC_AXIS), default="abstain_rate",
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


def _shared_legend(fig, ax) -> None:
    """One decluttered legend under the panels, deduped across the example axis."""
    handles, labels = ax.get_legend_handles_labels()
    seen, h, lab = set(), [], []
    for hd, lb in zip(handles, labels):
        if lb not in seen:
            seen.add(lb)
            h.append(hd)
            lab.append(lb)
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))


def build_figure(results: list[SteeringTestResult], metric: str):
    """One row of per-model abstention-vs-steering panels sharing a y-axis."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.7 * n, 5.0), sharey=True, squeeze=False)
    for ax, result in zip(axes[0], results):
        draw_panel(ax, result, metric)
    axes[0][0].set_ylabel(_METRIC_AXIS[metric], fontsize=11)
    _shared_legend(fig, axes[0][0])
    fig.tight_layout(rect=(0, 0.08, 1, 0.99))
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
