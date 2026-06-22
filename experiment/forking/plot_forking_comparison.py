"""Compare the forking-paths O_t dynamics of the SAME item WITH vs WITHOUT a scaffold.

Run-by-path driver: loads TWO ForkingTrajectory JSONs (a no-scaffold baseline and a
debiasing-scaffold condition captured on the SAME forced item) and renders ONE
comparison figure. Top two rows are the stacked-area outcome distribution O_t for
each condition (baseline on top, scaffold below); the bottom row overlays both
conditions' answer diversity H(O_t) and abstention (unknown) mass so the reader sees
whether the scaffold locks onto "no se sabe" earlier / more strongly than baseline.

Usage:
  uv run python sesgo/forking/plot_forking_comparison.py
  uv run python sesgo/forking/plot_forking_comparison.py \
      --baseline-trajectory out/sesgo/forking/Qwen3-0.6B/forking_trajectory.json \
      --scaffold-trajectory out/sesgo/forking/Qwen3-0.6B-interpretive_direction/forking_trajectory.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.common.logging import log, log_header  # noqa: E402
from src.common.math import probs_to_logprobs, shannon_entropy  # noqa: E402
from src.dynamics.forking_paths import ForkingTrajectory  # noqa: E402

from experiment.forking.forking_plot_styles import OUTCOME_COLORS, save_fig, titled  # noqa: E402

_BASELINE_DIR = Path("out/sesgo/forking/Qwen3-0.6B")
_SCAFFOLD_DIR = Path("out/sesgo/forking/Qwen3-0.6B-interpretive_direction")
_OVERLAY = {"baseline": "#999999", "scaffold": "#0072B2"}  # grey vs blue


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the scaffold-vs-baseline comparison plot."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-trajectory", type=Path, default=_BASELINE_DIR / "forking_trajectory.json")
    p.add_argument("--scaffold-trajectory", type=Path, default=_SCAFFOLD_DIR / "forking_trajectory.json")
    p.add_argument("--out", type=Path, default=Path("out/sesgo/forking/Qwen3-0.6B_scaffold_vs_baseline.png"))
    return p.parse_args()


def _outcome_matrix(traj: ForkingTrajectory) -> np.ndarray:
    """[T, dim] stack of the per-position outcome histograms o_t."""
    return np.array([p.outcome_histogram for p in traj.positions], dtype=float)


def _diversity(o_series: np.ndarray) -> np.ndarray:
    """Shannon entropy H(o_t) per token position (answer diversity)."""
    return np.array([float(shannon_entropy(probs_to_logprobs(list(row)))) for row in o_series])


def _unknown_mass(traj: ForkingTrajectory, o_series: np.ndarray) -> np.ndarray:
    """Abstention (unknown) probability mass per token position."""
    idx = traj.outcome_set.index_of("unknown")
    return o_series[:, idx] if o_series.size else np.array([])


def _stacked(ax, traj: ForkingTrajectory, label: str) -> None:
    """Stacked-area O_t vs token position for one condition, with a corner label."""
    o_series = _outcome_matrix(traj)
    if o_series.size == 0:
        return
    xs = np.arange(o_series.shape[0])
    colors = [OUTCOME_COLORS.get(lbl, "#cccccc") for lbl in traj.outcome_set.labels]
    ax.stackplot(xs, o_series.T, labels=traj.outcome_set.labels, colors=colors, alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(int(xs[-1]), 1))
    ax.set_ylabel("O_t")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.85)
    titled(ax, label)


def _overlay(ax, base: ForkingTrajectory, scaf: ForkingTrajectory, series_fn, ylabel: str) -> None:
    """Overlay one scalar series (diversity or unknown mass) for both conditions."""
    for traj, key, name in ((base, "baseline", "No scaffold"),
                            (scaf, "scaffold", "Interpretive-direction scaffold")):
        ys = series_fn(traj, _outcome_matrix(traj))
        ax.plot(np.arange(len(ys)), ys, "-o", ms=3, color=_OVERLAY[key], label=name)
    ax.set_xlabel("token position t")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def _diversity_overlay(ax, base, scaf) -> None:
    """Answer-diversity H(O_t) overlay panel."""
    _overlay(ax, base, scaf, lambda t, m: _diversity(m), "H(O_t)")
    titled(ax, "Answer diversity")


def _abstention_overlay(ax, base, scaf) -> None:
    """Abstention (unknown) mass overlay panel."""
    _overlay(ax, base, scaf, _unknown_mass, "unknown mass")
    ax.set_ylim(0, 1)
    titled(ax, "Abstention (no se sabe)")


def _figure(base: ForkingTrajectory, scaf: ForkingTrajectory):
    """Two stacked-area O_t rows + a diversity / abstention overlay row."""
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 3.0, 2.4], hspace=0.5, wspace=0.22)
    _stacked(fig.add_subplot(gs[0, :]), base, "No scaffold")
    _stacked(fig.add_subplot(gs[1, :]), scaf, "Interpretive-direction scaffold")
    _diversity_overlay(fig.add_subplot(gs[2, 0]), base, scaf)
    _abstention_overlay(fig.add_subplot(gs[2, 1]), base, scaf)
    return fig


def main() -> None:
    """Load both trajectories and render the scaffold-vs-baseline comparison figure."""
    args = parse_args()
    log_header("PLOT FORKING COMPARISON (scaffold vs baseline)")

    base = ForkingTrajectory.from_json(args.baseline_trajectory)
    scaf = ForkingTrajectory.from_json(args.scaffold_trajectory)
    log(f"[compare] baseline: {len(base.positions)} positions (item {base.item_question_id[:8]})")
    log(f"[compare] scaffold: {len(scaf.positions)} positions (item {scaf.item_question_id[:8]})")

    out_path = save_fig(_figure(base, scaf), args.out)
    log(f"[compare] wrote {out_path}")


if __name__ == "__main__":
    main()
