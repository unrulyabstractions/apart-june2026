"""Plot the forking-paths figures from the captured trajectory + analysis.

Run-by-path driver, STAGE 5 of the forking-paths study. Renders the headline
STACKED-AREA O_t-vs-token-position figure (one colored band per outcome category,
y in [0,1]) with the detected change-point token highlighted red and the base-path
tokens drawn beneath, colored by p(tau=t|y) on a yellow->red heatmap; plus the
p(tau=t|y) curve, the pull/drift/potential states, the diversity series, and the
survival curve as companion panels. Reads the two JSONs the capture + analysis
drivers wrote; writes PNGs alongside them.

Usage:
  uv run python sesgo/forking/plot_forking_dynamics.py
  uv run python sesgo/forking/plot_forking_dynamics.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sesgo.shard_output_paths import shard_out_dir  # noqa: E402
from src.common.logging import log, log_header  # noqa: E402
from src.dynamics.forking_paths import (  # noqa: E402
    ForkingTrajectory,
    build_branching_tree,
    most_divergent_branch_index,
)
from src.dynamics.forking_paths.forking_analysis_result import ForkingAnalysis  # noqa: E402

from sesgo.forking.forking_plot_styles import (  # noqa: E402
    OUTCOME_COLORS,
    save_fig,
    titled,
    token_strip,
)
from sesgo.forking.render_branching_tree import plot_branching_tree  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI args for plotting."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name (path key only)")
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    return p.parse_args()


def _stacked_area(ax, traj: ForkingTrajectory, cp) -> None:
    """Stacked-area O_t vs token position, change-point token marked with a red line."""
    o_series = np.array([p.outcome_histogram for p in traj.positions])  # [T, dim]
    if o_series.size == 0:
        return
    xs = np.arange(o_series.shape[0])
    colors = [OUTCOME_COLORS.get(lbl, "#cccccc") for lbl in traj.outcome_set.labels]
    ax.stackplot(xs, o_series.T, labels=traj.outcome_set.labels, colors=colors, alpha=0.9)
    if cp.forking_token_index >= 0:
        ax.axvline(cp.forking_token_index, color="red", lw=2.2, ls="--",
                   label=f"forking token (t={cp.forking_token_index})")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(xs[-1], 1))
    ax.set_ylabel("cumulative outcome prob")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.85)
    titled(ax, "Outcome distribution O_t along the base thinking path")


def _states_panel(ax, st) -> None:
    """Pull / drift / potential magnitudes vs token position."""
    xs = np.arange(len(st.pull))
    ax.plot(xs, st.pull, "-o", ms=3, color="#0072B2", label="pull ||O_t||")
    ax.plot(xs, st.drift, "-o", ms=3, color="#D55E00", label="drift ||O_t-O_0||")
    ax.plot(xs, st.potential, "-o", ms=3, color="#009E73", label="potential ||O_T-O_t||")
    ax.set_ylabel("magnitude")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.set_title("Dynamic states (pull / drift / potential)", fontsize=10, fontweight="bold")


def _diversity_panel(ax, dv) -> None:
    """Balance / disruption / deviance-mean & variance vs token position."""
    xs = np.arange(len(dv.balance))
    ax.plot(xs, dv.balance, "-o", ms=3, color="#CC79A7", label="balance H(O_t)")
    ax.plot(xs, dv.disruption, "-o", ms=3, color="#E69F00", label="disruption")
    ax.plot(xs, dv.mean_deviance, "-s", ms=3, color="#56B4E9", label="E[∂]")
    ax.plot(xs, dv.var_deviance, "-^", ms=3, color="#999999", label="Var[∂]")
    ax.set_ylabel("score")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.set_title("Diversity series", fontsize=10, fontweight="bold")


def _survival_tau_panel(ax, sv, cp) -> None:
    """Survival S(t) + hazard h(t) with the p(tau=t|y) change-point curve overlaid."""
    xs = np.arange(len(sv.survival))
    ax.plot(xs, sv.survival, "-o", ms=3, color="#0072B2", label="survival S(t)")
    ax.bar(xs, sv.hazard, color="#D55E00", alpha=0.35, label="hazard h(t)")
    tau = cp.tau_posterior
    ax.plot(np.arange(len(tau)), tau, "-", color="red", lw=1.6, label="p(τ=t|y)")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("token position t")
    ax.set_ylabel("prob")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.set_title("Survival + change-point posterior", fontsize=10, fontweight="bold")


def _figure(traj: ForkingTrajectory, analysis: ForkingAnalysis):
    """Assemble the stacked-area headline + token strip + three companion panels."""
    cp = analysis.change_points
    fig = plt.figure(figsize=(13, 13))
    # 4 rows: stacked area, token strip, a 2-col states/diversity row, survival row.
    gs = fig.add_gridspec(4, 2, height_ratios=[3.0, 0.8, 2.0, 2.0], hspace=0.55, wspace=0.22)
    _stacked_area(fig.add_subplot(gs[0, :]), traj, cp)
    token_strip(fig.add_subplot(gs[1, :]), traj.base_token_texts, cp.tau_posterior, cp.forking_token_index)
    _states_panel(fig.add_subplot(gs[2, 0]), analysis.dynamic_states)
    _diversity_panel(fig.add_subplot(gs[2, 1]), analysis.diversity)
    _survival_tau_panel(fig.add_subplot(gs[3, :]), analysis.survival, cp)
    return fig


def _trunk_index(traj: ForkingTrajectory, analysis: ForkingAnalysis) -> int:
    """Decision-token position for the branching tree: the forking token.

    Prefers the SIGNIFICANT change-point argmax when the CPD localizes one; else
    falls back to the position whose alternate continuations diverge most in
    outcome (``most_divergent_branch_index``) — the model-agnostic forking
    signature — so the tree always sits on a position that genuinely branches
    (never the degenerate single-alternate boilerplate the noisy Δ_t can pick).
    """
    cp = analysis.change_points
    if cp.significant and 0 <= cp.forking_token_index < len(traj.positions):
        return cp.forking_token_index
    return max(0, most_divergent_branch_index(traj))


def _branching_tree_figure(traj: ForkingTrajectory, analysis: ForkingAnalysis, path) -> str:
    """Build + render the left-to-right branching tree at the forking token."""
    tree = build_branching_tree(traj, _trunk_index(traj, analysis), max_branches=3)
    title = f"Forking branching tree — {traj.model} (item {traj.item_question_id[:8]})"
    return plot_branching_tree(tree, path, title)


def main() -> None:
    """Load the trajectory + analysis and render the forking-paths figures."""
    args = parse_args()
    log_header(f"PLOT FORKING DYNAMICS ({args.model})")

    out_dir = shard_out_dir(args.out_dir, "forking", args.model.split("/")[-1], 0, 1)
    traj = ForkingTrajectory.from_json(out_dir / "forking_trajectory.json")
    analysis = ForkingAnalysis.from_json(out_dir / "forking_analysis.json")

    fig = _figure(traj, analysis)
    out_path = save_fig(fig, out_dir / "forking_dynamics.png")
    log(f"[plot] wrote {out_path}")

    tree_path = _branching_tree_figure(traj, analysis, out_dir / "forking_branching_tree.png")
    log(f"[plot] wrote {tree_path}")


if __name__ == "__main__":
    main()
