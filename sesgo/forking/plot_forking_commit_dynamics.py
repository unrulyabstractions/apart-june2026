"""Clean TWO-PANEL forking figure: where the answer is decided in the reasoning.

Narrative key figure (forking-paths, arXiv:2601.06116). TOP: stacked-area outcome
distribution O_t over the chain-of-thought token index (one band per answer in
plain ROLE_LABEL words). BOTTOM: the change-point posterior p(tau=t) over the same
index, marking + naming the token where the trajectory commits. Reads the
trajectory + analysis JSONs and writes ONE PNG; companion to the 5-panel
plot_forking_dynamics.py.  Render: .venv/bin/python <this file>.
"""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sesgo.common.plain_language_labels import CATEGORY_LABEL, ROLE_LABEL  # noqa: E402
from sesgo.shard_output_paths import shard_out_dir  # noqa: E402
from src.common.file_io import load_json  # noqa: E402
from src.common.math.confidence_intervals import wilson_err  # noqa: E402
from src.dynamics.forking_paths import ForkingTrajectory  # noqa: E402
from src.dynamics.forking_paths.forking_analysis_result import ForkingAnalysis  # noqa: E402

from sesgo.forking.forking_plot_styles import OUTCOME_COLORS, save_fig  # noqa: E402

# Off-set / unparseable rollouts: the paper's catch-all "Other" outcome.
_OUTCOME_LABEL = {**ROLE_LABEL, "unparseable": "Unreadable answer"}


def _bias_axis(out_dir) -> str:
    """Plain-language bias axis (Gender/Racism/...) from the sibling selected item."""
    try:
        code = load_json(out_dir / "selected_item.json")["sample"]["bias_category"]
        return CATEGORY_LABEL.get(code, "social-bias")
    except (FileNotFoundError, KeyError):
        return "social-bias"


def _rollouts_at(traj: ForkingTrajectory, t: int) -> int:
    """Total rollouts behind position t's histogram (the Wilson n)."""
    return sum(len(a.rollout_labels) for a in traj.positions[t].alternates)


def _outcome_panel(ax, traj: ForkingTrajectory) -> None:
    """Stacked-area O_t over token index; Wilson 95% CI on the leading-answer share."""
    o_series = np.array([p.outcome_histogram for p in traj.positions])  # [T, dim]
    xs = np.arange(o_series.shape[0])
    labels = traj.outcome_set.labels
    ax.stackplot(xs, o_series.T, alpha=0.92, edgecolor="white", linewidth=0.4,
                 labels=[_OUTCOME_LABEL.get(l, l.title()) for l in labels],
                 colors=[OUTCOME_COLORS.get(l, "#999999") for l in labels])
    # Honest uncertainty: Wilson 95% CI on the leading answer's share (O_t sums to
    # 1, so the whisker sits on the one proportion with real n); thinned to ~20 ticks.
    ns = [_rollouts_at(traj, int(t)) for t in xs]
    pmax = o_series.max(axis=1)
    errs = np.array([wilson_err(int(round(p * n)), n) for p, n in zip(pmax, ns)]).T
    sel = xs[:: max(1, len(xs) // 20)]
    ax.errorbar(sel, pmax[sel], yerr=errs[:, sel], fmt="none", ecolor="#111111",
                elinewidth=0.9, capsize=2.2, alpha=0.55)
    n_med = int(np.median([n for n in ns if n > 0]) or 0)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.5, max(xs[-1], 1) + 0.5)
    ax.set_ylabel("Share of answers the model\nwould give here")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2, framealpha=0.9, title_fontsize=8.0,
              title=f"Possible answers (~{n_med} rollouts/token; whisker = Wilson 95% CI on the leading share)")
    ax.set_title("The model's likely answer shifts as its reasoning unfolds",
                 fontsize=13, fontweight="bold", pad=26)
    ax.text(0.5, 1.012,
            "How to read: each colour is one possible answer; a tall band means the model would mostly give that answer at this point in its reasoning.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6,
            color="#555555", style="italic")


def _commit_token(traj: ForkingTrajectory, t: int) -> str:
    """Readable base-path token text at index t (spaces/newlines made visible)."""
    tok = traj.base_token_texts[t].replace("\n", "\\n").strip()
    return tok or "(space)"


def _commit_panel(ax, traj: ForkingTrajectory, analysis: ForkingAnalysis) -> None:
    """Change-point posterior p(tau=t) over token index; deciding token highlighted."""
    cp = analysis.change_points
    tau = np.array(cp.tau_posterior)
    ax.bar(np.arange(len(tau)), tau, color="#CC79A7", width=0.85,
           label="Chance the answer is decided at this token")
    detected = 0 <= cp.forking_token_index < len(traj.base_token_texts)
    peak = cp.forking_token_index if detected else (int(np.argmax(tau)) if tau.size else -1)
    if peak >= 0:
        col = "#D55E00" if detected else "#0072B2"
        ax.bar([peak], [tau[peak]], color=col, width=0.85,
               label=("Deciding token" if detected else "Most-likely deciding token (not yet significant)"))
        ax.annotate(
            f'{"commits" if detected else "leans"} here -> "{_commit_token(traj, peak)}"',
            xy=(peak, tau[peak]), xytext=(0.5, 0.82), textcoords="axes fraction",
            ha="center", fontsize=11, fontweight="bold", color=col,
            arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
    ax.set_ylim(0, max(float(tau.max()) * 1.3, 0.04))
    ax.set_xlim(-0.5, max(len(tau) - 1, 1) + 0.5)
    ax.set_xlabel("Reasoning token index (order the model writes its chain of thought)")
    ax.set_ylabel("How likely the answer\nis decided here")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    if detected:
        title = "One token decides the final answer"
        sub = "How to read: the tall spike is the single reasoning token where the model's answer locks in."
    else:
        title = "The answer builds up gradually -- no single deciding token"
        sub = "How to read: probability is spread thinly across tokens; here no one token reaches the deciding threshold."
    ax.set_title(title, fontsize=13, fontweight="bold", pad=24)
    ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.6, color="#555555", style="italic")


def build_figure(traj: ForkingTrajectory, analysis: ForkingAnalysis, bias_axis: str):
    """Assemble the two stacked panels sharing the token-index x-axis."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11.5, 8.2), sharex=True,
        gridspec_kw=dict(height_ratios=[2.4, 1.0], hspace=0.42),
    )
    _outcome_panel(ax_top, traj)
    _commit_panel(ax_bot, traj, analysis)
    for ax in (ax_top, ax_bot):
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Tracking where {traj.model.split('/')[-1]} decides its answer to an ambiguous {bias_axis} question",
        fontsize=11, color="#333333", fontweight="bold", y=0.998,
    )
    return fig


def main() -> None:
    """Load the trajectory + analysis and render the two-panel commit figure."""
    out_dir = shard_out_dir(Path("out"), "forking", "Qwen3-0.6B", 0, 1)
    traj = ForkingTrajectory.from_json(out_dir / "forking_trajectory.json")
    analysis = ForkingAnalysis.from_json(out_dir / "forking_analysis.json")
    fig = build_figure(traj, analysis, _bias_axis(out_dir))
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_fig(fig, plots_dir / "forking_commit_dynamics.png")
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
