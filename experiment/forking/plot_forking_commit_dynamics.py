"""Clean TWO-PANEL forking figure: how the answer narrows as reasoning unfolds.

Narrative key figure (forking-paths, arXiv:2601.06116; the paper's F6). TOP:
stacked-area outcome distribution O_t over the FULL chain-of-thought token index
(one band per answer in plain ROLE_LABEL words; scales to any reasoning length).
BOTTOM: a SINGLE diversity curve = Shannon entropy H(O_t) of the live answers --
high when many answers are still in play, low once the model has committed. Reads
ONLY the trajectory JSON and writes ONE PNG. Render: .venv/bin/python <this file>.
"""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiment.common.plain_language_labels import CATEGORY_LABEL, ROLE_LABEL  # noqa: E402
from experiment.shard_output_paths import shard_out_dir  # noqa: E402
from src.common.file_io import load_json  # noqa: E402
from src.common.math import probs_to_logprobs, shannon_entropy  # noqa: E402
from src.common.math.confidence_intervals import wilson_err  # noqa: E402
from src.dynamics.forking_paths import ForkingTrajectory  # noqa: E402

from experiment.forking.forking_plot_styles import OUTCOME_COLORS, save_fig  # noqa: E402

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


def _outcome_matrix(traj: ForkingTrajectory) -> np.ndarray:
    """[T, dim] array of the per-position outcome distributions O_t."""
    return np.array([p.outcome_histogram for p in traj.positions])


def _outcome_panel(ax, traj: ForkingTrajectory) -> int:
    """Stacked-area O_t over token index; Wilson 95% CI on the leading-answer share.

    Returns the median rollout count behind a position (the figure-wide Wilson n).
    """
    o_series = _outcome_matrix(traj)  # [T, dim]
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
    ax.set_xlim(-0.5, max(int(xs[-1]), 1) + 0.5)
    ax.set_ylabel("Share of answers the model\nwould give here")
    # Legend OUTSIDE the axes (to the right) so it never sits on top of the bands.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
              framealpha=0.95, title_fontsize=8.0,
              title=f"Possible answers\n(~{n_med} rollouts/token;\nwhisker = Wilson 95% CI\non the leading share)")
    ax.set_title("The model's likely answer shifts as its reasoning unfolds",
                 fontsize=13, fontweight="bold", pad=26)
    ax.text(0.5, 1.012,
            "How to read: each colour is one possible answer; a tall band means the model would mostly give that answer at this point in its reasoning.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6,
            color="#555555", style="italic")
    return n_med


def _entropy_series(traj: ForkingTrajectory) -> np.ndarray:
    """Per-position answer diversity H(O_t) in nats (0 = committed, high = open)."""
    o_series = _outcome_matrix(traj)
    return np.array([float(shannon_entropy(probs_to_logprobs(row.tolist())))
                     for row in o_series])


def _diversity_panel(ax, traj: ForkingTrajectory) -> None:
    """Single diversity curve: Shannon entropy H(O_t) of the live answers.

    Plotted on its OWN y-axis (nats). The dashed line is the maximum possible
    diversity (a perfectly even split over all answer categories); the curve
    falling toward 0 is the model committing to one answer.
    """
    h = _entropy_series(traj)
    xs = np.arange(len(h))
    h_max = float(np.log(max(traj.outcome_set.dim, 2)))  # even split over categories
    ax.fill_between(xs, 0, h, color="#009E73", alpha=0.18)  # Okabe-Ito bluish-green
    ax.plot(xs, h, "-", color="#009E73", lw=2.0,
            label="Answer diversity H(O_t)")
    ax.axhline(h_max, ls="--", lw=1.2, color="#555555",
               label=f"All answers equally likely (max = ln {traj.outcome_set.dim} = {h_max:.2f})")
    ax.set_ylim(0, h_max * 1.08)
    ax.set_xlim(-0.5, max(len(xs) - 1, 1) + 0.5)
    ax.set_xlabel("Reasoning token index (order the model writes its chain of thought)")
    ax.set_ylabel("Answer diversity\n(entropy of live answers)")
    # Legend OUTSIDE the axes (to the right), matching the top panel.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
              framealpha=0.95)
    ax.set_title("How many answers are still in play as the model reasons",
                 fontsize=13, fontweight="bold", pad=24)
    ax.text(0.5, 1.012,
            "How to read: a high line means many answers are still possible here; the line dropping toward zero means the model has committed to one answer.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6,
            color="#555555", style="italic")


def build_figure(traj: ForkingTrajectory, bias_axis: str):
    """Assemble the two stacked panels sharing the full token-index x-axis."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12.5, 8.4), sharex=True,
        gridspec_kw=dict(height_ratios=[2.2, 1.1], hspace=0.42),
    )
    _outcome_panel(ax_top, traj)
    _diversity_panel(ax_bot, traj)
    for ax in (ax_top, ax_bot):
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Tracking where {traj.model.split('/')[-1]} decides its answer to an ambiguous {bias_axis} question",
        fontsize=11, color="#333333", fontweight="bold", y=0.998,
    )
    # Leave room on the right for the outside legends.
    fig.subplots_adjust(right=0.78)
    return fig


def main() -> None:
    """Load the trajectory and render the two-panel commit/diversity figure."""
    out_dir = shard_out_dir(Path("out"), "forking", "Qwen3-0.6B", 0, 1)
    traj = ForkingTrajectory.from_json(out_dir / "forking_trajectory.json")
    fig = build_figure(traj, _bias_axis(out_dir))
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_fig(fig, plots_dir / "forking_commit_dynamics.png")
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
