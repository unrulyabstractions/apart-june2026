"""How much the model's urge to answer 'unknown' wobbles with harmless rewording.

For each item we look at how strongly the model leans toward answering 'unknown'
(its abstention tendency) and measure how much that leaning swings across the
item's reworded formats. A wide swing means the same question dressed differently
makes the model far more or less willing to abstain. Two panels: ambiguous
questions (top), where 'unknown' is the correct answer, and clear questions
(bottom), where it is not.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.common.math import bootstrap_ci
from stability_plot_style import COND_COLOR, COND_LABEL, MEAN_RED, save_figure, titled


def _spread_panel(ax, spreads: list[float], cond: str) -> None:
    """One panel: distribution of per-item abstention swing + mean with 95% CI."""
    if not spreads:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="#999999")
        ax.set_ylim(bottom=0)
    elif np.ptp(spreads) < 1e-3:
        # Degenerate case: every item swings the same tiny amount, so a KDE would
        # explode to a meaningless needle. Just mark where they all pile up.
        x = float(np.mean(spreads))
        ax.axvline(x, color=COND_COLOR[cond], linewidth=3, zorder=2)
        ax.annotate(f"all {len(spreads)} items barely swing (about {x:.2f})",
                    (x, 0.5), xycoords=("data", "axes fraction"), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9,
                    color=COND_COLOR[cond])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
    else:
        sns.kdeplot(x=spreads, ax=ax, color=COND_COLOR[cond], fill=True, alpha=0.20,
                    linewidth=2, cut=0, bw_adjust=1.2)
        sns.rugplot(x=spreads, ax=ax, color=COND_COLOR[cond], height=0.07,
                    linewidth=2, alpha=0.9)
        m, lo, hi = bootstrap_ci(spreads)
        ax.axvspan(lo, hi, color=MEAN_RED, alpha=0.12, zorder=0)
        ax.axvline(m, color=MEAN_RED, linestyle="--", linewidth=2,
                   label=f"average swing = {m:.2f}  (95% CI {lo:.2f}-{hi:.2f}, "
                         f"{len(spreads)} items)")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_ylabel(f"{COND_LABEL[cond]}\nhow common", fontsize=9)


def plot_p_unknown_spread(
    spreads_by_cond: dict[str, list[float]], model: str, out_path: Path,
) -> Path:
    """Stacked ambiguous vs clear distributions of per-item abstention swing."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.8), layout="constrained", sharex=True)
    _spread_panel(axes[0], spreads_by_cond["ambig"], "ambig")
    _spread_panel(axes[1], spreads_by_cond["disambig"], "disambig")
    titled(
        axes[0],
        f"Does rewording change how often the model abstains?  ({model})",
        "further RIGHT = the model's willingness to answer 'unknown' swings more "
        "with harmless rewording (less stable)")
    axes[1].set_xlabel(
        "How much an item's tendency to answer 'unknown' swings across reworded formats")
    return save_figure(fig, out_path)
