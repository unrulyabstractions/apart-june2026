"""How often the model gives the SAME answer when only the format is reworded.

Each item is shown in 18 superficially different formats (option labels and the
order the groups are listed) that never change the correct answer. The histograms
count items by what share of those 18 formats produced the item's most common
answer: 1.0 means the model never wavered, lower means the wording alone changed
its mind. Two panels: answering with three options (top) and forced to a two-way
choice (bottom).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.math import sem
from stability_metrics_helpers import ConsistencySet
from stability_plot_style import (
    COND_COLOR, COND_LABEL, READOUT_NAME, ZONE_GREY, save_figure, titled,
)


def _consistency_panel(ax, sets: dict[str, ConsistencySet], readout: str) -> None:
    """One panel: clear-vs-ambiguous histograms with each condition's mean+-SEM."""
    bins = np.linspace(0, 1, 11)
    # Shade the "barely wavers" region (>0.8) so the eye finds the stable items.
    ax.axvspan(0.8, 1.0, color=ZONE_GREY, alpha=0.14, zorder=0)
    max_h = 0.0
    for cond in ("ambig", "disambig"):
        cs = sets[cond]
        if not cs.consistency:
            continue
        counts, _, _ = ax.hist(
            cs.consistency, bins=bins, alpha=0.60, color=COND_COLOR[cond],
            edgecolor="white", label=f"{COND_LABEL[cond]}  (n={len(cs.consistency)})")
        max_h = max(max_h, counts.max())
    top = max(1.0, max_h) * 1.40
    ax.set_ylim(0, top)
    # "Very stable" marker high in the shaded band, above the mean-marker rows.
    ax.text(0.9, top * 0.88, "very stable\n(>80% agree)", ha="center", va="center",
            fontsize=8.5, color="#888888")
    # Each condition's mean as a dashed line + a capped SEM whisker, parked in the
    # empty mid-band (below the upper-left legend) with the label offset onto
    # whitespace and a white bbox so nothing overprints the bars or the legend.
    for j, cond in enumerate(("ambig", "disambig")):
        cs = sets[cond]
        if not cs.consistency:
            continue
        m, e = float(np.mean(cs.consistency)), sem(cs.consistency)
        y = top * (0.58 - 0.085 * j)
        ax.axvline(m, color=COND_COLOR[cond], linestyle="--", linewidth=1.8, zorder=2)
        # Keep the dot off the right spine so a 100% mean is never clipped.
        mx = min(m, 0.985)
        ax.errorbar(mx, y, xerr=e, fmt="o", color=COND_COLOR[cond], capsize=4,
                    markersize=5, elinewidth=1.6, zorder=3)
        ha, dx = ("right", -e - 0.02) if m > 0.6 else ("left", e + 0.02)
        ax.annotate(f"average {m:.0%} (+-{e:.0%})", (mx, y), xytext=(mx + dx, y),
                    ha=ha, va="center", fontsize=8.5, color=COND_COLOR[cond],
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.set_xlim(0, 1)
    ax.set_ylabel(f"{readout}\nnumber of items", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)


def plot_consistency(
    three_opt: dict[str, ConsistencySet], two_opt: dict[str, ConsistencySet],
    model: str, out_path: Path,
) -> Path:
    """Stacked answer-stability histograms by context condition, both readouts."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 8.4), layout="constrained", sharex=True)
    _consistency_panel(axes[0], three_opt, READOUT_NAME[False])
    _consistency_panel(axes[1], two_opt, READOUT_NAME[True])
    titled(
        axes[0],
        f"Does only rewording the format change the answer?  ({model})",
        "bars further RIGHT = the model gives the same answer across reworded "
        "formats (more stable)")
    axes[1].set_xlabel(
        "Share of an item's reworded formats that gave its most common answer")
    return save_figure(fig, out_path)
