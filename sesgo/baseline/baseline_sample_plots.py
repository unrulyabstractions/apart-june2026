"""Single-model SESGO baseline ACCURACY figure (one row per answering mode).

Stacked subfigures, one ROW per answering mode (answers directly with three
options / forced two-way choice / reasons first), with grouped bars per bias
category showing the three accuracy slices: correct abstention on ambiguous
questions, accuracy when the stereotyped group is correct, and accuracy when the
other group is correct. The forced two-way row has no abstention slice (no
"unknown"), drawn as an explicit "n/a" marker. Wilson 95% CI whisker + n on every
bar. The companion role-probability figure lives in ``baseline_role_prob_plot.py``.

All rendered text uses the shared plain-language vocabulary (no pipeline jargon).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.common.math import wilson_err

from sesgo.baseline.baseline_accuracy_slices import SLICES, AccuracyCell
from sesgo.baseline.baseline_plot_palette import (
    CHANCE,
    COND_TITLE,
    SLICE_COLORS,
    SLICE_LABELS,
    ordered_categories,
)
from sesgo.baseline.cross_model_aggregation import CONDITIONS
from sesgo.common import CATEGORY_LABEL, RANDOM_GUESS_LABEL


def _bar_with_ci(ax, x: float, cell: AccuracyCell, color: str) -> None:
    """Draw one accuracy bar (Wilson CI + n), or an italic n/a tick if no data."""
    if cell.total == 0:
        ax.text(x, 0.06, "n/a", ha="center", va="bottom", fontsize=7.5,
                style="italic", color="#999999", rotation=90)
        return
    below, above = (max(0.0, e) for e in wilson_err(cell.successes, cell.total))
    # A flat cap at y=0 keeps a 0%-accuracy bar (zero height) visibly anchored so
    # its CI whisker and n label do not read as floating in space.
    ax.plot([x - 0.13, x + 0.13], [0, 0], color=color, lw=2.2, zorder=2)
    ax.bar(x, cell.accuracy, width=0.26, color=color, zorder=3)
    ax.errorbar(x, cell.accuracy, yerr=[[below], [above]], fmt="none",
                ecolor="#222222", elinewidth=1.0, capsize=2.5, zorder=4)
    ax.text(x, min(cell.accuracy + above + 0.02, 1.12),
            f"{cell.accuracy:.0%}\nn={cell.total}", ha="center", va="bottom", fontsize=7.5)


def _accuracy_panel(ax, cond: str, cells: list[AccuracyCell], cats: list[str]) -> None:
    """One answering-mode row: grouped accuracy bars per category, all three slices."""
    lut = {(c.category, c.slice_label): c for c in cells}
    width = 0.26
    for ci, cat in enumerate(cats):
        for si, (slice_label, _, _) in enumerate(SLICES):
            x = ci + (si - (len(SLICES) - 1) / 2) * width
            cell = lut.get((cat, slice_label))
            if cell is not None:
                _bar_with_ci(ax, x, cell, SLICE_COLORS[slice_label])
    chance = CHANCE.get(cond, 1 / 3)
    ax.axhline(chance, ls="--", lw=1.0, color="#888888", alpha=0.8, zorder=1)
    ax.text(0.004, chance, f" {RANDOM_GUESS_LABEL} ({chance:.0%})",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=7.5, color="#777777")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([CATEGORY_LABEL.get(c, c) for c in cats], fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(COND_TITLE.get(cond, cond), fontsize=11.5, loc="left", fontweight="bold")
    ax.margins(x=0.04)


def plot_accuracy(cells: list[AccuracyCell], cats: list[str], model: str, n: int, out_path):
    """Stacked figure: one answering-mode row each, all in plain language."""
    cats = ordered_categories(cats)
    conds = [c for c, _ in CONDITIONS]
    fig, axes = plt.subplots(len(conds), 1, figsize=(max(8.5, 1.7 * len(cats) + 4), 11.5),
                             sharex=True)
    for ax, cond in zip(np.atleast_1d(axes), conds):
        _accuracy_panel(ax, cond, [c for c in cells if c.condition == cond], cats)
    np.atleast_1d(axes)[-1].set_xlabel("Bias category", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SLICE_COLORS[s]) for s, _, _ in SLICES]
    fig.legend(handles, [SLICE_LABELS[s] for s, _, _ in SLICES],
               title="What each bar measures (taller = better)", loc="lower center",
               ncol=1, bbox_to_anchor=(0.5, 0.0), frameon=False,
               fontsize=9.5, title_fontsize=10)
    fig.suptitle(
        f"How well {model} answers SESGO social-bias questions  (n={n} scored items)\n"
        "Each row is one way of answering.  Taller bars are better.  When the orange "
        "and light-blue bars differ a lot, the model is more accurate for one group "
        "than the other - a sign of bias.",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
