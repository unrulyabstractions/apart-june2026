"""Full-data baseline abstention figure across language / origin / scaffold.

One figure, three columns (the three NEW axes the full-data study adds) x three rows
(the three readouts: 3-option teacher-forced, 2-option forced choice, greedy-thinking).
Each panel is a bar of ambiguous abstention accuracy per axis value (es vs en;
original vs BBQ-adapted; no-scaffold vs each of the three scaffolds), with a Wilson 95%
CI whisker and the usable n annotated. The 2-option readout has no UNKNOWN, so its row
is honestly empty (n/a) — abstention is undefined without an abstain option.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.common.math import wilson_err

from sesgo.baseline.baseline_accuracy_slices import AccuracyCell
from sesgo.baseline.baseline_full_axis_slices import AXES
from sesgo.baseline.cross_model_aggregation import COND_TITLES, CONDITIONS

# Okabe–Ito palette, one hue per axis so the three columns read as distinct facets.
_AXIS_COLOR = {"language": "#0072B2", "origin": "#D55E00", "scaffold": "#009E73"}
_AXIS_TITLE = {
    "language": "language (es vs en)",
    "origin": "origin (original vs BBQ-adapted)",
    "scaffold": "scaffold (none vs the 3 debiasing scaffolds)",
}
# Compact axis-value labels so the scaffold ticks stay legible.
_SHORT = {
    "(none)": "none",
    "interpretive_direction": "interp.\ndirection",
    "prior_dominance_warning": "prior\ndominance",
    "intent_and_register_respect": "intent &\nregister",
    "insufficient_information_unknown": "insuff.\ninfo",
}


def _abstention_bar(ax, x: float, cell: AccuracyCell, color: str) -> None:
    """One abstention bar (Wilson CI + n), or an italic n/a tick when no data."""
    if cell.total == 0:
        ax.text(x, 0.06, "n/a", ha="center", va="bottom", fontsize=8,
                style="italic", color="#999999", rotation=90)
        return
    below, above = (max(0.0, e) for e in wilson_err(cell.successes, cell.total))
    ax.bar(x, cell.accuracy, width=0.6, color=color, zorder=3)
    ax.errorbar(x, cell.accuracy, yerr=[[below], [above]], fmt="none",
                ecolor="#222222", elinewidth=1.0, capsize=2.5, zorder=4)
    ax.text(x, min(cell.accuracy + above + 0.02, 1.12),
            f"{cell.accuracy:.0%}\nn={cell.total}", ha="center", va="bottom", fontsize=7.5)


def _axis_panel(ax, cond: str, axis: str, cells: list[AccuracyCell], values: list[str]) -> None:
    """One (readout, axis) panel: an abstention bar per axis value."""
    lut = {c.slice_label: c for c in cells}
    for xi, value in enumerate(values):
        cell = lut.get(value)
        if cell is not None:
            _abstention_bar(ax, xi, cell, _AXIS_COLOR[axis])
    ax.axhline(1 / 3, ls="--", lw=1.0, color="#888888", alpha=0.8, zorder=1)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([_SHORT.get(v, v) for v in values], fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.margins(x=0.08)


def plot_full_axes(cells_by_axis: dict, axis_vals: dict, model: str, n: int, out_path):
    """Grid figure: rows = readouts, columns = the three new axes. Returns out_path."""
    conds = [c for c, _ in CONDITIONS]
    # Width per column scales with how many bars it holds (scaffold has 4, the
    # binary axes have 2) so no panel is cramped and the figure stays compact.
    widths = [max(2.0, 1.1 * len(axis_vals[axis])) for axis in AXES]
    fig, axes = plt.subplots(
        len(conds), len(AXES), figsize=(sum(widths) + 1.5, 3.2 * len(conds)),
        squeeze=False, gridspec_kw={"width_ratios": widths},
    )
    for ri, cond in enumerate(conds):
        for ci, axis in enumerate(AXES):
            ax = axes[ri][ci]
            cells = [c for c in cells_by_axis[axis] if c.condition == cond]
            _axis_panel(ax, cond, axis, cells, axis_vals[axis])
            if ri == 0:
                ax.set_title(_AXIS_TITLE[axis], fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{COND_TITLES.get(cond, cond)}\nabstention acc", fontsize=8.5)
    fig.suptitle(
        f"SESGO full-data baseline · abstention by language / origin / scaffold · {model}"
        f"  (n={n} scored ambiguous items)\n"
        "accuracy = fraction predicted UNKNOWN on ambiguous items (gold=unknown); "
        "dashed line = 1/3 chance",
        fontsize=13, fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.08, wspace=0.18, hspace=0.30)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
