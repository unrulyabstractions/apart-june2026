"""Assemble the SELECTION study's two stacked-subfigure figures (ONE file each).

``figure_accuracy_by_scaffold`` stacks the readout LEVELS as subfigure rows for a
single context condition: ambiguous → abstention (gold = UNKNOWN), disambiguated →
accuracy (gold = the ground-truth role). Only levels with data are drawn, and the
2-option level is omitted for ambiguous items (undefined there), so no degenerate
all-n/a panel ever appears.

``figure_two_vs_three_option`` stacks 2-OPTION (top) over 3-OPTION (bottom) for a
fixed category on disambiguated items — the only context where the forced choice is
defined — so the bias-direction readout and the full readout read together.

Both delegate every bar/CI/highlight detail to selection_plot_helpers; this module
only chooses which panels exist and wires the shared ranking + titles.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from selection_metrics_helpers import (
    LEVEL_TITLE,
    ScaffoldRanking,
    counts_by_scaffold,
    counts_by_scaffold_category,
    rank_scaffolds,
)
from selection_plot_helpers import _panel, _save, _suptitle

# Y-axis labels read differently per context: abstention vs accuracy.
_YLABEL = {
    "ambig": "abstention\n(pred = UNKNOWN)",
    "disambig": "accuracy\n(pred = gold role)",
}
_CONTEXT_TAKEAWAY = {
    "ambig": "ambiguous items — gold is always UNKNOWN, so higher = better abstention",
    "disambig": "disambiguated items — gold is the ground-truth role, so higher = better accuracy",
}
# Option count per level (drives the chance-line reference).
_OPT_COUNT = {
    "non_thinking": 3,
    "greedy_thinking": 3,
    "thinking": 3,
    "non_thinking_2opt": 2,
}


def _levels_for_context(dataset, context: str, levels: list[str]) -> list[str]:
    """Keep only levels that have at least one defined prediction in this context."""
    present = []
    for level in levels:
        counts = counts_by_scaffold(dataset, level, context)
        if any(c.total > 0 for c in counts.values()):
            present.append(level)
    return present


def figure_accuracy_by_scaffold(
    dataset, scaffolds: list[str], context: str, levels: list[str],
    select_level: str, model: str, out_path: Path,
) -> tuple[Path, ScaffoldRanking]:
    """Stacked per-scaffold accuracy across readout LEVELS for one context.

    Scaffolds are ranked best→worst on ``select_level`` and that order drives EVERY
    panel, so the SELECTED scaffold (gold star) sits in the same column throughout.
    """
    drawn = _levels_for_context(dataset, context, levels)
    sel_counts = counts_by_scaffold(dataset, select_level, context)
    ranking = rank_scaffolds(sel_counts, scaffolds)
    order = ranking.order

    # Always draw at least the SELECT level so the figure is never empty (its
    # bars carry the honest "n/a" stubs when a cell has no defined prediction).
    if not drawn:
        drawn = [select_level]
    n = len(drawn)
    fig, axes = plt.subplots(n, 1, figsize=(max(8.5, 1.7 * len(order)), 3.1 * n + 0.6),
                             layout="constrained", squeeze=False)
    for ax, level in zip(axes[:, 0], drawn):
        counts = sel_counts if level == select_level else counts_by_scaffold(
            dataset, level, context)
        _panel(ax, counts, order, ranking.selected, level, _YLABEL[context],
               _OPT_COUNT[level])
    axes[-1, 0].set_xlabel("scaffold condition  (sorted best→worst, baseline + 4 debiasers)")
    _suptitle(
        fig,
        f"SESGO selection — per-scaffold {('abstention' if context=='ambig' else 'accuracy')} "
        f"by readout  ·  {model}\nSELECTED: {ranking.selected or 'n/a'}  (by {LEVEL_TITLE[select_level]})",
        _CONTEXT_TAKEAWAY[context],
    )
    return _save(fig, out_path), ranking


def category_has_data(dataset, category: str, min_total: int = 2) -> bool:
    """True iff this category has enough disambiguated 3-option data to plot.

    Guards against emitting a degenerate all-"n/a" figure for a category that the
    subsample barely (or never) touched on disambiguated items.
    """
    three = counts_by_scaffold_category(dataset, "non_thinking", "disambig", category)
    return sum(c.total for c in three.values()) >= min_total


def figure_two_vs_three_option(
    dataset, scaffolds: list[str], category: str, model: str, out_path: Path,
) -> Path:
    """Stacked 2-OPTION (top) over 3-OPTION (bottom) accuracy for one category.

    Disambiguated items only (the forced choice has no UNKNOWN to abstain to). The
    3-option non-thinking accuracy drives the shared best→worst scaffold order.
    """
    three = counts_by_scaffold_category(dataset, "non_thinking", "disambig", category)
    two = counts_by_scaffold_category(dataset, "non_thinking_2opt", "disambig", category)
    ranking = rank_scaffolds(three, scaffolds)
    order = ranking.order

    fig, axes = plt.subplots(2, 1, figsize=(max(8.5, 1.7 * len(order)), 7.0),
                             layout="constrained", sharex=True)
    _panel(axes[0], two, order, ranking.selected, "non_thinking_2opt",
           "accuracy\n(target vs other)", 2)
    _panel(axes[1], three, order, ranking.selected, "non_thinking",
           "accuracy\n(pred = gold role)", 3)
    axes[1].set_xlabel("scaffold condition  (sorted best→worst by 3-option accuracy)")
    _suptitle(
        fig,
        f"SESGO selection — 2-option vs 3-option accuracy  ·  {category}  ·  {model}\n"
        f"SELECTED: {ranking.selected or 'n/a'}",
        "disambiguated items only — forced choice (2-opt) vs full readout (3-opt)",
    )
    return _save(fig, out_path)
