"""Distribution-shape cross-model figures: abstention spread, category heatmap, acc.

Three size-ordered comparisons showing the full SHAPE of a per-item distribution,
not just its mean: a violin of per-item abstention on ambiguous items (all-or-
nothing vs graded), a models × bias-category abstention heatmap (rate + n, blank
where a category is absent in a partial run), and a violin of per-item correctness
on disambiguated items (the mean is the accuracy). All rendered text is plain.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.common.math import wilson_interval

from sesgo.baseline.cross_model_distribution_stats import (
    CATEGORY_ENGLISH,
    CATEGORY_ORDER,
    ModelDistribution,
)
from sesgo.baseline.cross_model_plot_styles import (
    family_color,
    order_by_size,
    partial_note,
    tick_label,
)

# Per-model horizontal slot width: keeps violins readable however many models load.
_SLOT_W = 1.05


def _xticks(ax, models: list[ModelDistribution]) -> None:
    """Apply shared size-ordered, partial-flagged, rotated x tick labels."""
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels([tick_label(m) for m in models], fontsize=9,
                       rotation=35, ha="right", rotation_mode="anchor")


def _footer(fig, models: list[ModelDistribution]) -> None:
    """Stamp the partial-run footnote if any model is partial."""
    note = partial_note(models)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color="#555555", ha="left")


def _new_violin_fig(models: list[ModelDistribution]):
    """Per-model slot stays readable; height tracks width (~1.6:1) so a fit-to-page
    figure keeps its labels and violins large rather than squashed."""
    width = max(11.0, _SLOT_W * len(models) + 2.5)
    return plt.subplots(figsize=(width, max(8.0, width / 1.6)))


def _violin(ax, models, series, ylabel, title) -> None:
    """Coloured violins (family hue) with median box + per-model mean dot and n."""
    data = [np.asarray(s, dtype=float) for s in series]
    pos = np.arange(1, len(models) + 1)
    parts = ax.violinplot(data, positions=pos, showextrema=False, widths=0.82)
    for body, m in zip(parts["bodies"], models):
        body.set_facecolor(family_color(m))
        body.set_alpha(0.55)
        body.set_edgecolor("#333333")
    ax.boxplot(data, positions=pos, widths=0.16, showfliers=False,
               medianprops={"color": "#111111"}, boxprops={"color": "#333333"},
               whiskerprops={"color": "#333333"}, capprops={"color": "#333333"})
    for p, arr in zip(pos, data):  # mean diamond + item count above the violin
        ax.plot(p, arr.mean(), "D", color="#111111", markersize=5, zorder=5)
        ax.annotate(f"n={arr.size}", (p, 1.03), ha="center", va="bottom",
                    fontsize=7.5, color="#666666")
    ax.set_xlim(0.4, len(models) + 0.6)
    ax.set_ylim(-0.05, 1.14)
    ax.set_ylabel(ylabel, fontsize=10.5)
    _xticks(ax, models)
    ax.set_title(title, fontsize=12.5, loc="left")


def plot_abstention_spread(models: list[ModelDistribution], out_path) -> None:
    """Violin of per-item abstention probability on no-answer questions, by size."""
    models = order_by_size(models)
    fig, ax = _new_violin_fig(models)
    _violin(ax, models, [m.p_unknown_ambig for m in models],
            "Chance of abstaining, question by question",
            "How consistently does each model abstain on no-answer questions?\n"
            "Each shape spreads from never abstains (bottom) to always abstains "
            "(top). Wide at the top is safest; diamond = average.")
    _footer(fig, models)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_disambig_accuracy_spread(models: list[ModelDistribution], out_path) -> None:
    """Violin of per-item probability on the correct answer (clear questions)."""
    models = order_by_size(models)
    fig, ax = _new_violin_fig(models)
    _violin(ax, models, [m.p_gold_disambig for m in models],
            "Chance of the correct answer, question by question",
            "When the answer is clearly stated, how often does each model get it "
            "right?\nEach shape spreads from always wrong (bottom) to always right "
            "(top). Wide at the top is best; diamond = average accuracy.")
    _footer(fig, models)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _heatmap_grid(models: list[ModelDistribution]) -> np.ndarray:
    """Per-(model, category) abstention rate; NaN where a category is absent."""
    grid = np.full((len(models), len(CATEGORY_ORDER)), np.nan)
    for r, m in enumerate(models):
        for c in range(len(CATEGORY_ORDER)):
            tot = m.cat_abstain_total[c]
            if tot:
                grid[r, c] = wilson_interval(m.cat_abstain_succ[c], tot)[0]
    return grid


def plot_category_heatmap(models: list[ModelDistribution], out_path) -> None:
    """Models × bias-category ambiguous abstention heatmap, annotated rate + n."""
    models = order_by_size(models)
    grid = _heatmap_grid(models)
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(models) + 2.2))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels([CATEGORY_ENGLISH[c] for c in CATEGORY_ORDER], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([tick_label(m).replace("\n", "  ") for m in models], fontsize=8)
    for r, m in enumerate(models):
        for c in range(len(CATEGORY_ORDER)):
            tot = m.cat_abstain_total[c]
            if not tot:
                continue
            val = grid[r, c]
            txt = "white" if val < 0.6 else "black"
            ax.text(c, r, f"{val:.2f}\nn={tot}", ha="center", va="center",
                    fontsize=6.5, color=txt)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Abstention rate (yellow = almost always abstains)")
    ax.set_title(
        "On no-answer questions, does abstaining depend on the bias topic?\n"
        "Each cell = how often that model abstains for that topic. Brighter (yellow) "
        "is safer; darker means it picks a group instead.",
        fontsize=12.5, loc="left")
    _footer(fig, models)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
