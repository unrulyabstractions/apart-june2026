"""Plain-language THINKING vs WITHOUT-THINKING views on ambiguous questions.

Two figures built from the EXISTING divergence readouts (no new sampling), in the
same spirit as the full-data hero plot: on ambiguous SESGO questions (where the
only correct answer is 'unknown'), how does the model answer when it replies
DIRECTLY versus when it REASONS FIRST?

  * Outcome mix (``*_outcome_mix.png``): a stacked bar per (way-of-asking x social
    category) showing the share of answers that name the stereotyped group, name
    the other group, or correctly abstain. A tall blue 'Abstains' band is good.
  * Abstention rate (``*_abstention.png``): grouped bars of the abstention rate
    split by neutral vs negative wording, per social category, with Wilson 95% CIs
    and n. Taller = more correct abstention; reasoning should push bars UP.

Without-thinking abstention is the model's direct 'unknown' mass; with-thinking is
the fraction of free-form reasoning tries that abstained. Whiskers are Wilson 95%
CIs on the share of questions whose answer abstains; n is questions per cell.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.common.math import wilson_err
from src.datasets.sesgo_eval import SesgoSample
from sesgo.common.plain_language_labels import CATEGORY_LABEL, CATEGORY_ORDER
from .divergence_plot_styles import ROLE_COLORS, ROLES, save_fig
from .thinking_contrast_metrics import (
    WAYS,
    WORDINGS,
    abstains,
    mean_outcome_mix,
)

_OUTCOME_LABEL = {
    "target": "Stereotyped group",
    "other": "Other group",
    "unknown": "Abstains",
}


def _cats(samples: list[SesgoSample]) -> list[str]:
    """Social categories present, in stable display order."""
    return [c for c in CATEGORY_ORDER if any(s.bias_category == c for s in samples)]


def plot_outcome_mix(ambig: list[SesgoSample], model: str, out_path):
    """Stacked answer-mix bars per (way-of-asking x category): direct vs reasoned."""
    cats = _cats(ambig)
    fig, axes = plt.subplots(1, len(WAYS), figsize=(4.4 * len(WAYS), 5.4),
                             sharey=True, squeeze=False)
    for ax, (attr, way_title) in zip(axes[0], WAYS):
        xs = range(len(cats))
        bottoms = [0.0] * len(cats)
        for role in ROLES:
            heights = [mean_outcome_mix([s for s in ambig if s.bias_category == c],
                                        attr)[ROLES.index(role)] for c in cats]
            ax.bar(xs, heights, bottom=bottoms, width=0.66, color=ROLE_COLORS[role],
                   edgecolor="white", label=_OUTCOME_LABEL[role], zorder=2)
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        for x, c in zip(xs, cats):
            n = sum(1 for s in ambig if s.bias_category == c)
            ax.text(x, 1.02, f"n={n}", ha="center", va="bottom", fontsize=7.5,
                    color="#555555")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([CATEGORY_LABEL[c] for c in cats], fontsize=9.5)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(way_title, fontsize=12, fontweight="bold")
    axes[0][0].set_ylabel("Share of answers")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return save_fig(fig, out_path)


def _abstention_panel(ax, ambig: list[SesgoSample], attr: str, cat: str) -> None:
    """Grouped abstention-rate bars per wording for one (way, category) cell."""
    width = 0.8 / len(WORDINGS)
    for wi, (pol, label, color) in enumerate(WORDINGS):
        offset = (wi - (len(WORDINGS) - 1) / 2) * width
        cell = [s for s in ambig
                if s.bias_category == cat and s.question_polarity == pol]
        flags = [abstains(s, attr) for s in cell]
        usable = [f for f in flags if f is not None]
        if not usable:
            continue
        succ, total = sum(usable), len(usable)
        rate = succ / total
        below, above = (max(0.0, e) for e in wilson_err(succ, total))
        ax.bar(offset, rate, width, color=color, zorder=3,
               label=label)
        ax.errorbar(offset, rate, yerr=[[below], [above]], fmt="none",
                    ecolor="#222222", elinewidth=0.9, capsize=3, zorder=4)
        ax.text(offset, min(rate + above + 0.02, 1.1), f"{rate:.0%}", ha="center",
                va="bottom", fontsize=8, fontweight="bold")
        ax.text(offset, 0.02, f"n={total}", ha="center", va="bottom", fontsize=6.5,
                color="#555555", rotation=90)
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


def plot_abstention_contrast(ambig: list[SesgoSample], model: str, out_path):
    """Faceted abstention rate: way-of-asking (rows) x social category (cols)."""
    cats = _cats(ambig)
    fig, axes = plt.subplots(len(WAYS), len(cats),
                             figsize=(2.7 * len(cats) + 1.5, 4.4 * len(WAYS)),
                             sharey=True, squeeze=False)
    for ri, (attr, way_title) in enumerate(WAYS):
        for ci, cat in enumerate(cats):
            ax = axes[ri][ci]
            _abstention_panel(ax, ambig, attr, cat)
            if ri == 0:
                ax.set_title(CATEGORY_LABEL[cat], fontsize=12, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{way_title}\n\nAbstention rate", fontsize=9.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    return save_fig(fig, out_path)
