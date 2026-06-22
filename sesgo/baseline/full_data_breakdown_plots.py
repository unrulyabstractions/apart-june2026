"""Faceted ambiguous-abstention breakdown for the full-data SESGO study.

ONE figure that answers the core scaffold question in plain sight: on AMBIGUOUS
questions (where the only correct answer is 'unknown'), how often does the model
abstain -- broken down by whether it thought first, which debiasing scaffold was
used, the question's wording, and the social-group category. Taller bar = more
correct abstention; a good scaffold pushes bars UP versus 'No scaffold'.

Rows    = without thinking vs with thinking (the two readouts present in full_data).
Columns = bias category (Classism / Racism / Xenophobia / Gender).
Within  = grouped bars per scaffold, split by neutral vs negative wording.
Wilson 95% CI whisker + n on every bar. No pipeline jargon, all plain language.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.common.math import wilson_err
from src.datasets.sesgo_eval import SesgoSample

from sesgo.common.plain_language_labels import (
    CATEGORY_LABEL,
    CATEGORY_ORDER,
    SCAFFOLD_ORDER,
    SCAFFOLD_SHORT,
    normalize_scaffold,
)

# The two readouts full_data carries: direct answer vs reason-first. (attr, row title)
_READOUTS: tuple[tuple[str, str], ...] = (
    ("correct_non_thinking", "Without thinking"),
    ("correct_greedy_thinking", "With thinking"),
)
# Question wording -> bar colour (Okabe-Ito: neutral light-blue vs negative vermillion).
_WORDING: tuple[tuple[str, str, str], ...] = (
    ("nonneg", "Neutral", "#56B4E9"),
    ("neg", "Negative", "#D55E00"),
)
_CHANCE = 1 / 3  # three roles (target / other / unknown): random pick lands here.


def _abstention(samples: list[SesgoSample], attr: str, cat: str, scaffold, pol: str):
    """``(successes, n)`` abstention for one (category, scaffold, wording) cell.

    On ambiguous items gold IS unknown, so the correctness flag is the abstention
    indicator; unparsed (None) flags don't count toward n.
    """
    flags = [
        getattr(s, attr) for s in samples
        if s.bias_category == cat
        and normalize_scaffold(s.scaffold_id) == scaffold
        and s.question_polarity == pol
    ]
    usable = [f for f in flags if f is not None]
    return sum(bool(f) for f in usable), len(usable)


def _panel(ax, samples: list[SesgoSample], attr: str, cat: str) -> None:
    """One category panel: grouped abstention bars per scaffold, split by wording."""
    n_w = len(_WORDING)
    width = 0.8 / n_w
    for wi, (pol, label, color) in enumerate(_WORDING):
        offset = (wi - (n_w - 1) / 2) * width
        for xi, scaffold in enumerate(SCAFFOLD_ORDER):
            succ, total = _abstention(samples, attr, cat, scaffold, pol)
            if total == 0:
                continue
            rate = succ / total
            below, above = (max(0.0, e) for e in wilson_err(succ, total))
            ax.bar(xi + offset, rate, width, color=color, zorder=3,
                   label=label if xi == 0 else None)
            ax.errorbar(xi + offset, rate, yerr=[[below], [above]], fmt="none",
                        ecolor="#222222", elinewidth=0.8, capsize=2, zorder=4)
            # Bold percentage on top (the headline number); faint n just under it.
            ax.text(xi + offset, min(rate + above + 0.02, 1.08), f"{rate:.0%}",
                    ha="center", va="bottom", fontsize=7.2, fontweight="bold")
            ax.text(xi + offset, 0.02, f"n={total}", ha="center", va="bottom",
                    fontsize=6.0, color="#555555", rotation=90)
    ax.axhline(_CHANCE, ls="--", lw=0.9, color="#888888", alpha=0.8, zorder=1)
    ax.set_xticks(range(len(SCAFFOLD_ORDER)))
    ax.set_xticklabels([SCAFFOLD_SHORT[s] for s in SCAFFOLD_ORDER], fontsize=7.5)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


def plot_abstention_breakdown(ambig_samples: list[SesgoSample], out_path):
    """Faceted ambiguous-abstention figure: readout (rows) x bias category (cols)."""
    cats = [c for c in CATEGORY_ORDER if any(s.bias_category == c for s in ambig_samples)]
    nrow, ncol = len(_READOUTS), len(cats)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 5.0 * nrow),
                             sharey=True, squeeze=False)
    for ri, (attr, row_title) in enumerate(_READOUTS):
        for ci, cat in enumerate(cats):
            ax = axes[ri][ci]
            _panel(ax, ambig_samples, attr, cat)
            if ri == 0:
                ax.set_title(CATEGORY_LABEL.get(cat, cat), fontsize=12.5, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{row_title}\nabstention rate", fontsize=9.5)
            if ci == ncol - 1:
                ax.text(0.99, _CHANCE, " chance",
                        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                        fontsize=6.5, color="#888888")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1.0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
