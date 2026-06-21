"""Single-model SESGO baseline ROLE-PROBABILITY figure (direct-answer leanings).

One grouped-bar panel: the mean direct-answer probability the model places on each
group per bias category - stereotyped group / other group / abstains ('unknown') -
with SEM whiskers and a per-category n annotation. A tall bar for one group beside
a short bar for the other is a sign of bias. Companion of the accuracy figure in
``baseline_sample_plots.py``.

All rendered text uses the shared plain-language vocabulary (no pipeline jargon).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.common.math import sem
from src.datasets.sesgo_eval import SesgoSample

from sesgo.baseline.baseline_plot_palette import ROLE_COLORS, ROLE_NAMES, ordered_categories
from sesgo.common import CATEGORY_LABEL, ROLE_LABEL


def _role_panel(ax, samples_by_cat: dict[str, list[SesgoSample]], cats: list[str]) -> None:
    """Grouped mean role-prob bars (stereotyped / other / abstains) per category."""
    width = 0.8 / len(ROLE_NAMES)
    for ri, role in enumerate(ROLE_NAMES):
        offset = (ri - (len(ROLE_NAMES) - 1) / 2) * width
        for ci, cat in enumerate(cats):
            vals = [s.non_thinking.prob[ri] for s in samples_by_cat[cat]
                    if s.non_thinking is not None]
            if not vals:
                continue
            m = float(np.mean(vals))
            ax.bar(ci + offset, m, width, color=ROLE_COLORS[role],
                   label=ROLE_LABEL[role] if ci == 0 else None, zorder=3)
            ax.errorbar(ci + offset, m, yerr=sem(vals), fmt="none", ecolor="#222222",
                        elinewidth=0.9, capsize=2.0, zorder=4)
            if ri == 0:  # annotate per-category n once (same n across roles)
                ax.text(ci, 1.02, f"n={len(vals)}", ha="center", va="bottom", fontsize=8.5)


def plot_role_prob(samples: list[SesgoSample], cats: list[str], model: str, out_path):
    """Mean direct-answer probability on each group [stereotyped / other / abstains]."""
    cats = ordered_categories([c for c in cats if any(s.bias_category == c for s in samples)])
    by_cat = {cat: [s for s in samples if s.bias_category == cat] for cat in cats}
    fig, ax = plt.subplots(figsize=(max(7.5, 1.7 * len(cats) + 3), 5.2))
    _role_panel(ax, by_cat, cats)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([CATEGORY_LABEL.get(c, c) for c in cats], fontsize=11)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Average probability the model\nputs on each answer", fontsize=10.5)
    ax.set_xlabel("Bias category", fontsize=11)
    # Legend BELOW the axes so it never collides with the per-category n labels
    # that sit just above the bars near the top of the panel.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Answer the model leaned toward",
               ncol=len(ROLE_NAMES), loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False, fontsize=10, title_fontsize=10)
    n = sum(1 for s in samples if s.non_thinking is not None)
    fig.suptitle(
        f"Which answer {model} leans toward when answering directly  (n={n} scored items)\n"
        "Bars show how much probability the model puts on each group.  A tall orange bar next\n"
        "to a short light-blue bar (or the reverse) means it favours one group - a sign of bias.",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
