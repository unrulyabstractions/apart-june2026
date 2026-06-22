"""Per-group breakdowns of a default metric over one provenance axis.

Bars are the group mean with a bootstrap 95% CI whisker; per-item dots show the
honest spread; the group n is annotated under every tick. Used for both the
default-uncertainty (entropy) and default-deviation (JS) metrics across the
bias_category / question_polarity / language axes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.common.math import bootstrap_ci
from sesgo.common.plain_language_labels import (
    CATEGORY_LABEL,
    CATEGORY_ORDER,
    POLARITY_LABEL,
    POLARITY_ORDER,
)
from .divergence_plot_styles import AXIS_LABEL, save_fig, titled

# Plain-language name for each per-question divergence metric.
_METRIC_TITLE = {
    "default-uncertainty": "Indecision",
    "default-deviation": "Drift from abstaining",
}
_METRIC_YLABEL = {
    "default-uncertainty": "Answer entropy",
    "default-deviation": "Drift from abstaining",
}
# Raw provenance codes -> plain group labels (categories, wording); else verbatim.
_GROUP_LABEL = {**CATEGORY_LABEL, **POLARITY_LABEL}
# Stable plain-language display order per axis (so 'Neutral' precedes 'Negative').
_GROUP_ORDER = (*CATEGORY_ORDER, *POLARITY_ORDER)


def _ordered_keys(groups) -> list:
    """Group keys in the canonical plain-language order, unknown codes appended."""
    known = [k for k in _GROUP_ORDER if k in groups]
    return known + [k for k in groups if k not in known]


def plot_breakdown(metric: str, axis: str, groups, model, out_path, vmax: float):
    """Per-group mean of a divergence metric over one axis, with CIs + per-item dots."""
    keys = _ordered_keys(groups)
    means = [float(np.mean(groups[k])) if groups[k] else 0.0 for k in keys]
    ns = [len(groups[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(max(6.8, 1.9 * len(keys) + 2.5), 5.0),
                           layout="constrained")
    palette = sns.color_palette("colorblind", max(1, len(keys)))
    for i, (k, c) in enumerate(zip(keys, palette)):
        _, lo, hi = bootstrap_ci(list(groups[k]))
        lo = means[i] if np.isnan(lo) else lo
        hi = means[i] if np.isnan(hi) else hi
        ax.bar(i, means[i], color=c, edgecolor="white", width=0.62, zorder=1)
        ax.errorbar(i, means[i], yerr=[[max(0, means[i] - lo)], [max(0, hi - means[i])]],
                    fmt="none", ecolor="#222222", elinewidth=1.6, capsize=5,
                    capthick=1.6, zorder=4)
        ax.text(i, hi + vmax * 0.02, f"{means[i]:.2f}", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
        rng = np.random.default_rng(i)
        jx = i + (rng.random(ns[i]) - 0.5) * 0.30
        ax.scatter(jx, groups[k], s=30, color="#222222", alpha=0.5,
                   edgecolor="white", lw=0.5, zorder=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{_GROUP_LABEL.get(k, k)}\n(n={m})" for k, m in zip(keys, ns)],
                       fontsize=9.5)
    ax.set_ylim(0, vmax)
    ax.set_ylabel(_METRIC_YLABEL.get(metric, metric))
    axis_name = AXIS_LABEL.get(axis, axis)
    titled(ax, f"{_METRIC_TITLE.get(metric, metric)} by {axis_name}")
    return save_fig(fig, out_path)
