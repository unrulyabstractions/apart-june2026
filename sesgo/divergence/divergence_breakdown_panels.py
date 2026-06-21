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
from .divergence_plot_styles import save_fig, titled


def plot_breakdown(metric: str, axis: str, groups, model, out_path, vmax: float):
    """Per-group mean of a metric over one axis, with bootstrap CIs + per-item dots."""
    keys = list(groups.keys())
    means = [float(np.mean(groups[k])) if groups[k] else 0.0 for k in keys]
    ns = [len(groups[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(max(6.5, 1.7 * len(keys) + 2.5), 5.0),
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
        ax.text(i, hi + vmax * 0.02, f"{means[i]:.3f}", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
        rng = np.random.default_rng(i)
        jx = i + (rng.random(ns[i]) - 0.5) * 0.30
        ax.scatter(jx, groups[k], s=30, color="#222222", alpha=0.5,
                   edgecolor="white", lw=0.5, zorder=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k}\n(n={m})" for k, m in zip(keys, ns)], fontsize=9.5)
    ax.set_ylim(0, vmax)
    ax.set_ylabel(f"mean {metric}  (dots = per-item)")
    titled(ax, f"SESGO: {metric} by {axis}  ({model})",
           "bars = group mean, whiskers = bootstrap 95% CI; dots = items; n per group")
    return save_fig(fig, out_path)
