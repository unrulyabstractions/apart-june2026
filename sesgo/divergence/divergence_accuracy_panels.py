"""Per-condition accuracy across the four readouts, with Wilson CIs and n.

The NEW divergence data model scores each prompt at four readouts in escalating
effort — 3-option non-thinking, 2-option forced choice (no UNKNOWN), greedy
(deterministic) thinking, and sampled thinking — under two context conditions
(ambiguous: gold UNKNOWN == abstain; disambiguated: gold == ground-truth role).
This module renders that as ONE figure with ambiguous and disambiguated context
conditions STACKED as subfigures, each a grouped bar of the four readouts'
accuracy with Wilson score CIs (honest at small n) and the n behind every bar.

The 2-option readout has no UNKNOWN, so its accuracy is undefined on ambiguous
items (nothing to abstain to); that bar is drawn hollow + "N/A" so the gap is
explicit rather than hidden.
"""

from __future__ import annotations

import numpy as np

from src.common.math import wilson_err
from .divergence_plot_styles import (
    READOUT_COLORS,
    READOUT_LABELS,
    READOUTS,
    REF,
    save_fig,
)
import matplotlib.pyplot as plt

# Per (condition, readout) we carry (successes, n); a None n marks "undefined".
_CORRECT_FNS = {
    "non_thinking": lambda s: s.correct_non_thinking,
    "non_thinking_2opt": lambda s: s.correct_2opt,
    "greedy_thinking": lambda s: s.correct_greedy_thinking,
    "thinking": lambda s: s.correct_thinking,
}


def _counts(samples, readout: str) -> tuple[int, int]:
    """(#correct, #scorable) for a readout; correct_2opt==None items are dropped."""
    vals = [_CORRECT_FNS[readout](s) for s in samples]
    scored = [v for v in vals if v is not None]
    return sum(bool(v) for v in scored), len(scored)


def _draw_condition(ax, samples, condition: str) -> None:
    """One subfigure axes: grouped accuracy bars over the four readouts + CIs."""
    xs = np.arange(len(READOUTS))
    for x, readout in zip(xs, READOUTS):
        k, n = _counts(samples, readout)
        color = READOUT_COLORS[readout]
        if n == 0:  # undefined here (forced two-way on ambiguous): hollow placeholder
            ax.bar(x, 1.0, color="none", edgecolor=color, hatch="///",
                   width=0.7, lw=1.4, alpha=0.6)
            ax.text(x, 0.5, "N/A", ha="center", va="center", fontsize=11,
                    color=REF, style="italic")
            continue
        p = k / n
        lo, hi = wilson_err(k, n)
        ax.bar(x, p, color=color, edgecolor="white", width=0.7, zorder=2)
        ax.errorbar(x, p, yerr=[[lo], [hi]], fmt="none", ecolor="#222222",
                    elinewidth=1.6, capsize=5, capthick=1.6, zorder=4)
        ax.text(x, min(p + hi + 0.03, 0.99), f"{p:.0%}\n{k}/{n}", ha="center",
                va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([READOUT_LABELS[r] for r in READOUTS], fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Share correct")
    ax.axhline(1.0, color=REF, ls=":", lw=1.0, alpha=0.6)
    n_total = max((_counts(samples, r)[1] for r in READOUTS), default=0)
    ax.set_title(f"{condition}  (n={n_total})", fontsize=11.5,
                 fontweight="bold", loc="left")


def plot_accuracy_by_readout(by_condition, model, out_path):
    """Stacked subfigures: ambiguous (top) vs clear (bottom) accuracy."""
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.0), layout="constrained",
                             sharex=True)
    _draw_condition(axes[0], by_condition.get("ambig", []), "Ambiguous")
    _draw_condition(axes[1], by_condition.get("disambig", []), "Clear")
    return save_fig(fig, out_path)
