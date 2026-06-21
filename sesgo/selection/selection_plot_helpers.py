"""Low-level bar renderer for the SELECTION study: plain labels, CIs, n, no jargon.

One reusable panel draws a row of bars over an arbitrary categorical x-axis (readout
levels, or bias categories) where the bar HEIGHT is a rate in [0, 1] (abstention or
accuracy). Every bar carries a Wilson 95% CI whisker and its sample size; an empty
cell gets a faint "no data" stub instead of a misleading 0%. An optional dotted line
marks the random-guessing rate. All rendered text is plain English (see
``sesgo/common/plain_language_labels.py``) — never pipeline tokens.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np

from src.common.math import wilson_err, wilson_interval
from sesgo.common.plain_language_labels import RANDOM_GUESS_LABEL

# Colourblind-safe Okabe-Ito hues, one per series the study draws.
SERIES_COLOR = {
    "non_thinking": "#0072B2",   # blue   – answers directly
    "thinking": "#D55E00",       # vermdn – free-form reasoning
    "category": "#009E73",       # green  – per-category bars
}


def _bar_label(rate: float, correct: int, total: int) -> str:
    """Headline percentage over its sample size, e.g. ``80%`` then ``28 of 35``."""
    return f"{rate:.0%}\n{correct} of {total}"


def _draw_bars(ax, x, labels, counts, colors) -> list[float]:
    """One bar per category with Wilson 95% CI whiskers; annotate rate + n above."""
    heights, lo_err, hi_err = [], [], []
    for label in labels:
        c = counts.get(label)
        p, _, _ = wilson_interval(c.correct, c.total) if c and c.total else (0.0, 0, 0)
        below, above = wilson_err(c.correct, c.total) if c and c.total else (0.0, 0.0)
        heights.append(0.0 if np.isnan(p) else p)
        lo_err.append(below)
        hi_err.append(above)
    # Narrow bars when only one or two categories so a panel never reads as a
    # single fat slab; wider when many so they still fill the row.
    width = 0.42 if len(labels) <= 2 else 0.6
    bars = ax.bar(
        x, heights, width=width, yerr=[lo_err, hi_err], capsize=5, color=colors,
        zorder=3, error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
    )
    for bar, label, h, top in zip(bars, labels, heights, hi_err):
        c = counts.get(label)
        cx = bar.get_x() + bar.get_width() / 2
        if c is None or c.total == 0:
            ax.text(cx, 0.03, "no data", ha="center", va="bottom", fontsize=8,
                    color="#9a9a9a", style="italic")
        else:
            ax.text(cx, h + top + 0.02, _bar_label(h, c.correct, c.total),
                    ha="center", va="bottom", fontsize=9.5, linespacing=1.05)
    return heights


def _random_guess_line(ax, chance: float, show_legend: bool) -> None:
    """Dotted random-guessing reference line + optional collision-free legend.

    When labelled, a legend (top-right) names the line so the caption can never end
    up behind a bar however tall the bars grow — unlike an inline text annotation.
    """
    line = ax.axhline(chance, color="#888888", ls=(0, (4, 3)), lw=1.3, zorder=2,
                      label=f"{RANDOM_GUESS_LABEL} ({chance:.0%})")
    if show_legend:
        ax.legend(handles=[line], loc="upper right", fontsize=8.5, frameon=True,
                  framealpha=0.9, borderpad=0.4, handlelength=1.8)


def panel(ax, labels, counts, colors, ylabel, chance,
          badge=None, show_xticklabels=True, show_legend=True) -> None:
    """Draw one plain bar panel: bars + CI + random-guessing line + clean ticks."""
    x = np.arange(len(labels))
    n = len(labels)
    if chance is not None:
        _random_guess_line(ax, chance, show_legend)
    _draw_bars(ax, x, labels, counts, colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels if show_xticklabels else [""] * n, fontsize=11)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10.5)
    if badge is not None:
        ax.text(0.012, 0.96, badge, transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold", zorder=7,
                bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": "#cccccc",
                      "alpha": 0.92})


def figure_titles(fig, title: str, how_to_read: str) -> None:
    """Bold plain-sentence title over a wrapped italic 'how to read this' line."""
    fig.suptitle(title, fontsize=13.5, fontweight="bold")
    fig.text(0.5, 1.0, fill(how_to_read, width=86), ha="center", va="bottom",
             fontsize=9.5, color="#444444", style="italic",
             transform=fig.transFigure, linespacing=1.25)


def save_figure(fig, out_path: Path) -> Path:
    """Persist publication-clean: tight bbox, 150 dpi, then close."""
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
