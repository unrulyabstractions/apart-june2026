"""Render the SESGO baseline size-sweep: accuracy vs params, by family, stacked.

One figure, three stacked panels (3-option / 2-option / greedy-thinking). Each
panel plots accuracy vs model size on a log-x axis. Within a panel we draw three
metric SERIES — ambiguous abstention, disambiguated TARGET-gold, disambiguated
OTHER-gold — distinguished by line STYLE/marker, and one LINE PER FAMILY,
distinguished by COLOR. Every point carries a Wilson 95% CI whisker and an ``n=``
annotation, so the figure stays honest at the tiny-n sizes a partial fleet emits.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.common.math import wilson_err

from sesgo.baseline.cross_model_aggregation import CONDITIONS, CrossModelPoint

# Okabe–Ito colorblind-safe hues, one per family (stable across panels/series).
_FAMILY_COLORS = {
    "Qwen": "#0072B2", "Llama": "#D55E00", "Gemma": "#009E73", "Mistral": "#CC79A7",
}
# The three metric series share a family color but differ by marker + linestyle.
_SERIES_STYLE = {
    "ambig": ("o", "-", "abstention (ambig, gold=unknown)"),
    "disambig-target": ("^", "--", "disambig acc · gold=TARGET"),
    "disambig-other": ("s", ":", "disambig acc · gold=OTHER"),
}
# Human-readable panel titles for each generation condition.
_COND_TITLES = {
    "non_thinking": "3-option (non_thinking, teacher-forced)",
    "non_thinking_2opt": "2-option (non_thinking_2opt, forced choice)",
    "greedy_thinking": "greedy_thinking (single reasoning decode)",
}


def _by_family_series(
    points: list[CrossModelPoint],
) -> dict[tuple[str, str], list[CrossModelPoint]]:
    """Group points by (family, slice_label), each size-sorted for a clean line."""
    groups: dict[tuple[str, str], list[CrossModelPoint]] = defaultdict(list)
    for p in points:
        groups[(p.family, p.slice_label)].append(p)
    return {k: sorted(v, key=lambda p: p.params_b) for k, v in groups.items()}


def _draw_series(ax, fam: str, slice_label: str, pts: list[CrossModelPoint]) -> None:
    """Plot one family's one metric series with Wilson CI bars and n labels."""
    marker, ls, _ = _SERIES_STYLE[slice_label]
    color = _FAMILY_COLORS.get(fam, "#555555")
    xs = [p.params_b for p in pts]
    ys = [p.accuracy for p in pts]
    # wilson_err gives (below, above) offsets; clamp the tiny negative floats that
    # arise at p == 0/1 so matplotlib's yerr (which forbids negatives) is happy.
    err = np.array([[max(0.0, b), max(0.0, a)]
                    for b, a in (wilson_err(p.successes, p.total) for p in pts)]).T
    ax.errorbar(
        xs, ys, yerr=err, color=color, marker=marker, linestyle=ls, linewidth=1.6,
        markersize=6, capsize=3, elinewidth=1.0, alpha=0.9,
    )
    for p in pts:  # annotate n above each point so tiny-n bars stay honest
        ax.annotate(f"n={p.total}", (p.params_b, p.accuracy), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6.5, color=color, alpha=0.85)


def _panel(ax, cond: str, points: list[CrossModelPoint]) -> None:
    """Draw one condition's panel: every (family, series) line on a log-x axis."""
    for (fam, slice_label), pts in _by_family_series(points).items():
        _draw_series(ax, fam, slice_label, pts)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel("accuracy")
    ax.set_title(_COND_TITLES.get(cond, cond), fontsize=11, loc="left")
    ax.grid(True, which="both", axis="x", ls=":", alpha=0.4)
    # 3-way chance baseline (1/3); abstention/3-opt sit against this. Light dashed
    # so it reads as a reference without competing with the data lines.
    ax.axhline(1.0 / 3, ls="--", lw=1.0, color="#888888", alpha=0.7, zorder=0)
    ax.text(0.005, 1.0 / 3, " chance ⅓", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=7, color="#888888")


def _legend(fig, families: list[str]) -> None:
    """Two-part legend: family colors and metric-series marker/style keys."""
    fam_handles = [plt.Line2D([], [], color=_FAMILY_COLORS.get(f, "#555555"),
                              marker="o", linestyle="-", label=f) for f in families]
    series_handles = [plt.Line2D([], [], color="#444444", marker=m, linestyle=ls, label=lbl)
                      for m, ls, lbl in _SERIES_STYLE.values()]
    fig.legend(handles=fam_handles, title="family", loc="upper left",
               bbox_to_anchor=(1.005, 0.97), frameon=False, fontsize=9, title_fontsize=9)
    fig.legend(handles=series_handles, title="metric", loc="upper left",
               bbox_to_anchor=(1.005, 0.62), frameon=False, fontsize=9, title_fontsize=9)


def plot_size_sweep(points: list[CrossModelPoint], out_path, n_models: int) -> None:
    """Stacked 3-panel size-sweep figure (one panel per generation condition)."""
    conds = [c for c, _ in CONDITIONS]
    fig, axes = plt.subplots(len(conds), 1, figsize=(9.5, 11), sharex=True)
    for ax, cond in zip(np.atleast_1d(axes), conds):
        _panel(ax, cond, [p for p in points if p.condition == cond])
    np.atleast_1d(axes)[-1].set_xlabel("model size (billion params, log scale)")
    families = sorted({p.family for p in points})
    _legend(fig, families)
    fig.suptitle(
        f"SESGO baseline size sweep · accuracy vs model size  ({n_models} model(s))",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.82, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
