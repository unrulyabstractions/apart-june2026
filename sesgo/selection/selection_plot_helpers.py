"""Publication-quality SELECTION plots: stacked readouts/conditions, CIs + n.

Every figure is ONE file. The comparisons the study cares about stack as SUBFIGURE
rows so they read together: NON-THINKING vs THINKING (vs greedy-thinking) stack
vertically; 2-OPTION vs 3-OPTION stack vertically per category. Each bar carries a
Wilson 95% CI and its sample size, and the SELECTED scaffold keeps a gold highlight
band + star. Context is split honestly: ambiguous bars read as ABSTENTION (gold =
UNKNOWN), disambiguated bars read as ACCURACY (gold = the ground-truth role).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.math import wilson_err, wilson_interval
from selection_metrics_helpers import LEVEL_TITLE, ScaffoldCount

# House palette: one hue per readout level (colorblind-safe Okabe-Ito).
_LEVEL_COLOR = {
    "non_thinking": "#0072B2",       # blue   – teacher-forced 3-option
    "non_thinking_2opt": "#009E73",  # green  – forced-choice 2-option
    "greedy_thinking": "#56B4E9",    # sky    – greedy reasoning baseline
    "thinking": "#D55E00",           # red    – sampled reasoning
}
_SELECT_BG = "#ffe9a8"   # warm gold band behind the SELECTED scaffold group
_SELECT_EDGE = "#e0a800"
_CHANCE = {3: 1 / 3, 2: 1 / 2}  # random-guess accuracy reference per option count


def _wrap(text: str, width: int = 13) -> str:
    """Soft-wrap a snake_case scaffold id onto multiple lines for a readable tick."""
    return textwrap.fill(text.replace("_", "_ "), width=width).replace("_ ", "_")


def _ticklabels(scaffolds: list[str], selected: str | None) -> list[str]:
    """Wrapped x-tick labels; the SELECTED scaffold gets a [SELECTED] caption."""
    return [
        f"{_wrap(sc)}\n[SELECTED]" if sc == selected else _wrap(sc) for sc in scaffolds
    ]


def _highlight(ax, x: np.ndarray, scaffolds: list[str], selected: str | None) -> None:
    """Gold band + star for the SELECTED scaffold group (path marker, font-safe)."""
    if selected is None or selected not in scaffolds:
        return
    i = scaffolds.index(selected)
    ax.axvspan(x[i] - 0.5, x[i] + 0.5, color=_SELECT_BG, alpha=0.55, zorder=0)
    for edge in (x[i] - 0.5, x[i] + 0.5):
        ax.axvline(edge, color=_SELECT_EDGE, lw=1.0, ls=":", zorder=1)
    ax.scatter([x[i]], [1.12], marker="*", s=260, color=_SELECT_EDGE,
               edgecolor="#7a5c00", linewidth=0.6, zorder=6, clip_on=False)


def _bold_selected_ticks(ax, scaffolds: list[str], selected: str | None) -> None:
    """Bold + gold the SELECTED scaffold's x-tick label so it pops."""
    for lbl, sc in zip(ax.get_xticklabels(), scaffolds):
        if sc == selected:
            lbl.set_fontweight("bold")
            lbl.set_color("#8a6d00")


def _bars_with_ci(ax, x, counts: dict[str, ScaffoldCount], scaffolds, color):
    """Draw one bar per scaffold with Wilson 95% CI whiskers; annotate rate + n."""
    heights, errs = [], [[], []]
    for sc in scaffolds:
        c = counts.get(sc)
        p, _, _ = wilson_interval(c.correct, c.total) if c and c.total else (0.0, 0, 0)
        below, above = wilson_err(c.correct, c.total) if c and c.total else (0.0, 0.0)
        heights.append(0.0 if np.isnan(p) else p)
        errs[0].append(below)
        errs[1].append(above)
    bars = ax.bar(x, heights, width=0.62, yerr=errs, capsize=4, color=color, zorder=3,
                  error_kw={"elinewidth": 1.4, "ecolor": "#333333"})
    # Annotate each bar above its OWN upper whisker (errs[1][i]); a degenerate
    # (no-data) cell gets a faint "n/a" stub instead of a misleading 0% label.
    for bar, sc, h, top_err in zip(bars, scaffolds, heights, errs[1]):
        c = counts.get(sc)
        cx = bar.get_x() + bar.get_width() / 2
        if c is None or c.total == 0:
            ax.text(cx, 0.02, "n/a", ha="center", va="bottom", fontsize=7.5,
                    color="#9a9a9a")
        else:
            ax.text(cx, h + top_err + 0.012, f"{h:.0%}\n{c.correct}/{c.total}",
                    ha="center", va="bottom", fontsize=8, linespacing=1.0)
    return heights


def _panel(ax, counts, scaffolds, selected, level, ylabel, opt_count) -> None:
    """One scaffold-accuracy panel: highlighted bars + CI + chance line + ticks."""
    x = np.arange(len(scaffolds))
    _highlight(ax, x, scaffolds, selected)
    _bars_with_ci(ax, x, counts, scaffolds, _LEVEL_COLOR[level])
    chance = _CHANCE.get(opt_count)
    if chance is not None:
        ax.axhline(chance, color="#888888", ls=":", lw=1.2, zorder=2)
        ax.text(len(scaffolds) - 0.5, chance + 0.015, f"chance ({opt_count}-opt)",
                fontsize=7.5, color="#777777", ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(_ticklabels(scaffolds, selected), fontsize=8.5)
    _bold_selected_ticks(ax, scaffolds, selected)
    ax.set_xlim(-0.6, len(scaffolds) - 0.4)
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylabel(ylabel, fontsize=9.5)
    # Panel label as an in-axes badge (top-left) so it never collides with the
    # figure suptitle the way a matplotlib axes-title would when rows stack tight.
    ax.text(0.012, 0.965, LEVEL_TITLE[level], transform=ax.transAxes, ha="left",
            va="top", fontsize=10.5, fontweight="bold", zorder=7,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cccccc",
                  "alpha": 0.9})


def _suptitle(fig, title: str, takeaway: str) -> None:
    """Bold figure title over an italic plain-language takeaway, above the panels.

    constrained_layout reserves room for both because they are placed via the
    layout engine's title/text slots (suptitle) rather than overlapping the axes.
    """
    fig.suptitle(f"{title}\n", fontsize=13, fontweight="bold")
    fig.text(0.5, 1.0, takeaway, ha="center", va="bottom", fontsize=9.5,
             color="#444444", style="italic", transform=fig.transFigure)


def _save(fig, out_path: Path) -> Path:
    """Persist publication-clean: tight bbox, 150 dpi, then close."""
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
