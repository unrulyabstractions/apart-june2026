"""Two bar figures: which rewording flips the answer, and how accurate it is.

format_sensitivity: for each kind of harmless rewording (the option labels, and
the order the answer choices are listed), how often does changing JUST that one
thing flip the model's answer. Lower bars are better.

accuracy: how often the model's answer matches the correct answer, split by
whether the question is ambiguous or clear, and by whether 'unknown' was offered.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.common.math import bootstrap_ci, wilson_err, wilson_interval
from stability_metrics_helpers import AccuracyCount, FlipRate
from stability_plot_style import (
    AMBIG_BLUE, COND_COLOR, COND_LABEL, DISAMBIG_ORANGE, ZONE_GREY,
    save_figure, titled,
)

# Plain-English names for the two harmless rewording axes the grid varies.
_AXIS_LABEL: dict[str, str] = {
    "label_style": "Option labels\n(A/B/C vs 1/2/3)",
    "permutation": "Order of the\nanswer choices",
}


def plot_format_sensitivity(
    flips_by_cond: dict[str, list[FlipRate]], model: str, out_path: Path,
) -> Path:
    """Grouped bars: how often rewording one thing flips the answer, +-95% CI."""
    axes_names = [f.axis for f in next(iter(flips_by_cond.values()))]
    fig, ax = plt.subplots(figsize=(8.4, 5.4), layout="constrained")
    width, x = 0.36, np.arange(len(axes_names))
    for i, cond in enumerate(("ambig", "disambig")):
        rates, errs, ns = [], [[], []], []
        for fr in flips_by_cond[cond]:
            _, lo, hi = bootstrap_ci(fr.flips) if fr.flips else (fr.rate, fr.rate, fr.rate)
            r = 0.0 if np.isnan(fr.rate) else fr.rate
            rates.append(r)
            errs[0].append(max(0.0, r - (lo if not np.isnan(lo) else r)))
            errs[1].append(max(0.0, (hi if not np.isnan(hi) else r) - r))
            ns.append(fr.n_groups)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, rates, width, yerr=errs, capsize=4,
                      color=COND_COLOR[cond], label=COND_LABEL[cond],
                      error_kw={"elinewidth": 1.4, "ecolor": "#333333"})
        for bar, r, n, above in zip(bars, rates, ns, errs[1]):
            ax.text(bar.get_x() + bar.get_width() / 2, r + above + 0.012,
                    f"{r:.0%}\nn={n}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_AXIS_LABEL.get(a, a) for a in axes_names], fontsize=10.5)
    ax.set_ylim(0, max(0.05, ax.get_ylim()[1] * 1.14))
    ax.set_ylabel("How often this rewording alone flips the answer")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    titled(ax, f"Which harmless rewording most changes the answer?  ({model})",
           "taller bars = the model's answer is more easily swayed by wording "
           "that should not matter")
    return save_figure(fig, out_path)


def _is_ambig(label: str) -> bool:
    """Whether an accuracy label is the AMBIGUOUS condition (exact, not substring)."""
    return label.split()[-1] == "ambig"


def _accuracy_text(c: AccuracyCount, h: float) -> str:
    """Per-bar annotation: percent + count, or an N/A note when nothing scorable."""
    if c.total == 0:
        return "not scored\n('unknown' was\nnot offered)"
    return f"{h:.0%}\n{c.correct}/{c.total}"


def plot_accuracy(counts: list[AccuracyCount], model: str, out_path: Path) -> Path:
    """Per-condition accuracy bars, both readouts, with Wilson 95% CIs and n."""
    fig, ax = plt.subplots(figsize=(9.2, 5.6), layout="constrained")
    xs = list(range(len(counts)))
    for i, c in enumerate(counts):
        p, _, _ = wilson_interval(c.correct, c.total)
        below, above = wilson_err(c.correct, c.total)
        # Guard against tiny negative offsets from float rounding at the [0,1] clamp.
        below, above = max(0.0, below), max(0.0, above)
        h = 0.0 if np.isnan(p) else p
        color = AMBIG_BLUE if _is_ambig(c.label) else DISAMBIG_ORANGE
        hatch = "" if c.label.startswith("3-opt") else "///"
        ax.bar(i, h, yerr=[[below], [above]], capsize=5, width=0.6, color=color,
               hatch=hatch, edgecolor="white", linewidth=0,
               error_kw={"elinewidth": 1.5, "ecolor": "#333333"})
        y = (h + above + 0.03) if c.total else 0.04
        ax.text(i, y, _accuracy_text(c, h), ha="center", va="bottom", fontsize=9)
    ax.axhline(1 / 3, color=ZONE_GREY, linestyle=":", linewidth=1.4)
    # Park the reference label over the empty "not scored" bar so it never sits on
    # top of a filled bar (the line itself crosses every bar; the text should not).
    empty = next((i for i, c in enumerate(counts) if c.total == 0), len(counts) - 1)
    ax.text(empty, 1 / 3 + 0.02, "random guessing (1 in 3)", fontsize=8,
            color="#777777", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_xticks(xs)
    ax.set_xticklabels([_xtick(c.label) for c in counts], fontsize=9.5)
    handles = [
        Patch(facecolor=AMBIG_BLUE, label="Ambiguous question (no clear answer)"),
        Patch(facecolor=DISAMBIG_ORANGE, label="Clear question (answer is stated)"),
        Patch(facecolor="#FFFFFF", edgecolor="#333333",
              label="Three options offered (with 'unknown')"),
        Patch(facecolor="#FFFFFF", edgecolor="#333333", hatch="///",
              label="Forced two-way choice (no 'unknown')"),
    ]
    # Legend UNDER the panel so it never collides with bar-top labels, even when
    # every bar saturates at 100% (no room for an in-axes legend then).
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.95, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share of answers that were correct  (Wilson 95% CI)")
    titled(ax, f"How often is the model's answer correct?  ({model})",
           "taller bars = more correct; on ambiguous questions the correct answer "
           "is 'unknown'")
    return save_figure(fig, out_path)


def _xtick(label: str) -> str:
    """Plain two-line x-tick from a '<readout> <condition>' accuracy label."""
    readout = "Three options" if label.startswith("3-opt") else "Two-way choice"
    cond = "Ambiguous" if _is_ambig(label) else "Clear"
    return f"{cond}\n{readout}"
