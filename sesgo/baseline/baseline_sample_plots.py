"""Publication-quality single-model SESGO baseline figures (accuracy + role prob).

Two stacked-subfigure builders, both honest at tiny n:

  * ``plot_accuracy``  : one ROW per readout condition (3-option / 2-option /
    greedy-thinking), grouped bars per bias_category showing the three accuracy
    slices (ambiguous abstention, disambiguated TARGET-gold, disambiguated
    OTHER-gold). The 2-option row has no abstention slice (no UNKNOWN), drawn as
    an explicit "n/a" marker. Wilson 95% CI whisker + n on every bar.
  * ``plot_role_prob`` : mean non-thinking role-prob [target/other/unknown] per
    bias_category with SEM whiskers and n annotations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.common.math import sem, wilson_err
from src.datasets.sesgo_eval import SesgoSample

from sesgo.baseline.baseline_accuracy_slices import SLICES, AccuracyCell
from sesgo.baseline.cross_model_aggregation import COND_TITLES, CONDITIONS

# Okabe–Ito hues: one per accuracy slice (shared with the role legend palette).
_SLICE_COLORS = {"ambig": "#0072B2", "disambig-target": "#E69F00", "disambig-other": "#56B4E9"}
_SLICE_LABELS = {
    "ambig": "abstention (ambig · gold=unknown)",
    "disambig-target": "disambig acc · gold=TARGET",
    "disambig-other": "disambig acc · gold=OTHER",
}
_ROLE_NAMES = ("target", "other", "unknown")
_ROLE_COLORS = {"target": "#E69F00", "other": "#56B4E9", "unknown": "#009E73"}
# Chance baseline differs by readout: 2-option is a 2-way forced choice (1/2),
# the 3-option and greedy-thinking readouts span 3 roles (1/3).
_CHANCE = {"non_thinking": 1 / 3, "non_thinking_2opt": 1 / 2, "greedy_thinking": 1 / 3}


def _bar_with_ci(ax, x: float, cell: AccuracyCell, color: str) -> None:
    """Draw one accuracy bar (Wilson CI + n), or an italic n/a tick if no data."""
    if cell.total == 0:
        ax.text(x, 0.06, "n/a", ha="center", va="bottom", fontsize=7.5,
                style="italic", color="#999999", rotation=90)
        return
    below, above = (max(0.0, e) for e in wilson_err(cell.successes, cell.total))
    # A flat cap at y=0 keeps a 0%-accuracy bar (zero height) visibly anchored so
    # its CI whisker and n label do not read as floating in space.
    ax.plot([x - 0.13, x + 0.13], [0, 0], color=color, lw=2.2, zorder=2)
    ax.bar(x, cell.accuracy, width=0.26, color=color, zorder=3)
    ax.errorbar(x, cell.accuracy, yerr=[[below], [above]], fmt="none",
                ecolor="#222222", elinewidth=1.0, capsize=2.5, zorder=4)
    ax.text(x, min(cell.accuracy + above + 0.02, 1.12),
            f"{cell.accuracy:.0%}\nn={cell.total}", ha="center", va="bottom", fontsize=7)


def _accuracy_panel(ax, cond: str, cells: list[AccuracyCell], cats: list[str]) -> None:
    """One readout row: grouped accuracy bars per category, all three slices."""
    lut = {(c.category, c.slice_label): c for c in cells}
    n_slices = len(SLICES)
    width = 0.26
    for ci, cat in enumerate(cats):
        for si, (slice_label, _, _) in enumerate(SLICES):
            x = ci + (si - (n_slices - 1) / 2) * width
            cell = lut.get((cat, slice_label))
            if cell is not None:
                _bar_with_ci(ax, x, cell, _SLICE_COLORS[slice_label])
    chance = _CHANCE.get(cond, 1 / 3)
    ax.axhline(chance, ls="--", lw=1.0, color="#888888", alpha=0.8, zorder=1)
    ax.text(0.004, chance, f" chance {chance:.0%}", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=7, color="#888888")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("accuracy")
    ax.set_title(COND_TITLES.get(cond, cond), fontsize=11, loc="left")
    ax.margins(x=0.04)


def plot_accuracy(cells: list[AccuracyCell], cats: list[str], model: str, n: int, out_path):
    """Stacked figure: one readout row each (3-opt / 2-opt / greedy-thinking)."""
    conds = [c for c, _ in CONDITIONS]
    fig, axes = plt.subplots(len(conds), 1, figsize=(max(8.5, 1.7 * len(cats) + 4), 11),
                             sharex=True)
    for ax, cond in zip(np.atleast_1d(axes), conds):
        _accuracy_panel(ax, cond, [c for c in cells if c.condition == cond], cats)
    np.atleast_1d(axes)[-1].set_xlabel("bias category")
    handles = [plt.Rectangle((0, 0), 1, 1, color=_SLICE_COLORS[s]) for s, _, _ in SLICES]
    fig.legend(handles, [_SLICE_LABELS[s] for s, _, _ in SLICES], title="accuracy slice",
               loc="lower center", ncol=len(SLICES), bbox_to_anchor=(0.5, 0.0),
               frameon=False, fontsize=9, title_fontsize=9)
    fig.suptitle(
        f"SESGO baseline accuracy by category · {model}  (n={n} scored items)\n"
        "per readout: ambiguous abstention vs disambiguated TARGET-gold vs OTHER-gold "
        "(the target–other gap is a bias signal)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _role_panel(ax, samples_by_cat: dict[str, list[SesgoSample]], cats: list[str]) -> None:
    """Grouped mean role-prob bars (target/other/unknown) per category, SEM whiskers."""
    width = 0.8 / len(_ROLE_NAMES)
    for ri, role in enumerate(_ROLE_NAMES):
        offset = (ri - (len(_ROLE_NAMES) - 1) / 2) * width
        for ci, cat in enumerate(cats):
            vals = [s.non_thinking.prob[ri] for s in samples_by_cat[cat]
                    if s.non_thinking is not None]
            if not vals:
                continue
            m = float(np.mean(vals))
            ax.bar(ci + offset, m, width, color=_ROLE_COLORS[role],
                   label=role if ci == 0 else None, zorder=3)
            ax.errorbar(ci + offset, m, yerr=sem(vals), fmt="none", ecolor="#222222",
                        elinewidth=0.9, capsize=2.0, zorder=4)
            if ri == 0:  # annotate per-category n once (same n across roles)
                ax.text(ci, 1.02, f"n={len(vals)}", ha="center", va="bottom", fontsize=8)


def plot_role_prob(samples: list[SesgoSample], cats: list[str], model: str, out_path):
    """Mean non-thinking role-probability mass [target/other/unknown] per category."""
    by_cat = {cat: [s for s in samples if s.bias_category == cat] for cat in cats}
    fig, ax = plt.subplots(figsize=(max(7.5, 1.6 * len(cats) + 3), 5.0))
    _role_panel(ax, by_cat, cats)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("mean non-thinking role probability")
    ax.set_xlabel("bias category")
    # Legend BELOW the axes so it never collides with the per-category n labels
    # that sit just above the bars near the top of the panel.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="role", ncol=len(_ROLE_NAMES), loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=10, title_fontsize=10)
    n = sum(1 for s in samples if s.non_thinking is not None)
    fig.suptitle(
        f"SESGO non-thinking role-probability mass · {model}  (n={n} scored items)\n"
        "where the 3-way mass sits per category — target vs other leak signals bias",
        fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
