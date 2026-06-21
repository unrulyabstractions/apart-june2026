"""Bar-style cross-model DISTRIBUTION figures: outcome mass, bias gap, readouts.

Three size-ordered cross-model comparisons, each one figure:
  * ``plot_outcome_distribution`` — stacked mean 3-opt role mass (TARGET/OTHER/
    UNKNOWN) on ambiguous items, asking whether UNKNOWN (abstention) mass grows
    with scale.
  * ``plot_target_other_gap`` — diverging bars of the bias signal (disambiguated
    TARGET-gold accuracy minus OTHER-gold accuracy), with a propagated Wilson band.
  * ``plot_readout_agreement`` — grouped bars of ambiguous abstention under the
    three readouts (3-opt / 2-opt / greedy-thinking), each with a Wilson CI and n.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.common.math import wilson_err, wilson_interval

from sesgo.baseline.cross_model_distribution_stats import ROLE_NAMES, ModelDistribution
from sesgo.baseline.cross_model_plot_styles import (
    ROLE_COLORS,
    order_by_size,
    partial_note,
    tick_label,
)

_READOUTS = (
    ("3-opt", "abstain_3opt_succ", "abstain_3opt_total", "#0072B2"),
    ("2-opt (no UNKNOWN)", "abstain_2opt_succ", "abstain_2opt_total", "#E69F00"),
    ("greedy-thinking", "abstain_greedy_succ", "abstain_greedy_total", "#009E73"),
)


def _xticks(ax, models: list[ModelDistribution]) -> None:
    """Apply shared size-ordered, partial-flagged, rotated x tick labels."""
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([tick_label(m) for m in models], fontsize=8,
                       rotation=35, ha="right", rotation_mode="anchor")


def _footer(fig, models: list[ModelDistribution]) -> None:
    """Stamp the partial-run footnote under the figure, if any model is partial."""
    note = partial_note(models)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color="#555555", ha="left")


def plot_outcome_distribution(models: list[ModelDistribution], out_path) -> None:
    """Stacked mean 3-opt role mass on ambiguous items, ordered by size."""
    models = order_by_size(models)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(models))
    for i, role in enumerate(ROLE_NAMES):
        vals = np.array([m.mean_role_mass[i] for m in models])
        ax.bar(x, vals, bottom=bottom, color=ROLE_COLORS[role], label=role,
               edgecolor="white", linewidth=0.5)
        bottom += vals
    for xi, m in zip(x, models):  # annotate UNKNOWN mass (the abstention signal)
        ax.text(xi, 1.01, f"{m.mean_role_mass[2]:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=ROLE_COLORS["unknown"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mean 3-opt probability mass (ambiguous items)")
    _xticks(ax, models)
    ax.legend(title="role", loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_title("Outcome distribution across the model sweep — does UNKNOWN "
                 "(abstention) mass grow with scale?", fontsize=12, loc="left")
    _footer(fig, models)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _gap_band(m: ModelDistribution) -> tuple[float, float]:
    """Bias gap (target acc - other acc) and a conservative Wilson half-width."""
    pt, _, _ = wilson_interval(m.target_succ, m.target_total)
    po, _, _ = wilson_interval(m.other_succ, m.other_total)
    et = max(wilson_err(m.target_succ, m.target_total))
    eo = max(wilson_err(m.other_succ, m.other_total))
    return pt - po, float(np.hypot(et, eo))  # independent-band half-width


def plot_target_other_gap(models: list[ModelDistribution], out_path) -> None:
    """Diverging bars of disambiguated TARGET-minus-OTHER accuracy (bias signal)."""
    models = order_by_size(models)
    gaps, errs = zip(*(_gap_band(m) for m in models))
    x = np.arange(len(models))
    colors = ["#D55E00" if g >= 0 else "#0072B2" for g in gaps]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, gaps, yerr=errs, color=colors, capsize=3, edgecolor="white",
           error_kw={"elinewidth": 1.0, "ecolor": "#444444"})
    ax.axhline(0, color="#333333", lw=1.0)
    for xi, m in zip(x, models):
        ax.annotate(f"n={m.target_total}/{m.other_total}", (xi, 0),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=6.5, color="#666666")
    _xticks(ax, models)
    ax.set_ylabel("accuracy gap\n(TARGET-gold − OTHER-gold)")
    ax.set_title("Bias signal across the sweep — disambiguated accuracy gap by "
                 "gold role (>0: better on TARGET-gold; Wilson 95% band)",
                 fontsize=12, loc="left")
    _footer(fig, models)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_readout_agreement(models: list[ModelDistribution], out_path) -> None:
    """Grouped bars: ambiguous abstention under 3-opt / 2-opt / greedy readouts."""
    models = order_by_size(models)
    x = np.arange(len(models))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for j, (label, s_attr, t_attr, color) in enumerate(_READOUTS):
        offs = x + (j - 1) * width
        rates = [wilson_interval(getattr(m, s_attr), getattr(m, t_attr))[0]
                 for m in models]
        errs = np.array([[max(0.0, b), max(0.0, a)] for b, a in
                         (wilson_err(getattr(m, s_attr), getattr(m, t_attr))
                          for m in models)]).T
        ax.bar(offs, rates, width, yerr=errs, color=color, label=label,
               capsize=2, error_kw={"elinewidth": 0.8, "ecolor": "#444444"})
    ax.axhline(1.0 / 3, ls="--", lw=1.0, color="#888888")
    ax.text(0.0, 1.0 / 3, " chance ⅓", transform=ax.get_yaxis_transform(),
            fontsize=7, color="#888888", va="bottom")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("ambiguous abstention rate (chose UNKNOWN)")
    _xticks(ax, models)
    ax.legend(title="readout", loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title("Do the readouts agree? Ambiguous abstention by readout "
                 "(2-opt has no UNKNOWN → structurally 0)", fontsize=12, loc="left")
    _footer(fig, models)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
