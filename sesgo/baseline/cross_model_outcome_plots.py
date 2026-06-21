"""Bar-style cross-model DISTRIBUTION figures: outcome mass, bias gap, readouts.

Three size-ordered cross-model comparisons, each one figure: stacked role mass on
ambiguous items (does abstention grow with scale?); the diverging target-vs-other
disambiguated accuracy gap (the bias signal, with a propagated Wilson band); and
abstention under all three answering modes. All rendered text is plain language.
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
from sesgo.common.plain_language_labels import RANDOM_GUESS_LABEL, ROLE_LABEL

# Each way of reading the answer, in plain words, with its bar colour. The forced
# two-way choice never offers 'unknown', so its abstention bar is structurally 0.
_READOUTS = (
    ("Without thinking", "abstain_3opt_succ", "abstain_3opt_total", "#0072B2"),
    ("Forced two-way choice", "abstain_2opt_succ", "abstain_2opt_total", "#E69F00"),
    ("With thinking", "abstain_greedy_succ", "abstain_greedy_total", "#009E73"),
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
        ax.bar(x, vals, bottom=bottom, color=ROLE_COLORS[role], label=ROLE_LABEL[role],
               edgecolor="white", linewidth=0.5)
        bottom += vals
    for xi, m in zip(x, models):  # annotate the abstention share (green band height)
        ax.text(xi, 1.01, f"{m.mean_role_mass[2]:.0%}", ha="center", va="bottom",
                fontsize=8, color=ROLE_COLORS["unknown"])
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Share of each answer on no-answer questions", fontsize=10.5)
    _xticks(ax, models)
    ax.legend(title="What the model answered", loc="lower right",
              framealpha=0.95, fontsize=9.5, title_fontsize=10)
    ax.set_title(
        "On questions with no correct answer, how often does each model abstain?\n"
        "Green = abstains ('unknown'); taller green is safer. Numbers on top = "
        "abstention rate. Models ordered small to large.",
        fontsize=12.5, loc="left")
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
    # n labels along the bottom fringe so they never collide with the bars.
    span = max(abs(min(gaps)), abs(max(gaps))) * 1.35 or 0.1
    ax.set_ylim(-span, span)
    for xi, m in zip(x, models):
        ax.annotate(f"n={m.target_total}/{m.other_total}", (xi, -span),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=6.5, color="#777777")
    _xticks(ax, models)
    ax.set_ylabel("More accurate for the\nstereotyped group  ->", fontsize=10)
    ax.set_title(
        "When the answer is clear, is the model more accurate for one group?\n"
        "Bars above 0 = more accurate for the stereotyped group; below 0 = for the "
        "other group. Near 0 = even-handed. Whiskers are 95% confidence.",
        fontsize=12.5, loc="left")
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
    ax.text(0.0, 1.0 / 3, f" {RANDOM_GUESS_LABEL} (1 in 3)",
            transform=ax.get_yaxis_transform(),
            fontsize=8, color="#777777", va="bottom")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Abstention rate on no-answer questions", fontsize=10.5)
    _xticks(ax, models)
    ax.legend(title="How the model answered", loc="upper left",
              fontsize=9.5, framealpha=0.95, title_fontsize=10)
    ax.set_title(
        "Does the abstention story hold up across the three ways of answering?\n"
        "Higher = abstains more on no-answer questions. The forced two-way choice "
        "offers no 'unknown', so it sits at 0 by design.",
        fontsize=12.5, loc="left")
    _footer(fig, models)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
