"""Shared plotting primitives + behavioural plots for the SESGO geometry viz.

The single helper module behind ``visualize_geometry_samples.py``. It holds:
  * the house style, colour helpers, and the CI-aware bar drawer (Wilson score
    error bars + n annotations);
  * the PCA-cloud drawing routines (per-axis recolour, robust framing);
  * the two behavioural figures (per-condition 3-opt vs 2-opt; per-readout
    non-thinking vs greedy-thinking vs thinking), both with Wilson CIs and n.
Keeping the heavy drawing here lets the driver read as a high-level figure list.
"""

from __future__ import annotations

import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.math import wilson_err
from src.datasets.sesgo_eval import GeometryDataset

from sesgo.geometry.geometry_plain_labels import axis_value_label

# Internal sentinel for the no-scaffold group; rendered as "No scaffold".
BASELINE = "(baseline)"
# Colourblind-safe Okabe-Ito palette; index 0 anchors the baseline / first group.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
AXIS_PALETTE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#F0E442", "#999999", "#7E2F8E", "#4DBEEE",
)
OTHER_COLOUR = "#bbbbbb"  # neutral grey for a folded-in "(other)" bucket
OTHER_BUCKET = "(other)"
TOP_K = 8
SAVE = dict(dpi=150, bbox_inches="tight")


def apply_style() -> None:
    """A clean, legible, publication-ready global matplotlib style."""
    plt.rcParams.update({
        "figure.constrained_layout.use": True,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.edgecolor": "#555555",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
    })


def wrapped(text: str, width: int = 58) -> str:
    """Wrap a long title so the model name never clips off the right edge."""
    return "\n".join(textwrap.wrap(text, width=width))


def finish(fig: plt.Figure, out_path: Path) -> Path:
    """Save crisp + tightly cropped, then close the figure."""
    fig.savefig(out_path, **SAVE)
    plt.close(fig)
    return out_path


# ── Proportion bars with Wilson CIs ───────────────────────────────────────────


def proportion_bars(
    ax,
    labels: list[str],
    successes: list[int],
    ns: list[int],
    colours: list[str],
    ylabel: str,
) -> None:
    """Draw proportion bars with asymmetric Wilson 95% CIs + n annotations.

    Each bar is successes/n with a Wilson score interval error bar (honest at
    small n and never escaping [0,1]); the value and n are annotated above the CI
    cap. Bars with n==0 render empty with an "n=0" note so degenerate cells stay
    visible rather than vanishing.
    """
    props = [s / n if n else 0.0 for s, n in zip(successes, ns)]
    err_lo, err_hi = [], []
    for s, n in zip(successes, ns):
        lo, hi = wilson_err(s, n)
        err_lo.append(lo)
        err_hi.append(hi)
    x = np.arange(len(labels))
    ax.bar(x, props, color=colours, zorder=3)
    ax.errorbar(
        x, props, yerr=[err_lo, err_hi], fmt="none", ecolor="#333333",
        elinewidth=1.6, capsize=5, capthick=1.6, zorder=4,
    )
    for xi, p, hi, n in zip(x, props, err_hi, ns):
        txt = f"{p:.0%}\n(n={n})" if n else "n=0"
        ax.text(xi, p + hi + 0.03, txt, ha="center", va="bottom",
                fontsize=9, fontweight="bold", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=9.5)
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_ylabel(ylabel)


# ── PCA cloud drawing ─────────────────────────────────────────────────────────


def robust_limits(coords: np.ndarray, anchors: np.ndarray, pad: float = 0.18):
    """Tukey-fence x/y limits keeping the cluster + every anchor centroid in view.

    A handful of PCA outliers can stretch the raw range by an order of magnitude
    at small n, so we clip to median ± 2.5·IQR per axis, then widen to guarantee
    every supplied centroid (and thus any shift arrow) stays inside the frame.
    """
    q1, med, q3 = np.percentile(coords, [25, 50, 75], axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    lo, hi = med - 2.5 * iqr, med + 2.5 * iqr
    if anchors.size:
        lo = np.minimum(lo, anchors.min(axis=0))
        hi = np.maximum(hi, anchors.max(axis=0))
    span = np.maximum(hi - lo, 1e-6)
    return (
        (lo[0] - pad * span[0], hi[0] + pad * span[0]),
        (lo[1] - pad * span[1], hi[1] + pad * span[1]),
    )


def legend_order(values: list[str]) -> list[str]:
    """Most-frequent-first ordering, capped at TOP_K + an "(other)" bucket.

    The baseline scaffold label, when present, is pinned first for stable reading.
    """
    ranked = [v for v, _ in Counter(values).most_common()]
    if BASELINE in ranked:
        ranked = [BASELINE] + [v for v in ranked if v != BASELINE]
    if len(ranked) <= TOP_K:
        return ranked
    return ranked[:TOP_K] + [OTHER_BUCKET]


def _axis_colour(label: str, ordered: list[str]) -> str:
    """Stable colour per axis value; the folded "(other)" bucket is neutral grey."""
    if label == OTHER_BUCKET:
        return OTHER_COLOUR
    return AXIS_PALETTE[ordered.index(label) % len(AXIS_PALETTE)]


def draw_axis_scatter(ax, coords: np.ndarray, values: list[str], evr: list[float],
                      axis_key: str = "") -> int:
    """Scatter the PCA cloud coloured by one axis; return the off-view count.

    Each group's centroid anchors the robust framing so a minority group can never
    be clipped entirely out of view. ``axis_key`` selects the plain-language value
    relabelling so the legend reads in human terms, not raw row codes. Shared by
    the standalone per-axis figures and the small-multiples grid (single source of
    truth for capping / colour / frame).
    """
    ordered = legend_order(values)
    kept = {v for v in ordered if v != OTHER_BUCKET}
    centroids: list[list[float]] = []
    for lab in ordered:
        idx = [i for i, v in enumerate(values) if (v if v in kept else OTHER_BUCKET) == lab]
        if not idx:
            continue
        centroids.append(coords[idx].mean(axis=0).tolist())
        pretty = axis_value_label(axis_key, lab) if axis_key else lab
        ax.scatter(
            coords[idx, 0], coords[idx, 1], s=60, alpha=0.8,
            color=_axis_colour(lab, ordered), edgecolor="white", linewidth=0.5,
            label=f"{pretty} (n={len(idx)})", zorder=4,
        )
    xlim, ylim = robust_limits(coords, np.asarray(centroids, dtype=float))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1  ({evr[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2  ({evr[1]:.0%} of variance)" if len(evr) > 1 else "PC2")
    inside = (
        (coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1])
        & (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1])
    )
    return int((~inside).sum())


def axis_caption(values: list[str]) -> str | None:
    """Note capping when many distinct values were folded to the top few + rest."""
    n_distinct = len(set(values))
    if n_distinct <= TOP_K:
        return None
    return f"{n_distinct} distinct values; legend shows the {TOP_K} most common, rest grouped"


# ── Behavioural accuracy plots (per condition, Wilson-CI'd) ───────────────────

# Context conditions in display order, with the plain right-answer rule annotated.
_CONDITIONS = (
    ("ambig", "Ambiguous question\n(right answer: 'unknown')"),
    ("disambig", "Clear question\n(right answer is stated)"),
)
# Bar colour per condition: blue = ambiguous, orange = clear.
_COND_COLOUR = {"ambig": PALETTE[0], "disambig": PALETTE[1]}


def _bucket_by_condition(samples, correct_attr: str, defined) -> dict[str, tuple[int, int]]:
    """Successes / n of ``correct_attr`` per context condition.

    ``defined(sample)`` gates which samples are scored (e.g. a thinking draw was
    parsed, or the 2-option is defined for that condition).
    """
    out: dict[str, list[int]] = {c: [0, 0] for c, _ in _CONDITIONS}
    for s in samples:
        cond = s.context_condition
        if cond not in out or not defined(s):
            continue
        out[cond][1] += 1
        if getattr(s, correct_attr):
            out[cond][0] += 1
    return {c: (v[0], v[1]) for c, v in out.items()}


def _condition_panel(ax, buckets: dict, ylabel: str, title: str) -> None:
    """One condition-split accuracy panel with Wilson CIs + n on every bar."""
    labels = [pretty for _c, pretty in _CONDITIONS]
    succ = [buckets[c][0] for c, _ in _CONDITIONS]
    ns = [buckets[c][1] for c, _ in _CONDITIONS]
    colours = [_COND_COLOUR[c] for c, _ in _CONDITIONS]
    proportion_bars(ax, labels, succ, ns, colours, ylabel)
    ax.set_title(f"{title}  (n={sum(ns)})", fontsize=11.5)


def plot_accuracy_by_condition(dataset: GeometryDataset, out_path: Path) -> Path:
    """How often the model is right, three-way (top) vs forced two-way (bottom)."""
    samples = dataset.samples
    fig, (ax3, ax2) = plt.subplots(2, 1, figsize=(8.0, 9.0), sharey=True)
    _condition_panel(
        ax3,
        _bucket_by_condition(samples, "correct_non_thinking",
                             lambda s: s.predicted_non_thinking is not None),
        "Share of answers that are right",
        "Three-way answer (model may say 'unknown')")
    # The forced two-way choice is undefined for ambiguous items (no 'unknown' to
    # pick), so only score items where correct_2opt is a bool (clear question).
    _condition_panel(
        ax2,
        _bucket_by_condition(samples, "correct_2opt",
                             lambda s: s.picked_2opt is not None and s.correct_2opt is not None),
        "Share that pick the right group",
        "Forced two-way choice (no 'unknown' offered)")
    # Annotate the blank ambiguous bar in the empty space above it (not below the
    # axis, where it would collide with the two-line tick labels).
    ax2.text(0.25, 0.5, "No bar here: with no\n'unknown' option, no\ngroup can be correct",
             transform=ax2.transAxes, fontsize=8.5, color="#777777", style="italic",
             ha="center", va="center")
    fig.suptitle(wrapped("How often the model answers correctly, by question type "
                         f"({dataset.model_name})", 66)
                 + "\nTaller bars are better. Top: with an 'unknown' option. "
                   "Bottom: forced to pick a group.",
                 fontsize=12, fontweight="bold")
    return finish(fig, out_path)


def plot_accuracy_by_readout(dataset: GeometryDataset, out_path: Path) -> Path:
    """Accuracy under each way of reading the answer out, split by question type."""
    samples = dataset.samples
    readouts = [
        ("Without thinking (answers directly)", "correct_non_thinking",
         lambda s: s.predicted_non_thinking is not None),
        ("With thinking (single reasoned answer)",
         "correct_greedy_thinking", lambda s: s.predicted_greedy_thinking is not None),
        ("Free-form thinking (sampled reasoning draws)", "correct_thinking",
         lambda s: s.predicted_thinking is not None),
    ]
    fig, axes = plt.subplots(len(readouts), 1, figsize=(8.0, 4.0 * len(readouts)),
                             sharey=True)
    for ax, (title, attr, defined) in zip(axes, readouts):
        _condition_panel(ax, _bucket_by_condition(samples, attr, defined),
                         "Share of answers that are right", title)
    fig.suptitle(wrapped("Does letting the model reason change how often it is right? "
                         f"({dataset.model_name})", 66),
                 fontsize=13, fontweight="bold")
    return finish(fig, out_path)
