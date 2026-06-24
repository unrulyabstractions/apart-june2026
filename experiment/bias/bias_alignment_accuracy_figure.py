"""Render the two-panel bias-alignment-vs-accuracy figure from BiasSegments.

Pure rendering: takes pre-built ``BiasSegment`` rows + a per-group colour/label
table and draws the SESGO layout. Two panels (Ambiguous | Disambiguated). In each:
y = Accuracy (0..1); x = "Bias Alignment" from -1 (F(other), left) to +1 (F(target),
right) with a vertical line at 0. Each model is a short HORIZONTAL segment at its
accuracy height spanning its neutral-vs-negative wording-bias range, with a filled dot
at each wording endpoint and a diamond at the pooled bias; a faint vertical whisker is
its Wilson 95% CI on accuracy.

Because the alignment clusters near 0, every model's name sits in a SINGLE de-collided
column just inside the panel's outer edge (ambiguous -> left, disambiguated -> right),
sorted by accuracy so the thin leader lines never cross. Okabe-Ito family colours,
darker = larger model.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window

import matplotlib.pyplot as plt  # noqa: E402

from experiment.bias.bias_segments import BiasSegment  # noqa: E402
from experiment.bias.segment_label_layout import spread_labels  # noqa: E402

_PANEL_TITLE: dict[str, str] = {
    "ambig": "Ambiguous questions  (correct = abstains)",
    "disambig": "Clear questions  (correct = picks right group)",
}
_PANEL_ORDER: tuple[str, ...] = ("ambig", "disambig")
_MIN_LABEL_GAP = 0.052
_LABEL_TOP = 0.99
# Single label column just inside each panel's outer edge (kept off the data, which
# clusters near x=0): ambiguous labels hug the left, disambiguated the right.
_LEFT_COL, _RIGHT_COL = -0.62, 0.62
# Verbose HF suffixes stripped from labels so model names stay short and readable.
_NAME_JUNK = ("-Instruct-2501", "-Instruct-2512", "-Instruct-v0.3",
              "-Instruct", "-2501", "-2512")


def _short(name: str) -> str:
    for junk in _NAME_JUNK:
        name = name.replace(junk, "")
    return name.strip()


def _draw_segment(ax, seg: BiasSegment, colour: str) -> None:
    """Wording-bias bar at the model's accuracy: CI whisker, the bar, endpoint dots,
    and a diamond at the pooled bias (the labelled value)."""
    y, (left, right) = seg.accuracy, seg.span
    _, lo, hi = seg.wilson
    ax.plot([seg.align_pooled, seg.align_pooled], [lo, hi], color=colour,
            lw=1.2, alpha=0.30, zorder=2, solid_capstyle="round")
    ax.plot([left, right], [y, y], color=colour, lw=3.0, alpha=0.9,
            zorder=4, solid_capstyle="round")
    ax.plot([left, right], [y, y], "o", color=colour, ms=5.5, zorder=5)
    ax.plot([seg.align_pooled], [y], "D", color=colour, mec="white", mew=0.8,
            ms=6.5, zorder=6)


def _draw_label(ax, seg: BiasSegment, colour: str, name: str, y_label: float,
                col_x: float, to_right: bool) -> None:
    """Coloured name+bias label in the outer column + a thin leader to its segment."""
    left, right = seg.span
    end_x, ha = (right, "left") if to_right else (left, "right")
    ax.plot([end_x, col_x], [seg.accuracy, y_label], color=colour, lw=0.6,
            alpha=0.45, zorder=3, solid_capstyle="round")
    label = f"{_short(name)} ({seg.bias_score:+.2f})"
    ax.text(col_x, y_label, label, color=colour, fontsize=8.5, va="center", ha=ha,
            fontweight="bold", zorder=6)


def _draw_panel(ax, segs: list[BiasSegment], colors: dict, names: dict,
                col_x: float, to_right: bool) -> None:
    """Draw all segments, then one de-collided label column sorted by accuracy so the
    leaders are monotonic (and never cross)."""
    for seg in segs:
        _draw_segment(ax, seg, colors[seg.group_key])
    ranked = sorted(segs, key=lambda s: s.accuracy)
    slots = spread_labels([s.accuracy for s in ranked], _MIN_LABEL_GAP, hi=_LABEL_TOP)
    for slot in slots:
        seg = ranked[slot.index]
        _draw_label(ax, seg, colors[seg.group_key], names[seg.group_key],
                    slot.y_label, col_x, to_right)


def _style_panel(ax, panel: str) -> None:
    """Axes cosmetics shared by both panels (limits, zero line, side descriptors)."""
    ax.axvspan(-0.05, 0.05, color="#000000", alpha=0.04, zorder=0)  # faint "near-unbiased" band
    ax.axvline(0.0, color="#333333", lw=1.0, zorder=1)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Bias alignment", fontsize=11)
    ax.set_title(_PANEL_TITLE[panel], fontsize=12, fontweight="bold", pad=14)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(-1.0, -0.10, "F(other)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color="#0072B2", fontweight="bold")
    ax.text(1.0, -0.10, "F(target)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color="#D55E00", fontweight="bold")


def plot_bias_alignment(
    segments: list[BiasSegment], colors: dict, names: dict,
    order: list[str], suptitle: str, out_path,
) -> None:
    """Render and save the two-panel figure for the given groups (size-ordered)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.5), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, wspace=0.06, top=0.88, bottom=0.11)
    rank = {key: i for i, key in enumerate(order)}
    panel_cols = {"ambig": (_LEFT_COL, False), "disambig": (_RIGHT_COL, True)}
    for ax, panel in zip(axes, _PANEL_ORDER):
        panel_segs = sorted(
            (s for s in segments if s.panel == panel),
            key=lambda s: rank.get(s.group_key, len(order)),
        )
        col_x, to_right = panel_cols[panel]
        _draw_panel(ax, panel_segs, colors, names, col_x, to_right)
        _style_panel(ax, panel)
    axes[0].set_ylabel("Accuracy", fontsize=11, labelpad=6)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.965)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
