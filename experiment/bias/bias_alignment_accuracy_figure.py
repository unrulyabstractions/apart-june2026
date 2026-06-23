"""Render the two-panel bias-alignment-vs-accuracy figure from BiasSegments.

Pure rendering: takes pre-built ``BiasSegment`` rows + a per-group colour/label
table and draws the reference layout. Two panels (Ambiguous | Disambiguated). In
each: y = Accuracy (0..1); x = "Bias Alignment" from -1 (F(other), left) to +1
(F(target), right) with a vertical line at 0. Each group is a short HORIZONTAL
segment at its accuracy height spanning its wording bias range, plus a coloured
text label carrying the signed pooled bias, e.g. ``Llama-3.2 1B (0.69)``.

Wilson 95% CI on accuracy is drawn as a faint vertical whisker at the segment's
pooled-bias point; n is annotated in the label. Okabe-Ito colours throughout.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window

import matplotlib.pyplot as plt  # noqa: E402

from experiment.bias.bias_segments import BiasSegment  # noqa: E402
from experiment.bias.segment_label_layout import spread_labels  # noqa: E402

# Plain-sentence panel headings (what the accuracy means in each context).
_PANEL_TITLE: dict[str, str] = {
    "ambig": "Ambiguous questions  -  accuracy = correctly answers 'unknown'",
    "disambig": "Clear questions  -  accuracy = picks the stated correct group",
}
_PANEL_ORDER: tuple[str, ...] = ("ambig", "disambig")
_LABEL_GREY = "#444444"
# Minimum vertical gap (accuracy units) between two text labels before de-collision.
# Each margin is split into TWO interleaved sub-columns, so this gap only needs to
# clear labels that land in the SAME sub-column (every other model by accuracy rank).
_MIN_LABEL_GAP = 0.052
# Highest a de-collided label may sit, leaving headroom under the panel title.
_LABEL_TOP = 1.0
# Where the de-collided label columns sit in each margin (outside the plotted band).
# Two sub-columns per side, staggered outward, so the dense band can interleave its
# labels left/right instead of piling into one over-tall stack.
_LEFT_COL_INNER, _LEFT_COL_OUTER = -1.30, -2.95
_RIGHT_COL_INNER, _RIGHT_COL_OUTER = 1.16, 2.81


def _draw_segment(ax, seg: BiasSegment, colour: str) -> None:
    """The horizontal bias bar at the segment's accuracy + its Wilson CI whisker."""
    y, (left, right) = seg.accuracy, seg.span
    _, lo, hi = seg.wilson
    ax.plot([seg.align_pooled, seg.align_pooled], [lo, hi], color=colour,
            lw=1.0, alpha=0.35, zorder=2, solid_capstyle="round")
    ax.plot([left, right], [y, y], color=colour, lw=3.4, alpha=0.95,
            zorder=4, solid_capstyle="round")
    ax.plot([left, right], [y, y], "|", color=colour, markersize=8,
            markeredgewidth=1.6, zorder=5)


def _draw_label(ax, seg: BiasSegment, colour: str, name: str, y_label: float,
                col_x: float, to_right: bool) -> None:
    """Coloured label in a margin sub-column + a thin leader line to its segment."""
    left, right = seg.span
    end_x, ha = (right, "left") if to_right else (left, "right")
    ax.plot([end_x, col_x], [seg.accuracy, y_label], color=colour, lw=0.6,
            alpha=0.55, zorder=3, solid_capstyle="round", clip_on=False)
    label = f"{name} ({seg.align_pooled:+.2f})  n={seg.total}"
    ax.text(col_x, y_label, label, color=colour, fontsize=8.0, va="center", ha=ha,
            fontweight="bold", zorder=6, clip_on=False)


def _label_side(ax, side: list[BiasSegment], colors: dict, names: dict,
                inner_col: float, outer_col: float, to_right: bool) -> None:
    """De-collide one margin's labels across TWO interleaved sub-columns.

    Sorting the side by accuracy and assigning even ranks to the inner column and
    odd ranks to the outer column halves the vertical density each de-collision
    sweep must resolve, so the dense accuracy band stops piling into one stack.
    """
    ranked = sorted(range(len(side)), key=lambda i: side[i].accuracy)
    columns = ([], [])  # (inner-column segs, outer-column segs)
    for rank, i in enumerate(ranked):
        columns[rank % 2].append(side[i])
    for col_segs, col_x in zip(columns, (inner_col, outer_col)):
        slots = spread_labels([s.accuracy for s in col_segs], _MIN_LABEL_GAP, hi=_LABEL_TOP)
        for slot in slots:
            seg = col_segs[slot.index]
            _draw_label(ax, seg, colors[seg.group_key], names[seg.group_key],
                        slot.y_label, col_x, to_right)


def _draw_panel(ax, segs: list[BiasSegment], colors: dict, names: dict) -> None:
    """Draw all segments, then de-collided two-column margin labels with leaders."""
    for seg in segs:
        _draw_segment(ax, seg, colors[seg.group_key])
    left_segs = [s for s in segs if s.align_pooled > 0]   # labelled in left margin
    right_segs = [s for s in segs if s.align_pooled <= 0]  # labelled in right margin
    _label_side(ax, left_segs, colors, names, _LEFT_COL_INNER, _LEFT_COL_OUTER,
                to_right=False)
    _label_side(ax, right_segs, colors, names, _RIGHT_COL_INNER, _RIGHT_COL_OUTER,
                to_right=True)


def _style_panel(ax, panel: str) -> None:
    """Axes cosmetics shared by both panels (limits, zero line, side labels)."""
    ax.axvline(0.0, color="#333333", lw=1.1, zorder=1)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Bias Alignment", fontsize=11)
    ax.set_title(_PANEL_TITLE[panel], fontsize=11.5, fontweight="bold", loc="left", pad=12)
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    # End-anchored "F(other)" / "F(target)" descriptors under the x-extremes.
    ax.text(-1.0, -0.105, "-1\nF(other)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color="#0072B2", fontweight="bold")
    ax.text(1.0, -0.105, "+1\nF(target)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color="#D55E00", fontweight="bold")
    ax.text(0.0, 1.012, "unbiased", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color=_LABEL_GREY)


def plot_bias_alignment(
    segments: list[BiasSegment], colors: dict, names: dict,
    order: list[str], suptitle: str, out_path,
) -> None:
    """Render and save the two-panel figure for the given groups (size-ordered)."""
    fig, axes = plt.subplots(1, 2, figsize=(26, 12))
    # Wide gutters + extra height: each margin now carries TWO label sub-columns
    # (inner + far-outer), so the panels need large outer breathing room (left/right)
    # AND vertical room for the de-collided stacks to spread without overlapping.
    fig.subplots_adjust(left=0.25, right=0.75, wspace=2.6, top=0.9, bottom=0.08)
    rank = {key: i for i, key in enumerate(order)}
    for ax, panel in zip(axes, _PANEL_ORDER):
        panel_segs = sorted(
            (s for s in segments if s.panel == panel),
            key=lambda s: rank.get(s.group_key, len(order)),
        )
        _draw_panel(ax, panel_segs, colors, names)
        _style_panel(ax, panel)
    axes[0].set_ylabel("Accuracy", fontsize=11, labelpad=8)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
