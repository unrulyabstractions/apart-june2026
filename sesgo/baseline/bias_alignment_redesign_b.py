"""Render the redesigned two-panel bias-alignment-vs-accuracy figure (aesthetic B).

Soft-modern variant. Each group is ONE point at (x = bias alignment, y = accuracy)
with a thin white edge and a faint Wilson 95% accuracy whisker. The realizable
region -- |x| <= 1 - accuracy, apex at the ideal point (0, 1), base (-1,0)..(+1,0)
-- is a soft pastel fill with no harsh outline. The ideal point is a ringed star.
ONE legend below, grouped by family in columns. Panels: Ambiguous | Disambiguated.
No suptitle, no how-to-read prose.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from sesgo.baseline.bias_alignment_points_b import BiasPoint  # noqa: E402
from sesgo.baseline.bias_legend_layout_b import legend_columns  # noqa: E402

_PANEL_TITLE = {"ambig": "Ambiguous", "disambig": "Disambiguated"}
_PANEL_ORDER = ("ambig", "disambig")
_TRI_FILL = "#9ecae1"  # soft pastel blue for the realizable region
_IDEAL = "#22303f"  # deep slate for the ideal star + ring
_END_OTHER, _END_TARGET = "#0072B2", "#D55E00"  # F(other) / F(target) end hues
_GREY = "#5b6168"


def _draw_triangle(ax) -> None:
    """Soft pastel realizable region: apex (0,1), base (-1,0)..(+1,0), no outline."""
    tri = Polygon([(-1, 0), (1, 0), (0, 1)], closed=True, facecolor=_TRI_FILL,
                  edgecolor="none", alpha=0.16, zorder=0)
    ax.add_patch(tri)
    ax.axvline(0.0, color="#aeb4bb", lw=1.0, zorder=1)


def _draw_ideal(ax) -> None:
    """Ringed star at the ideal point (0, 1) with a minimal label."""
    ax.scatter([0], [1], marker="o", s=260, facecolors="none",
               edgecolors=_IDEAL, linewidths=1.2, zorder=5)
    ax.scatter([0], [1], marker="*", s=150, color=_IDEAL, zorder=6)
    ax.annotate("ideal", (0, 1), (15, 6), textcoords="offset points",
                ha="left", va="center", fontsize=8.5, color=_IDEAL, zorder=6)


def _draw_point(ax, pt: BiasPoint, colour: str) -> None:
    """One model/scaffold marker with white edge + faint Wilson accuracy whisker."""
    _, lo, hi = pt.wilson
    ax.plot([pt.bias, pt.bias], [lo, hi], color=colour, lw=1.1, alpha=0.30,
            zorder=3, solid_capstyle="round")
    ax.scatter([pt.bias], [pt.accuracy], s=130, color=colour, edgecolors="white",
               linewidths=1.1, zorder=4)


def _style_panel(ax, panel: str) -> None:
    """Limits, ticks, end descriptors and the single subplot title (plain word)."""
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-0.06, 1.08)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Bias alignment", fontsize=11)
    ax.set_title(_PANEL_TITLE[panel], fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, color="#e7e9ec", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(-1.0, -0.11, "F(other)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8.5, color=_END_OTHER, fontweight="bold")
    ax.text(1.0, -0.11, "F(target)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8.5, color=_END_TARGET, fontweight="bold")


def _legend_handle(colour: str, name: str) -> Line2D:
    """A white-edged dot proxy carrying one group's colour + display name."""
    return Line2D([0], [0], marker="o", color="none", markerfacecolor=colour,
                  markeredgecolor="white", markeredgewidth=1.0, markersize=9, label=name)


def _draw_legend(fig, order: list[str], colors: dict, names: dict, ncol: int) -> None:
    """One figure-level legend below, grouped into family columns (aesthetic B)."""
    keys = legend_columns(order, ncol)  # column-major fill so families stay together
    handles = [_legend_handle(colors[k], names[k]) for k in keys]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=9.5, handletextpad=0.5, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.02))


def plot_bias_redesign(
    points: list[BiasPoint], colors: dict, names: dict,
    order: list[str], out_path, ncol: int = 4,
) -> None:
    """Render and save the redesigned two-panel figure for the given groups."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), sharey=True)
    rank = {key: i for i, key in enumerate(order)}
    for ax, panel in zip(axes, _PANEL_ORDER):
        _draw_triangle(ax)
        panel_pts = sorted((p for p in points if p.panel == panel),
                           key=lambda p: rank.get(p.group_key, len(order)))
        for pt in panel_pts:
            _draw_point(ax, pt, colors[pt.group_key])
        _draw_ideal(ax)
        _style_panel(ax, panel)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    _draw_legend(fig, order, colors, names, ncol)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.30, wspace=0.07)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
