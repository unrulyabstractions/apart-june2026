"""Minimal house-style chrome for the depth-scatter panels (2D + 3D).

One place that owns the colourblind-safe categorical palette, the short legend
text (axis value + n, no explanatory title), the continuous colormap, and the
tiny axis labels. Per the figures' MINIMAL-TEXT rule there is NO suptitle, NO
"how to read" line, and NO separation-score prose: a panel carries only its data,
short tick/legend labels, and minimal axis labels. The 2D and 3D renderers reuse
the shared ``COLOR_AXES`` registry, ``legend_order`` capping, and the plain value
relabelling so they stay tiny and read identically.
"""

from __future__ import annotations

import numpy as np

from sesgo.geometry.geometry_plain_labels import axis_value_label
from sesgo.geometry.geometry_plot_helpers import (
    AXIS_PALETTE,
    OTHER_BUCKET,
    OTHER_COLOUR,
    legend_order,
)

# Sequential, perceptually-uniform colormap for every continuous colour axis.
SEQ_CMAP = "viridis"
# Neutral grey for points whose continuous value is missing / non-finite.
MISSING_COLOUR = "#cccccc"


def cat_groups(values: list[str], axis_key: str) -> list[tuple[str, str, list[int]]]:
    """(colour, short label, point indices) per categorical group, in legend order.

    Caps to the top-K most common values plus a folded "(other)" bucket (reusing
    ``legend_order``), recolours via the shared Okabe-Ito palette, and reads each
    label through the plain value relabelling. Label is just ``value (n=...)`` —
    no explanatory legend title, per the minimal-text rule.
    """
    ordered = legend_order(values)
    kept = {v for v in ordered if v != OTHER_BUCKET}
    out: list[tuple[str, str, list[int]]] = []
    for lab in ordered:
        idx = [i for i, v in enumerate(values) if (v if v in kept else OTHER_BUCKET) == lab]
        if not idx:
            continue
        colour = OTHER_COLOUR if lab == OTHER_BUCKET else AXIS_PALETTE[
            ordered.index(lab) % len(AXIS_PALETTE)]
        out.append((colour, f"{axis_value_label(axis_key, lab)} (n={len(idx)})", idx))
    return out


def outside_legend(ax, handles, labels) -> None:
    """Place the colour legend OUTSIDE the axes, centred below the plot."""
    ncol = min(4, max(1, -(-len(labels) // 3)))  # at most three rows of entries.
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=ncol, frameon=False, fontsize=9, markerscale=2.0,
              handletextpad=0.3, columnspacing=1.2)


def draw_categorical(ax, pts: np.ndarray, values: list, axis_key: str, dims: int) -> None:
    """Scatter one categorical axis (capped legend, plain labels) into 2D or 3D."""
    handles, labels = [], []
    for colour, label, idx in cat_groups([str(v) for v in values], axis_key):
        cols = tuple(pts[idx, d] for d in range(dims))
        h = ax.scatter(*cols, s=9 if dims == 3 else 11, alpha=0.4 if dims == 3 else 0.5,
                       color=colour, edgecolors="none",
                       **({"depthshade": True} if dims == 3 else {}))
        handles.append(h)
        labels.append(label)
    outside_legend(ax, handles, labels)


def draw_continuous(ax, fig, pts: np.ndarray, values: list, dims: int) -> None:
    """Scatter one continuous axis with a sequential colormap + a colorbar."""
    vals = np.asarray([v if isinstance(v, (int, float)) else np.nan for v in values],
                      dtype=float)
    finite = np.isfinite(vals)
    if (~finite).any():
        ax.scatter(*[pts[~finite, d] for d in range(dims)], s=8 if dims == 3 else 10,
                   color=MISSING_COLOUR, edgecolors="none")
    vmin = float(vals[finite].min()) if finite.any() else 0.0
    vmax = float(vals[finite].max()) if finite.any() else 1.0
    sc = ax.scatter(*[pts[finite, d] for d in range(dims)], c=vals[finite],
                    s=9 if dims == 3 else 11, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax,
                    alpha=0.5 if dims == 3 else 0.7, edgecolors="none")
    fig.colorbar(sc, ax=ax, fraction=0.03 if dims == 3 else 0.046,
                 pad=0.08 if dims == 3 else 0.02)
