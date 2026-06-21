"""Layer-aware + continuous-colormap drawing primitives for the geometry viz.

Companion to ``geometry_plot_helpers`` holding the NEW drawers the multi-layer /
continuous-signal upgrade needs, kept separate so each file stays small:

  * ``draw_continuous_scatter`` - scatter the PCA cloud coloured by a CONTINUOUS
    answer-distribution scalar (prob / logit / entropy / diversity / inv-ppl) via
    a sequential colormap + a colorbar (categorical axes keep the discrete legend
    in geometry_plot_helpers.draw_axis_scatter).
  * ``draw_silhouette_heatmap`` - the (layer x axis) silhouette grid showing at
    what DEPTH each colour-by axis becomes separable.
  * ``draw_layer_sweep`` - silhouette-vs-layer line sweep for a few KEY axes, so
    the per-layer differentiation is visible at a glance.

All framing reuses ``robust_limits`` from geometry_plot_helpers (single source of
truth for the PCA frame), so continuous and categorical clouds line up.
"""

from __future__ import annotations

import numpy as np

from .geometry_plot_helpers import robust_limits

# Sequential, perceptually-uniform colormap for every continuous signal.
_SEQ_CMAP = "viridis"


def draw_continuous_scatter(ax, coords: np.ndarray, values: list[float], evr: list[float]):
    """Scatter the PCA cloud coloured by a continuous scalar; return the mappable.

    Points are coloured by ``values`` on a sequential colormap (low->high); the
    caller adds a colorbar from the returned mappable. Non-finite scalars are
    dropped from the colour range but still plotted in neutral grey so the cloud
    shape is preserved. Framing reuses ``robust_limits`` (no anchors needed — the
    whole cloud is one group) so it matches the categorical panels.
    """
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    xlim, ylim = robust_limits(coords, np.empty((0, 2)))
    if (~finite).any():
        ax.scatter(coords[~finite, 0], coords[~finite, 1], s=46, color="#cccccc",
                   edgecolor="white", linewidth=0.4, zorder=3)
    vmin = float(vals[finite].min()) if finite.any() else 0.0
    vmax = float(vals[finite].max()) if finite.any() else 1.0
    sc = ax.scatter(coords[finite, 0], coords[finite, 1], c=vals[finite], s=60,
                    cmap=_SEQ_CMAP, vmin=vmin, vmax=vmax, alpha=0.9,
                    edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1  ({evr[0]:.0%} EV)")
    ax.set_ylabel(f"PC2  ({evr[1]:.0%} EV)" if len(evr) > 1 else "PC2")
    return sc


def _silhouette_grid(table: dict) -> tuple[np.ndarray, list[str], list[int]]:
    """Pack the layer_axis_silhouette table into a [n_axes, n_layers] float grid."""
    axes = table["axes"]
    layers = table["layers"]
    grid = np.array(
        [[np.nan if v is None else float(v) for v in table["values"][a]] for a in axes],
        dtype=float,
    )
    return grid, axes, layers


def draw_silhouette_heatmap(ax, table: dict) -> None:
    """Heatmap of silhouette separability over (axis rows x layer columns).

    Diverging colormap centred at 0 (a silhouette of 0 == no separation), so the
    viewer reads off WHICH axis separates and AT WHAT DEPTH. NaN cells (axis not
    scorable at that layer) render blank. The numeric value is annotated per cell.
    """
    grid, axes, layers = _silhouette_grid(table)
    vmax = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(L) for L in layers])
    ax.set_yticks(range(len(axes)))
    ax.set_yticklabels([a.replace("_", " ") for a in axes], fontsize=8.5)
    ax.set_xlabel("transformer layer (mid -> last)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="#222222")
    ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
                       label="silhouette separability")


def draw_layer_sweep(ax, table: dict, axis_keys, palette) -> None:
    """Silhouette-vs-layer line sweep for the KEY axes (depth of separation).

    One line per requested axis (skipping any the table never scored), so the
    per-layer differentiation is legible: where each line rises is the depth at
    which that axis first separates the representation.
    """
    layers = table["layers"]
    for i, key in enumerate(axis_keys):
        series = table["values"].get(key)
        if series is None:
            continue
        ys = [np.nan if v is None else float(v) for v in series]
        ax.plot(layers, ys, marker="o", lw=2.0, color=palette[i % len(palette)],
                label=key.replace("_", " "))
    ax.axhline(0, color="#888888", lw=1.0, zorder=1)
    ax.set_xlabel("transformer layer (mid -> last)")
    ax.set_ylabel("silhouette separability")
    ax.legend(title="axis", frameon=True, fontsize=9)
