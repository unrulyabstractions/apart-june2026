"""Render 2D + 3D PCA depth scatters recoloured by EVERY colour axis, per model.

For Qwen3-0.6B and Qwen3-32B this streams the answer-token depth cells out of the
giant projections.json (via depth_scatter_data_loader) and, at each target depth
(0.5 / 0.7 / 0.9), draws a 2D and a 3D PCA scatter coloured by each axis in the
shared ``COLOR_AXES`` registry. Categorical axes get a capped legend OUTSIDE the
axes; continuous axes get a colorbar. Per the minimal-text rule a panel carries
only its data, a <=4-word title, short axis labels, and the legend/colorbar.

Output is one tidy subfolder per colour axis:
  out/sesgo/geometry/<MODEL>/plots/depth_scatters/<axis>/{2d,3d}_depth{50,70,90}.png

Run by path:  .venv/bin/python sesgo/geometry/depth_scatter_panel_render.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers '3d' proj)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sesgo.geometry.depth_scatter_data_loader import DepthCell, load_depth_cells  # noqa: E402
from sesgo.geometry.depth_scatter_figure_style import (  # noqa: E402
    draw_categorical,
    draw_continuous,
)
from sesgo.geometry.geometry_color_axes import COLOR_AXES  # noqa: E402

MODELS = ("Qwen3-0.6B", "Qwen3-32B")


def _out_path(cell: DepthCell, axis_key: str, kind: str) -> Path:
    """plots/depth_scatters/<axis>/<kind>_depth<pct>.png (one subfolder per axis)."""
    folder = Path(f"out/sesgo/geometry/{cell.model}/plots/depth_scatters/{axis_key}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{kind}_depth{int(cell.depth * 100)}.png"


def _robust_limits(values: np.ndarray) -> tuple[float, float]:
    """Frame the bulk of the data so rare outliers don't crush the cloud."""
    lo, hi = np.percentile(values, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    pad = 0.08 * (hi - lo) or 1.0
    return lo - pad, hi + pad


def _frame(ax, pts: np.ndarray, dims: int) -> None:
    """Clip each plotted axis to robust limits."""
    for col, name in enumerate(("set_xlim", "set_ylim", "set_zlim")[:dims]):
        getattr(ax, name)(*_robust_limits(pts[:, col]))


def _depth_tag(ax, cell: DepthCell) -> None:
    """A single short depth tag in the corner (no sentence, per minimal text)."""
    fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    fn(0.0, 1.0, f"depth {cell.depth * 100:.0f}%", transform=ax.transAxes,
       ha="left", va="bottom", fontsize=9, color="#666")


def _scatter(ax, fig, pts: np.ndarray, cell: DepthCell, axis, dims: int) -> None:
    """Colour the cloud by one axis: continuous colorbar or capped categorical legend."""
    values = cell.columns[axis.key]
    if axis.continuous:
        draw_continuous(ax, fig, pts, values, dims)
    else:
        draw_categorical(ax, pts, values, axis.key, dims)


def render_2d(cell: DepthCell, axis) -> Path:
    """2D PCA scatter (PC1 vs PC2) coloured by one axis at one depth."""
    pts = np.asarray(cell.coords2d, dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    _scatter(ax, fig, pts, cell, axis, dims=2)
    _frame(ax, pts, dims=2)
    ax.set_title(axis.key.replace("_", " "), fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(f"PC1  ({cell.pc_var[0]:.0f}%)", fontsize=10)
    ax.set_ylabel(f"PC2  ({cell.pc_var[1]:.0f}%)", fontsize=10)
    _depth_tag(ax, cell)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5)
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.09, right=0.97)
    return _save(fig, _out_path(cell, axis.key, "2d"))


def render_3d(cell: DepthCell, axis) -> Path:
    """3D PCA scatter (PC1/PC2/PC3) coloured by one axis at one depth."""
    pts = np.asarray(cell.coords3d, dtype=float)
    fig = plt.figure(figsize=(9.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    _scatter(ax, fig, pts, cell, axis, dims=3)
    _frame(ax, pts, dims=3)
    ax.view_init(elev=18, azim=-60)
    ax.set_title(axis.key.replace("_", " "), fontsize=13, fontweight="bold", y=0.97)
    ax.set_xlabel(f"\nPC1 ({cell.pc_var[0]:.0f}%)", fontsize=9)
    ax.set_ylabel(f"\nPC2 ({cell.pc_var[1]:.0f}%)", fontsize=9)
    ax.set_zlabel(f"\nPC3 ({cell.pc_var[2]:.0f}%)", fontsize=9)
    _depth_tag(ax, cell)
    ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.04, right=0.97)
    return _save(fig, _out_path(cell, axis.key, "3d"))


def _save(fig, path: Path) -> Path:
    """Write a figure at review-quality DPI and free its memory."""
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def render_all() -> list[Path]:
    """Render every axis's 2D + 3D depth panel for both Qwen models; return paths."""
    written = []
    for model in MODELS:
        for cell in load_depth_cells(model):
            for axis in COLOR_AXES:
                written.append(render_2d(cell, axis))
                written.append(render_3d(cell, axis))
    return written


if __name__ == "__main__":
    for p in render_all():
        print(f"wrote {p}")
