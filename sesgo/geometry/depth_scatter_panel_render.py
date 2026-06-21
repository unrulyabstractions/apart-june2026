"""Render 2D + 3D PCA scaffold scatters at depths 0.5 / 0.7 / 0.9, per model.

For Qwen3-0.6B and Qwen3-32B this streams the depth cells out of the giant
projections.json (via depth_scatter_data_loader) and draws, at each target depth,
one 2D PCA scatter and one matplotlib-3D PCA scatter coloured by debiasing
scaffold. Plain-sentence titles, a 'how to read' subtitle, an Okabe-Ito legend
OUTSIDE the axes, and the separation score + 95% CI in its corrected reading.

Output lands in a tidy per-model subfolder so plots are not a flat dump:
  out/sesgo/geometry/<MODEL>/plots/depth_scatters/{2d,3d}_depth{50,70,90}.png

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
    HOW_TO_READ,
    panel_title,
    scaffold_legend,
    separation_text,
)

MODELS = ("Qwen3-0.6B", "Qwen3-32B")


def _out_path(model: str, kind: str, depth: float) -> Path:
    """Tidy per-model subfolder path: plots/depth_scatters/<kind>_depth<pct>.png."""
    folder = Path(f"out/sesgo/geometry/{model}/plots/depth_scatters")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{kind}_depth{int(depth * 100)}.png"


def _clouds(cell: DepthCell) -> tuple[np.ndarray, np.ndarray]:
    """The baseline and scaffold point clouds as Nx3 arrays."""
    return np.array(cell.baseline_xyz), np.array(cell.scaffold_xyz)


def _robust_limits(values: np.ndarray) -> tuple[float, float]:
    """Axis range that frames the bulk of the data, so rare outliers don't crush it.

    Clip to the 0.5-99.5 percentile band and pad by 8% of that span; a handful of
    extreme points still render but no longer compress the two clusters into a
    sliver. Falls back to the raw min/max when the band is degenerate.
    """
    lo, hi = np.percentile(values, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    pad = 0.08 * (hi - lo) or 1.0
    return lo - pad, hi + pad


def _score_box(ax, cell: DepthCell) -> None:
    """Draw the separation-score box in the top-left corner of either axes type."""
    text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text  # 3D vs 2D axes.
    text_fn(0.02, 0.98, separation_text(cell.silhouette, cell.ci_low, cell.ci_high),
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bbb", alpha=0.95))


def _legend(ax, cell: DepthCell, handles) -> None:
    """Place the scaffold legend OUTSIDE the axes, centred below the plot."""
    rows = scaffold_legend(len(cell.baseline_xyz), len(cell.scaffold_xyz))
    ax.legend(handles, [lbl for *_, lbl in rows], loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=10.5,
              markerscale=2.2, handletextpad=0.3)


def _frame_axis(ax, both: np.ndarray, dims: int) -> None:
    """Clip each plotted axis to robust limits so outliers don't crush the clouds."""
    names = ("set_xlim", "set_ylim", "set_zlim")[:dims]  # set_zlim only on 3D axes.
    for col, name in enumerate(names):
        getattr(ax, name)(*_robust_limits(both[:, col]))


def render_2d(cell: DepthCell) -> Path:
    """2D PCA scatter (PC1 vs PC2) coloured by scaffold for one depth."""
    base, scaf = _clouds(cell)
    fig, ax = plt.subplots(figsize=(10.5, 7.8))
    handles = [ax.scatter(c[:, 0], c[:, 1], s=10, alpha=0.45, color=color,
                          edgecolors="none")
               for c, (color, *_) in zip((base, scaf), scaffold_legend(len(base), len(scaf)))]
    ax.set_title(panel_title("2D", cell.depth, cell.layer, cell.n_layers),
                 fontsize=14.5, fontweight="bold", pad=58)
    ax.text(0.5, 1.085, HOW_TO_READ, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=10.5, color="#444")
    ax.set_xlabel(f"Main axis of variation  ({cell.pc_var[0]:.0f}% of the spread)", fontsize=11)
    ax.set_ylabel(f"Second axis of variation  ({cell.pc_var[1]:.0f}% of the spread)", fontsize=11)
    _frame_axis(ax, np.vstack([base, scaf]), dims=2)
    _legend(ax, cell, handles)
    _score_box(ax, cell)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9.5)
    fig.subplots_adjust(top=0.72, bottom=0.15, left=0.085, right=0.975)
    return _save(fig, _out_path(cell.model, "2d", cell.depth))


def render_3d(cell: DepthCell) -> Path:
    """3D PCA scatter (PC1/PC2/PC3) coloured by scaffold for one depth."""
    base, scaf = _clouds(cell)
    fig = plt.figure(figsize=(10.5, 8.4))
    ax = fig.add_subplot(111, projection="3d")
    handles = [ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=8, alpha=0.35, color=color,
                          edgecolors="none", depthshade=True)
               for c, (color, *_) in zip((base, scaf), scaffold_legend(len(base), len(scaf)))]
    fig.suptitle(panel_title("3D", cell.depth, cell.layer, cell.n_layers),
                 fontsize=14.5, fontweight="bold", y=0.98)
    fig.text(0.5, 0.855, HOW_TO_READ, ha="center", va="top", fontsize=10.5, color="#444")
    ax.set_xlabel(f"\nMain axis ({cell.pc_var[0]:.0f}%)", fontsize=10)
    ax.set_ylabel(f"\nSecond axis ({cell.pc_var[1]:.0f}%)", fontsize=10)
    ax.set_zlabel(f"\nThird axis ({cell.pc_var[2]:.0f}%)", fontsize=10)
    _frame_axis(ax, np.vstack([base, scaf]), dims=3)
    ax.view_init(elev=18, azim=-60)
    _legend(ax, cell, handles)
    _score_box(ax, cell)
    ax.tick_params(labelsize=8)
    fig.subplots_adjust(top=0.78, bottom=0.1, left=0.04, right=0.97)
    return _save(fig, _out_path(cell.model, "3d", cell.depth))


def _save(fig, path: Path) -> Path:
    """Write a figure at review-quality DPI and free its memory."""
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def render_all() -> list[Path]:
    """Render every 2D + 3D depth panel for both Qwen models; return the paths."""
    written = []
    for model in MODELS:
        for cell in load_depth_cells(model):
            written.append(render_2d(cell))
            written.append(render_3d(cell))
    return written


if __name__ == "__main__":
    for p in render_all():
        print(f"wrote {p}")
