"""Render a left-to-right forking BRANCHING TREE (arXiv:2601.06116, Fig. 11/22).

Shared house-style plotter for BOTH the forking study and the divergence study:
given a flat ``BranchingTree`` of outcome-distribution nodes, draws the paper's
root → trunk → branch tree left-to-right, with each NODE shown as a horizontal
STACKED BAR over the outcome categories (Okabe-Ito palette) and each EDGE drawn
as a curved connector whose width encodes the branch's probability mass. The
decision-token labels sit on the edges; a single legend names the outcome
categories. Presentation-only — every number comes from the BranchingTree.

The plot-body functions take the flat BaseSchema tree (no nested dict/list); the
two drivers only build a BranchingTree and call ``plot_branching_tree``.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

from src.dynamics.forking_paths.forking_tree_model import (
    BRANCH_LEVEL,
    BranchingTree,
    BranchingTreeNode,
)

from sesgo.forking.forking_plot_styles import OUTCOME_COLORS, REF, save_fig

# Geometry of one node's horizontal stacked bar, in axes data units.
_NODE_W = 0.62  # bar length (the full outcome distribution spans this)
_NODE_H = 0.34  # bar height
_LEVEL_X = {"root": 0.5, "trunk": 2.4, "branch": 4.6}  # left-to-right column x


def _node_color(label: str, idx: int) -> str:
    """Okabe-Ito color for an outcome category (palette-cycled fallback)."""
    palette = ("#D55E00", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9")
    return OUTCOME_COLORS.get(label, palette[idx % len(palette)])


def _draw_node(ax, node: BranchingTreeNode, x: float, y: float, labels: list[str]) -> None:
    """Draw one node: a horizontal stacked outcome bar + its token label.

    The bar fills left-to-right by the node's outcome_histogram (so the dominant
    outcome's band is widest — the node's "answer"); a thin frame and the token
    label above keep it legible at tree scale, echoing the paper's node glyphs.
    """
    x0 = x - _NODE_W / 2
    y0 = y - _NODE_H / 2
    cursor = x0
    for i, (lbl, frac) in enumerate(zip(labels, node.outcome_histogram)):
        seg = _NODE_W * float(frac)
        if seg > 0:
            ax.add_patch(mpatches.Rectangle(
                (cursor, y0), seg, _NODE_H, facecolor=_node_color(lbl, i),
                edgecolor="white", linewidth=0.6, zorder=3,
            ))
        cursor += seg
    # Frame + token label (bold, the decision token the node sits on).
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), _NODE_W, _NODE_H, boxstyle="round,pad=0.012,rounding_size=0.04",
        facecolor="none", edgecolor="#333333", linewidth=1.1, zorder=4,
    ))
    n_txt = f"  (n={node.n_rollouts})" if node.n_rollouts else ""
    ax.text(x, y + _NODE_H / 2 + 0.07, f"“{node.label}”{n_txt}", ha="center",
            va="bottom", fontsize=8.5, fontweight="bold", color="#111111", zorder=6)


def _draw_edge(ax, x0: float, y0: float, x1: float, y1: float,
               weight: float, token: str, color: str) -> None:
    """Curved connector from parent (x0,y0) to child (x1,y1); width ∝ weight.

    The cubic Bézier mirrors the paper's swept branch edges; the line weight
    encodes p(x_t = token) (thicker = more probable continuation), and the
    branch-opening token rides the edge as a small italic label.
    """
    mid = (x0 + x1) / 2
    verts = [(x0, y0), (mid, y0), (mid, y1), (x1, y1)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    lw = 1.2 + 5.0 * max(0.0, min(1.0, weight))
    ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), facecolor="none",
                                    edgecolor=color, linewidth=lw, alpha=0.85, zorder=2))
    ax.text(mid, (y0 + y1) / 2, f"{token}\np={weight:.2f}", ha="center", va="center",
            fontsize=7.0, style="italic", color="#222222",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor=color, linewidth=0.8), zorder=5)


def _branch_layout(n_branches: int) -> list[float]:
    """Vertical y positions for ``n_branches`` branch nodes, centered on 0."""
    if n_branches <= 1:
        return [0.0]
    span = max(1.0, 0.95 * (n_branches - 1))
    top = span / 2
    return [top - i * (span / (n_branches - 1)) for i in range(n_branches)]


def _node_positions(tree: BranchingTree) -> list[tuple[float, float]]:
    """(x, y) for every node: root/trunk on the centerline, branches fanned out."""
    pos: list[tuple[float, float]] = [(0.0, 0.0)] * len(tree.nodes)
    root_i = tree.root_index
    pos[root_i] = (_LEVEL_X["root"], 0.0)
    trunk_children = tree.children_of(root_i)
    for ti in trunk_children:  # one trunk in practice
        pos[ti] = (_LEVEL_X["trunk"], 0.0)
        branches = tree.children_of(ti)
        ys = _branch_layout(len(branches))
        for bi, y in zip(branches, ys):
            pos[bi] = (_LEVEL_X["branch"], y)
    return pos


def _legend(ax, labels: list[str]) -> None:
    """Outcome-category legend (Okabe-Ito swatches) below the title."""
    handles = [mpatches.Patch(facecolor=_node_color(lbl, i), edgecolor="white", label=lbl)
               for i, lbl in enumerate(labels)]
    ax.legend(handles=handles, loc="lower center", ncol=len(labels), fontsize=8.5,
              frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.04),
              title="outcome category", title_fontsize=8.5)


def plot_branching_tree(tree: BranchingTree, path, title: str) -> str:
    """Render the branching tree to ``path`` and return the written path.

    ``title`` is the figure heading (the study + item); the paper-style subtitle
    and the per-node outcome distributions come straight from the tree.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    pos = _node_positions(tree)
    labels = tree.outcome_labels

    # Edges first (under the nodes): color each branch by its dominant outcome.
    for ci, node in enumerate(tree.nodes):
        if node.parent_index < 0:
            continue
        px, py = pos[node.parent_index]
        cx, cy = pos[ci]
        if node.level == BRANCH_LEVEL and node.outcome_histogram:
            dom = max(range(len(labels)), key=lambda i: node.outcome_histogram[i])
            color = _node_color(labels[dom], dom)
        else:
            color = REF
        _draw_edge(ax, px + _NODE_W / 2, py, cx - _NODE_W / 2, cy,
                   node.edge_weight, node.edge_token or node.label.split(" (")[0], color)

    for ci, node in enumerate(tree.nodes):
        _draw_node(ax, node, pos[ci][0], pos[ci][1], labels)

    ax.set_xlim(-0.1, _LEVEL_X["branch"] + _NODE_W)
    ys = [p[1] for p in pos]
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.axis("off")
    # Title + two stacked subtitles, each on its own line above the data so none
    # of the chrome ever collides with the title or the top branch node.
    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=0.99)
    ax.text(0.5, 1.075, "branching-tree of outcome distributions (arXiv:2601.06116)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
            color=REF, style="italic")
    ax.text(0.5, 1.035, tree.caption, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8.5, color="#444444")
    _legend(ax, labels)
    return save_fig(fig, path)
