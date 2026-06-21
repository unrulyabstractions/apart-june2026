"""Narrative key figure F7: the clearest scaffold decision boundary in the model.

One standalone 2D-PCA scatter at the single (layer, token-position) cell whose
internal state separates MOST cleanly by debiasing scaffold. We scan the geometry
projections for the cell with the highest scaffold separation score, then draw its
2D cloud coloured by scaffold so a reviewer can SEE that a one-line debiasing
preamble carves the model's mid-network state into two distinct regions. The 869 MB
projections file is never fully parsed: a streaming regex pulls only the winning
cell's per-sample 2D coordinates plus that cell's separation score + CI.

Run by path:  .venv/bin/python sesgo/geometry/scaffold_boundary_pca_figure.py
"""

from __future__ import annotations

import bisect
import mmap
import pathlib
import re
import sys
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from sesgo.common import SCAFFOLD_LABEL, normalize_scaffold  # noqa: E402
from src.common.base_schema import BaseSchema  # noqa: E402

PROJ = pathlib.Path("out/sesgo/geometry/Qwen3-0.6B/analysis/projections.json")
OUT = pathlib.Path("out/sesgo/geometry/Qwen3-0.6B/plots/scaffold_boundary_pca.png")
TOTAL_LAYERS = 28  # Qwen3-0.6B hidden layers, indexed 0..27.
LAYERS = [str(i) for i in range(14, 28)] + ["mean"]
POSITIONS = ["im_end", "newline", "im_start", "assistant", "think_open",
             "think_close", "answer_prefix", "label"]
OK = {"baseline": "#0072B2", "scaffold": "#E69F00"}  # Okabe-Ito, colourblind-safe.


@dataclass
class BoundaryCell(BaseSchema):
    """The winning (layer, position) cell: its scaffold separation + sample cloud."""

    layer: str = ""
    position: str = ""
    silhouette: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    pc1_var: float = 0.0
    pc2_var: float = 0.0
    baseline_xy: list = field(default_factory=list)
    scaffold_xy: list = field(default_factory=list)


def _cell_spans(mm: memoryview) -> dict:
    """Byte span of every (layer, position) results cell, via header offsets."""
    lpat = re.compile(rb'\n        "(' + b"|".join(l.encode() for l in LAYERS) + rb')": \{')
    ppat = re.compile(rb'\n            "(' + b"|".join(p.encode() for p in POSITIONS) + rb')": \{')
    lhdr = [(m.start(), m.group(1).decode()) for m in lpat.finditer(mm)]
    phdr = [(m.start(), m.group(1).decode()) for m in ppat.finditer(mm)]
    loff = [o for o, _ in lhdr]
    marks = sorted(phdr + [(o, None) for o in loff] + [(len(mm), None)])
    spans = {}
    for i, (off, name) in enumerate(marks):
        if name is None:
            continue
        layer = lhdr[bisect.bisect_right(loff, off) - 1][1]
        spans[(layer, name)] = (off, marks[i + 1][0])
    return spans


_SIL = re.compile(
    rb'"scaffold_stats".*?"silhouette": ([\-0-9eE.]+),\s*'
    rb'"silhouette_ci_low": ([\-0-9eE.]+),\s*"silhouette_ci_high": ([\-0-9eE.]+)', re.S)
_EVR = re.compile(rb'"explained_variance_ratio": \[\s*([\-0-9eE.]+),\s*([\-0-9eE.]+)')
_SAMP = re.compile(
    rb'"scaffold_id": (null|"[^"]*").*?"coord2d": \[\s*([\-0-9eE.]+),\s*([\-0-9eE.]+)\s*\]', re.S)


def _best_cell(mm: memoryview, spans: dict) -> tuple:
    """The (layer, position) with the maximum scaffold separation score."""
    best, best_sil = None, -2.0
    for key, (o, end) in spans.items():
        m = _SIL.search(mm[o:end])
        if m and float(m.group(1)) > best_sil:
            best_sil, best = float(m.group(1)), key
    return best


def _read_cell(mm: memoryview, span: tuple, key: tuple) -> BoundaryCell:
    """Parse the winning cell's score, variance, and per-scaffold 2D coordinates."""
    o, end = span
    region = mm[o:end]
    sil = _SIL.search(region)
    evr = _EVR.search(region)
    samples = region[: region.find(b'"scaffold_stats"')]
    base, scaf = [], []
    for m in _SAMP.finditer(samples):
        xy = [float(m.group(2)), float(m.group(3))]
        sid = None if m.group(1) == b"null" else m.group(1).decode().strip('"')
        (base if normalize_scaffold(sid) is None else scaf).append(xy)
    return BoundaryCell(
        layer=key[0], position=key[1],
        silhouette=float(sil.group(1)), ci_low=float(sil.group(2)), ci_high=float(sil.group(3)),
        pc1_var=float(evr.group(1)) * 100, pc2_var=float(evr.group(2)) * 100,
        baseline_xy=base, scaffold_xy=scaf)


def load_boundary() -> BoundaryCell:
    """Stream the projections file and return only the clearest-boundary cell."""
    with open(PROJ, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        spans = _cell_spans(mm)
        key = _best_cell(mm, spans)
        return _read_cell(mm, spans[key], key)


def render(cell: BoundaryCell) -> None:
    """Draw the PC1-PC2 scatter coloured by scaffold with a plain-language caption."""
    base = np.array(cell.baseline_xy)
    scaf = np.array(cell.scaffold_xy)
    frac = int(cell.layer) / (TOTAL_LAYERS - 1)
    where = ("a quarter of the way" if frac < 0.4 else
             "halfway" if frac < 0.6 else
             "two thirds of the way" if frac < 0.75 else "deep")
    base_name = SCAFFOLD_LABEL[None]
    scaf_name = SCAFFOLD_LABEL["interpretive_direction"]

    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    for xy, c, lbl in ((base, OK["baseline"], f"{base_name}  (n={len(base)})"),
                       (scaf, OK["scaffold"], f"{scaf_name}  (n={len(scaf)})")):
        ax.scatter(xy[:, 0], xy[:, 1], s=10, alpha=0.45, color=c, edgecolors="none", label=lbl)

    ax.set_title(
        "A one-sentence debiasing instruction splits the model's internal state\n"
        f"into two clearly separated regions  (layer {cell.layer} of "
        f"{TOTAL_LAYERS - 1}, {where} through the network)",
        fontsize=15, fontweight="bold", pad=58)
    ax.text(0.5, 1.085,
            "How to read this: each dot is one question's internal state; the further apart\n"
            "the two colours sit, the more the instruction reshaped what the model represents.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, color="#444")

    ax.set_xlabel(f"Main axis of variation  ({cell.pc1_var:.0f}% of the spread)", fontsize=11)
    ax.set_ylabel(f"Second axis of variation  ({cell.pc2_var:.0f}% of the spread)", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
              frameon=False, fontsize=10.5, markerscale=2.2, handletextpad=0.3)

    sep = (f"Separation score = {cell.silhouette:.2f}  "
           f"(95% CI {cell.ci_low:.2f}-{cell.ci_high:.2f})\n"
           "1 = the two groups sit fully apart    0 = they overlap    "
           "below 0 = they are intermixed")
    ax.text(0.025, 0.965, sep, transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bbb", alpha=0.95))

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9.5)
    fig.subplots_adjust(top=0.72, bottom=0.16, left=0.085, right=0.975)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    render(load_boundary())
    print(f"wrote {OUT}")
