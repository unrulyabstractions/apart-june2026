"""Stream the giant projections.json and pull only the depth cells we plot.

A model's ``analysis/projections.json`` is hundreds of MB to ~2 GB, so we never
parse it whole. For a chosen token position we locate the byte span of each
(layer, position) cell with a header regex, then pull from the WANTED layers only:
each sample's per-scaffold 2D + 3D PCA coordinates, the explained-variance of the
first three PCA axes, and the scaffold-separation score with its 95% CI. Depth is
``layer / n_layers``; we map each target depth to the nearest available layer.

Used by ``depth_scatter_panel_render.py`` (run-by-path) to draw 2D + 3D scatters.
"""

from __future__ import annotations

import bisect
import mmap
import re
from dataclasses import dataclass, field
from pathlib import Path

from sesgo.common import normalize_scaffold
from src.common.base_schema import BaseSchema

# One source of truth for how deep each model is (depth = layer / n_layers).
MODEL_LAYERS: dict[str, int] = {"Qwen3-0.6B": 28, "Qwen3-32B": 64}
TARGET_DEPTHS: tuple[float, ...] = (0.5, 0.7, 0.9)
# The answer token: the state right where the model commits to its answer.
POSITION = "label"


@dataclass
class DepthCell(BaseSchema):
    """One (model, depth) cell: scaffold-split 2D + 3D clouds, score, variance."""

    model: str = ""
    depth: float = 0.0
    layer: str = ""
    n_layers: int = 0
    silhouette: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    pc_var: list = field(default_factory=list)  # [PC1%, PC2%, PC3%] of spread.
    baseline_xyz: list = field(default_factory=list)  # [[x,y,z], ...] no-scaffold.
    scaffold_xyz: list = field(default_factory=list)  # [[x,y,z], ...] with scaffold.


def nearest_layers(model: str) -> dict[float, str]:
    """Map each target depth to the available layer closest to it for this model."""
    n = MODEL_LAYERS[model]
    avail = available_layers(model)
    out: dict[float, str] = {}
    for depth in TARGET_DEPTHS:
        target = depth * n
        out[depth] = min(avail, key=lambda lyr: abs(int(lyr) - target))
    return out


def available_layers(model: str) -> list[str]:
    """Numeric layer ids present in this model's projections, ascending."""
    with _mmap(model) as mm:
        return [name for _, name in _layer_headers(mm)]


def load_depth_cells(model: str) -> list[DepthCell]:
    """Stream the projections file and return one DepthCell per target depth."""
    n = MODEL_LAYERS[model]
    want = nearest_layers(model)
    layer_of_depth = {layer: depth for depth, layer in want.items()}
    with _mmap(model) as mm:
        spans = _cell_spans(mm)
        cells = []
        for depth, layer in want.items():
            o, end = spans[(layer, POSITION)]
            cell = _read_cell(mm[o:end])
            cell.model, cell.depth, cell.layer, cell.n_layers = model, depth, layer, n
            cells.append(cell)
        # Keep depths in ascending order regardless of file/header order.
        return sorted(cells, key=lambda c: c.depth)


def _projections_path(model: str) -> Path:
    return Path(f"out/sesgo/geometry/{model}/analysis/projections.json")


class _mmap:
    """Context-managed read-only memory map of a model's projections file."""

    def __init__(self, model: str) -> None:
        self._path = _projections_path(model)

    def __enter__(self) -> memoryview:
        self._f = open(self._path, "r+b")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm

    def __exit__(self, *exc) -> None:
        self._mm.close()
        self._f.close()


_LAYER_HDR = re.compile(rb'\n        "(\d+|mean)": \{')
_POS_HDR = re.compile(rb'\n            "([a-z_]+)": \{')
_SIL = re.compile(
    rb'"scaffold_stats".*?"silhouette": ([\-0-9eE.]+),\s*'
    rb'"silhouette_ci_low": ([\-0-9eE.]+),\s*"silhouette_ci_high": ([\-0-9eE.]+)', re.S)
_EVR = re.compile(
    rb'"explained_variance_ratio": \[\s*([\-0-9eE.]+),\s*([\-0-9eE.]+),\s*([\-0-9eE.]+)')
_SAMP = re.compile(
    rb'"scaffold_id": (null|"[^"]*").*?'
    rb'"coord3d": \[\s*([\-0-9eE.]+),\s*([\-0-9eE.]+),\s*([\-0-9eE.]+)\s*\]', re.S)


def _layer_headers(mm: memoryview) -> list[tuple[int, str]]:
    """(byte offset, layer name) for every numeric/mean layer header, in file order."""
    hdr = [(m.start(), m.group(1).decode()) for m in _LAYER_HDR.finditer(mm)]
    return [h for h in hdr if h[1].isdigit()]


def _cell_spans(mm: memoryview) -> dict:
    """Byte span of every (layer, position) results cell via interleaved headers."""
    lhdr = _layer_headers(mm)
    loff = [o for o, _ in lhdr]
    phdr = [(m.start(), m.group(1).decode()) for m in _POS_HDR.finditer(mm)]
    marks = sorted(phdr + [(o, None) for o in loff] + [(len(mm), None)])
    spans = {}
    for i, (off, name) in enumerate(marks):
        if name is None:
            continue
        layer = lhdr[bisect.bisect_right(loff, off) - 1][1]
        spans[(layer, name)] = (off, marks[i + 1][0])
    return spans


def _read_cell(region: memoryview) -> DepthCell:
    """Parse one cell's score, 3-axis variance, and per-scaffold 3D coordinates."""
    sil, evr = _SIL.search(region), _EVR.search(region)
    samples = region[: region.find(b'"scaffold_stats"')]
    base, scaf = [], []
    for m in _SAMP.finditer(samples):
        xyz = [float(m.group(i)) for i in (2, 3, 4)]
        sid = None if m.group(1) == b"null" else m.group(1).decode().strip('"')
        (base if normalize_scaffold(sid) is None else scaf).append(xyz)
    return DepthCell(
        silhouette=float(sil.group(1)), ci_low=float(sil.group(2)), ci_high=float(sil.group(3)),
        pc_var=[float(evr.group(i)) * 100 for i in (1, 2, 3)],
        baseline_xyz=base, scaffold_xyz=scaf)
