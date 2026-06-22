"""Stream the giant projections.json and pull the full sample rows we plot.

A model's ``analysis/projections.json`` is hundreds of MB to ~2 GB, so we never
parse it whole. For the answer token we locate the byte span of each (layer,
position) cell with a header regex, then for the WANTED depth layers we parse
every sample object in that cell (each carries its 2D + 3D PCA coordinates plus
all ~20 colour-by fields) and the explained variance of the first three PCA axes.
Depth is ``layer / n_layers``; each target depth maps to the nearest layer.

Used by ``depth_scatter_panel_render.py`` to draw a 2D + 3D scatter per depth,
recoloured by every axis in the shared ``COLOR_AXES`` registry.
"""

from __future__ import annotations

import bisect
import json
import mmap
import re
from dataclasses import dataclass, field
from pathlib import Path

from sesgo.geometry.geometry_color_axes import COLOR_AXES
from src.common.base_schema import BaseSchema

# One source of truth for how deep each model is (depth = layer / n_layers).
MODEL_LAYERS: dict[str, int] = {"Qwen3-0.6B": 28, "Qwen3-32B": 64}
TARGET_DEPTHS: tuple[float, ...] = (0.5, 0.7, 0.9)
# The answer token: the state right where the model commits to its answer.
POSITION = "label"
# Every colour-by field name we keep per sample (DRY: the shared registry).
AXIS_KEYS: tuple[str, ...] = tuple(a.key for a in COLOR_AXES)


@dataclass
class DepthCell(BaseSchema):
    """One (model, depth) cell: every sample's 2D/3D coords + all colour columns."""

    model: str = ""
    depth: float = 0.0
    layer: str = ""
    n_layers: int = 0
    pc_var: list = field(default_factory=list)  # [PC1%, PC2%, PC3%] of spread.
    coords2d: list = field(default_factory=list)  # [[x, y], ...]
    coords3d: list = field(default_factory=list)  # [[x, y, z], ...]
    # One column per colour-by axis: {axis_key: [value per sample, ...]} (1D dict).
    columns: dict = field(default_factory=dict)


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
_EVR = re.compile(
    rb'"explained_variance_ratio": \[\s*([\-0-9eE.]+),\s*([\-0-9eE.]+),\s*([\-0-9eE.]+)')
# One sample object: a flat brace-bounded blob ending at its coord3d array.
_SAMPLE = re.compile(rb'\{[^{}]*?"coord3d": \[[^\]]*\]\s*\}', re.S)


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
    """Parse one cell's 3-axis variance + every sample's coords and colour columns."""
    evr = _EVR.search(region)
    samples = region[: region.find(b'"scaffold_stats"')]
    coords2d, coords3d = [], []
    columns: dict[str, list] = {k: [] for k in AXIS_KEYS}
    for m in _SAMPLE.finditer(samples):
        row = json.loads(m.group(0))
        coords2d.append(row["coord2d"])
        coords3d.append(row["coord3d"])
        for key in AXIS_KEYS:
            columns[key].append(row.get(key))
    return DepthCell(
        pc_var=[float(evr.group(i)) * 100 for i in (1, 2, 3)],
        coords2d=coords2d, coords3d=coords3d, columns=columns)
