"""Discover and lazily load the geometry data for every captured model.

The interactive viz server is multi-model: instead of binding to a single
``response_samples.json`` at startup, it scans ``out/sesgo/geometry/*/`` for
every model that has BOTH a ``response_samples.json`` (the GeometryDataset) and
a sibling ``analysis/projections.json`` (the PCA projection written by
analyze_geometry.py). Each such directory is one switchable model.

WHY a registry (not eager dict-of-dicts in the server): keeps the server file
focused on routing, gives one place that owns the "what does a loadable model
look like" rule, and lets us load each model's ~6 MB blobs lazily on first
selection so startup stays instant no matter how many models are present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.datasets.sesgo_eval import GeometryDataset, GeometrySample

# Layout written by collect/analyze: <root>/<MODEL>/{response_samples.json,
# analysis/projections.json}. Both must exist for a model to be switchable.
SAMPLES_NAME = "response_samples.json"
PROJECTIONS_REL = Path("analysis") / "projections.json"


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN/Inf) with None.

    analyze_geometry.py writes literal ``NaN`` for un-computable bootstrap CI
    bounds (Python's json.dump permits them), but strict JSON — and therefore
    both FastAPI's JSONResponse and the browser's JSON.parse — rejects them.
    Mapping them to ``null`` is semantically faithful (the CI was undefined) and
    the frontend already guards every CI read against null.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


class GeometryModel:
    """One switchable model: its on-disk paths + lazily loaded geometry data."""

    def __init__(self, name: str, model_dir: Path) -> None:
        self.name = name
        self.model_dir = model_dir
        self.samples_path = model_dir / SAMPLES_NAME
        self.projections_path = model_dir / PROJECTIONS_REL
        # Loaded on first access so adding more models never slows startup.
        self._projections: dict | None = None
        self._by_idx: dict[int, GeometrySample] | None = None

    @property
    def projections(self) -> dict:
        """The parsed projections.json (loaded + cached on first access)."""
        if self._projections is None:
            with open(self.projections_path) as f:
                self._projections = _json_safe(json.load(f))
        return self._projections

    @property
    def by_idx(self) -> dict[int, GeometrySample]:
        """sample_idx -> GeometrySample, for O(1) per-sample detail lookups."""
        if self._by_idx is None:
            dataset = GeometryDataset.from_json(self.samples_path)
            self._by_idx = {s.sample_idx: s for s in dataset.samples}
        return self._by_idx


def discover_models(geometry_root: Path) -> dict[str, GeometryModel]:
    """Map model_name -> GeometryModel for every loadable dir under ``root``.

    A directory qualifies only when both the samples file and the projections
    file exist; half-finished captures are silently skipped. Returned ordering
    is alphabetical so the frontend selector is stable across restarts.
    """
    if not geometry_root.is_dir():
        return {}
    models: dict[str, GeometryModel] = {}
    for child in sorted(geometry_root.iterdir()):
        if not child.is_dir():
            continue
        m = GeometryModel(child.name, child)
        if m.samples_path.exists() and m.projections_path.exists():
            models[child.name] = m
    return models


def default_model_name(models: dict[str, GeometryModel], preferred: str | None) -> str:
    """Pick the initially selected model: ``preferred`` if present, else the
    primary Qwen3-0.6B, else the first Qwen, else the first model.

    Discovery is alphabetical, which booted the UI into Llama-3.1-70B (the
    sparsest model, fewest axes) instead of the rich primary model. Prefer Qwen.
    """
    if preferred and preferred in models:
        return preferred
    if "Qwen3-0.6B" in models:
        return "Qwen3-0.6B"
    qwen = next((m for m in models if m.startswith("Qwen")), None)
    return qwen or next(iter(models))
