"""Discover which layers the geometry capture saved, and pick the primary one.

Captured layers are read straight off the activations (14..27 for the 28-layer
Qwen3-0.6B). The PRIMARY steering layer defaults to the mid-depth layer where the
scaffold silhouette peaks — already identified by the geometry analysis and read
from ``analysis/projections.json`` via the existing ``select_feature_layer``. When
projections.json is absent we fall back to the layer nearest 0.50 relative depth.
"""

from __future__ import annotations

from pathlib import Path

from sesgo.geometry.geometry_feature_layer import select_feature_layer
from src.common.file_io import load_json
from src.datasets.sesgo_eval import GeometrySample

# Relative depth at which the scaffold structure peaks (the steering spec's ~0.50).
_TARGET_RELATIVE_DEPTH = 0.50
# Position whose silhouette the feature-layer pick is keyed on (the answer label).
_FEATURE_POSITION = "label"


def captured_layers(samples: list[GeometrySample]) -> list[int]:
    """Sorted unique transformer layers present in the captured activations."""
    layers = {a.layer for s in samples for a in s.activations}
    return sorted(layers)


def _depth_fallback_layer(layers: list[int]) -> tuple[int, float]:
    """Layer nearest _TARGET_RELATIVE_DEPTH, with its relative depth.

    Total layer count is inferred from the deepest captured layer (the captured
    band ends at the model's last layer), matching geometry_feature_layer.
    """
    total = max(layers) + 1
    best = min(layers, key=lambda L: abs(L / total - _TARGET_RELATIVE_DEPTH))
    return best, best / total


def primary_layer(root: Path, layers: list[int]) -> tuple[int, float]:
    """The primary steering layer + its relative depth.

    Prefers the geometry analysis's silhouette-peak layer (projections.json); on
    absence/parse-failure falls back to the captured layer nearest 0.50 depth.
    """
    projections = root / "analysis" / "projections.json"
    if projections.exists():
        results = load_json(projections).get("results", {})
        feature = select_feature_layer(results, _FEATURE_POSITION)
        if feature is not None and feature.layer_index in layers:
            return feature.layer_index, feature.relative_depth
    return _depth_fallback_layer(layers)
