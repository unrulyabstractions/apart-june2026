"""Adaptively pick the most-informative INTEGER layer for the 2D PCA scatters.

The static PCA scatters (``pca_by_*`` / ``pca_scatter_*``) must show the depth
where the structure is CLEAREST, not a hard-coded absolute index. Absolute index
27 is the last layer of a 28-layer model but only ~0.75 depth of a 36-layer one,
so it is not comparable across model sizes.

This module reads the silhouette separability ALREADY stored per (layer, position)
in ``projections.json`` (no recompute) and selects, at the answer-``label``
position, the captured mid->last layer that MAXIMISES the scaffold silhouette —
the depth where the interpretive structure is most cleanly separated. Selecting by
structure makes the chosen RELATIVE DEPTH comparable across model sizes.

The captured band is ``floor(L/2)..L-1`` (see collect_geometry_samples.py), so the
deepest captured integer layer is the model's last layer ``L-1`` and the model's
total layer count is ``max_int_layer + 1`` — enough to report a relative depth
(e.g. "layer 31/36 (0.86 depth)") without the model config.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema

# Layers whose scaffold silhouette is within this of the max are treated as tied;
# the tie-break then prefers the DEEPER layer (later representations crystallise
# the task), so we never pick a needlessly shallow layer over a near-equal deeper one.
SILHOUETTE_TIE_EPSILON = 0.01


@dataclass
class FeatureLayer(BaseSchema):
    """The adaptively chosen scatter layer + its model-relative depth.

    ``layer_key`` is the string layer key into ``results`` (so the caller indexes
    ``results[layer_key]`` directly). ``total_layers`` is the model's full layer
    count inferred from the deepest captured integer layer; ``relative_depth`` is
    ``layer / total_layers`` in [0, 1]. ``by_silhouette`` records whether the pick
    came from the stored silhouettes or fell back to the last captured layer.
    """

    layer_key: str
    layer_index: int
    total_layers: int
    relative_depth: float
    silhouette: float | None
    by_silhouette: bool

    def title_suffix(self) -> str:
        """Compact "layer 31/36 (0.86 depth)" tag for plot titles/subtitles.

        Degrades to "layer=<key>" for a non-integer fallback view (e.g. "mean")
        where a model-relative depth is undefined.
        """
        if self.total_layers <= 0:
            return f"layer={self.layer_key}"
        return (f"layer {self.layer_index}/{self.total_layers} "
                f"({self.relative_depth:.2f} depth)")


def _int_layer_keys(results: dict) -> list[str]:
    """The captured INTEGER layer keys (mid->last), ascending by depth."""
    return sorted((L for L in results if L.lstrip("-").isdigit()), key=int)


def _scaffold_silhouette(results: dict, layer_key: str, position: str) -> float | None:
    """The stored scaffold silhouette at one (layer, position), or None if absent."""
    block = results.get(layer_key, {}).get(position)
    if block is None:
        return None
    return block.get("scaffold_stats", {}).get("silhouette")


def _make(layer_key: str, total_layers: int, sil: float | None, by_sil: bool) -> FeatureLayer:
    """Assemble a FeatureLayer, computing relative depth from the total layer count."""
    idx = int(layer_key)
    depth = idx / total_layers if total_layers else 0.0
    return FeatureLayer(layer_key, idx, total_layers, depth, sil, by_sil)


def select_feature_layer(results: dict, position: str = "label") -> FeatureLayer | None:
    """Pick the captured layer with the CLEAREST scaffold structure at ``position``.

    Among the captured mid->last integer layers, choose the one whose stored
    scaffold silhouette at ``position`` is maximal; on a near-tie (within
    SILHOUETTE_TIE_EPSILON of the max) prefer the DEEPER layer. Falls back to the
    deepest captured layer when no silhouette is scorable, and returns None only
    when ``results`` has no integer layer at all (the caller then degrades to its
    own fallback). Reads only the precomputed silhouettes — never recomputes PCA.
    """
    int_keys = _int_layer_keys(results)
    if not int_keys:
        return None
    total_layers = int(int_keys[-1]) + 1  # captured band ends at the model's last layer

    scored = [(k, _scaffold_silhouette(results, k, position)) for k in int_keys]
    valid = [(k, s) for k, s in scored if s is not None]
    if not valid:
        # No separability anywhere -> fall back to the deepest captured layer.
        return _make(int_keys[-1], total_layers, None, by_sil=False)

    best = max(s for _k, s in valid)
    # Tie-break: among near-max layers, prefer the deepest (int_keys is ascending).
    tied = [k for k, s in valid if best - s <= SILHOUETTE_TIE_EPSILON]
    chosen_key = tied[-1]
    chosen_sil = next(s for k, s in valid if k == chosen_key)
    return _make(chosen_key, total_layers, chosen_sil, by_sil=True)
