"""Compute per-layer diff-of-means steering vectors from contrastive pairs.

For each captured layer L, the steering direction is the mean over (TRAIN pairs x
the four change-of-turn positions im_end/newline/im_start/assistant) of the
per-snapshot difference ``resid_scaffold - resid_noscaffold``. This is the causal
inverse of the geometry capture: adding ``alpha * v[L]`` to resid_post at layer L
pushes a no-scaffold forward toward where the scaffold moved the representation.

Averaging over the four chat-template boundary positions (not the think/answer
positions) targets the structural turn boundaries the scaffold reshapes, and
pooling across the 4 positions x many pairs gives a low-variance direction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.math import l2_norm
from .contrastive_pair_index import ContrastivePair
from .geometry_residual_loader import load_residual
from .steering_vector_schema import LayerSteeringVector

# The four chat-template boundary positions the scaffold reshapes (the steering
# spec's "change-of-turn" set; the other four are think/answer positions).
CHANGE_OF_TURN_POSITIONS = ("im_end", "newline", "im_start", "assistant")


def _pair_deltas(
    root: Path, pair: ContrastivePair, layer: int, positions: tuple[str, ...]
) -> list[np.ndarray]:
    """All (scaffold - no_scaffold) deltas for one pair across the positions.

    A position contributes only when BOTH members expose its (position, layer)
    snapshot (they always do in the balanced dataset; the guard keeps it robust).
    """
    deltas: list[np.ndarray] = []
    for pos in positions:
        scaf = load_residual(root, pair.scaffold, pos, layer)
        base = load_residual(root, pair.no_scaffold, pos, layer)
        if scaf is None or base is None:
            continue
        deltas.append(scaf - base)
    return deltas


def steering_vector_for_layer(
    root: Path,
    pairs: list[ContrastivePair],
    layer: int,
    positions: tuple[str, ...] = CHANGE_OF_TURN_POSITIONS,
) -> LayerSteeringVector | None:
    """Diff-of-means steering vector at one layer over the given train pairs.

    Returns None when no (pair, position) snapshot was available at this layer.
    """
    deltas: list[np.ndarray] = []
    for pair in pairs:
        deltas.extend(_pair_deltas(root, pair, layer, positions))
    if not deltas:
        return None
    mean_delta = np.mean(np.stack(deltas), axis=0).astype(np.float32)
    return LayerSteeringVector(
        layer=layer,
        vector=[float(x) for x in mean_delta],
        norm=l2_norm(mean_delta.tolist()),
        n_terms=len(deltas),
    )


def steering_vectors_all_layers(
    root: Path,
    pairs: list[ContrastivePair],
    layers: list[int],
    positions: tuple[str, ...] = CHANGE_OF_TURN_POSITIONS,
) -> list[LayerSteeringVector]:
    """Per-layer diff-of-means vectors for every captured layer (skips empties)."""
    out: list[LayerSteeringVector] = []
    for layer in layers:
        vec = steering_vector_for_layer(root, pairs, layer, positions)
        if vec is not None:
            out.append(vec)
    return out
