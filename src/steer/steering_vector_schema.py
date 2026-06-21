"""Schemas for the per-layer SESGO steering vectors and their train/test split.

A steering vector at layer L is the diff-of-means direction
``mean(resid_scaffold - resid_noscaffold)`` taken over the TRAIN contrastive
pairs and the four change-of-turn structural positions (im_end / newline /
im_start / assistant). Adding ``alpha * v[L]`` to ``blocks.L.hook_resid_post``
is the causal inverse of the geometry capture.

Everything stays flat per CLAUDE.md: a vector is one ``LayerSteeringVector``
(layer + a 1-D ``list[float]``), and the bundle is a ``list`` of those plus flat
scalar metadata. The seeded split is two flat ``list[str]`` of question_ids so
Run and Verify reuse the EXACT same held-out items.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema


@dataclass
class LayerSteeringVector(BaseSchema):
    """One layer's diff-of-means steering direction (a flat ``[d_model]`` list)."""

    layer: int
    vector: list[float]  # length d_model; mean(scaffold - no_scaffold) over train
    norm: float  # L2 norm of ``vector`` (the captured magnitude, pre-normalize)
    n_terms: int  # train pairs x change-of-turn positions averaged into it


@dataclass
class SteeringVectorBundle(BaseSchema):
    """All per-layer steering vectors for one model, plus the reproducible split.

    ``vectors`` holds one ``LayerSteeringVector`` per captured layer;
    ``primary_layer`` is the mid-depth layer whose scaffold silhouette peaks (the
    default steering layer). ``train_question_ids`` / ``test_question_ids`` are the
    seeded split over the 231 contrastive pairs — saved here so the test driver
    steers ONLY on held-out items the vectors never saw.
    """

    model: str
    d_model: int
    seed: int
    train_fraction: float
    primary_layer: int
    primary_layer_depth: float  # primary_layer / total_layers (relative depth)
    change_of_turn_positions: list[str] = field(default_factory=list)
    train_question_ids: list[str] = field(default_factory=list)
    test_question_ids: list[str] = field(default_factory=list)
    vectors: list[LayerSteeringVector] = field(default_factory=list)

    def vector_for(self, layer: int) -> LayerSteeringVector | None:
        """The steering vector captured at ``layer``, or None if absent."""
        return next((v for v in self.vectors if v.layer == layer), None)
