"""Schemas for the held-out steering test: abstention vs steering strength.

The causal experiment sweeps the steering strength ``alpha`` (including 0 = the
unsteered baseline and a negative control) on the held-out TEST ambiguous items
WITHOUT a scaffold in the prompt, measuring how much abstention (UNKNOWN mass)
the +v hook induces. One ``SweepPoint`` per alpha aggregates the per-item
metrics; the ``ScaffoldReference`` is the same readout on the ACTUAL scaffold
prompt (the behaviour +v is trying to reproduce). Everything is flat per
CLAUDE.md: lists of scalar-only BaseSchema rows, no nested structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema


@dataclass
class SweepPoint(BaseSchema):
    """Aggregate abstention at one steering strength over the held-out test set."""

    alpha: float  # steering strength (0 = unsteered baseline; <0 = control)
    n_items: int
    mean_unknown_prob: float  # mean teacher-forced UNKNOWN probability
    abstain_rate: float  # fraction of items whose argmax role is UNKNOWN
    delta_unknown_prob: float  # mean_unknown_prob minus the alpha=0 baseline


@dataclass
class ScaffoldReference(BaseSchema):
    """Unsteered readout on the ACTUAL scaffold prompt (the target behaviour)."""

    n_items: int
    mean_unknown_prob: float
    abstain_rate: float


@dataclass
class SteeringTestResult(BaseSchema):
    """Full held-out steering test: the alpha sweep plus the scaffold reference."""

    model: str
    layer: int
    normalize: bool  # whether v was unit-normalized (alpha = absolute magnitude)
    seed: int
    n_test_pairs: int
    n_ambiguous_test_items: int
    alphas: list[float] = field(default_factory=list)
    sweep: list[SweepPoint] = field(default_factory=list)
    scaffold_reference: ScaffoldReference = field(default_factory=ScaffoldReference)
