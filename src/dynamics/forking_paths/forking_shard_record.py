"""The partial-trajectory record ONE fleet box writes for its position shard.

Shard k of N forks only positions ``[t for t in range(P) if t % N == k]`` and
writes a ``ForkingShard`` carrying just those ``ForkPosition``s (with their REAL
index t). Only shard 0 also computes the prior o_0 (``prior_histogram`` is empty on
every other shard). The merge driver concatenates all shards' positions by real t,
validates 0..P-1 is covered exactly once, and rebuilds a full ``ForkingTrajectory``
the plotter consumes. Every field is a flat scalar / 1D list of BaseSchema, so it
roundtrips cleanly with no nested dict/list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema

from .forking_path_types import ForkPosition


@dataclass
class ForkingShard(BaseSchema):
    """One fleet box's slice of a forking trajectory (its forked positions).

    ``shard_index`` / ``num_shards`` identify the slice; ``outcome_labels`` mirror
    ``ForkOutcomeSet.labels`` (the histogram order); ``prior_histogram`` is o_0 and
    is non-empty ONLY for shard 0. ``positions`` are this shard's ``ForkPosition``s
    keyed by their REAL base-path index, ready to be merged by index.
    """

    shard_index: int
    num_shards: int
    model: str
    item_question_id: str
    prompt_text: str
    base_path_text: str
    base_token_texts: list[str]
    prompt_token_count: int
    outcome_labels: list[str]
    prior_histogram: list[float] = field(default_factory=list)
    positions: list[ForkPosition] = field(default_factory=list)
