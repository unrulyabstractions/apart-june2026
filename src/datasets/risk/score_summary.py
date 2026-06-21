"""Summary statistics over a set of sampled risk scores.

The thinking level draws many independent risk estimates from the model; this
collapses that cloud into a single comparable record. Spread is described two
ways: a plain std (how far estimates sit from their mean) and an
entropy/diversity pair over the score *distribution* (how concentrated vs.
spread the probability mass is), which captures multimodality that std alone
misses. All statistics reuse src.common.math so the maths stays consistent
with the rest of the codebase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.common import BaseSchema
from src.common.math import (
    AggregationMethod,
    aggregate,
    normalize,
    probs_to_logprobs,
    q_diversity,
    shannon_entropy,
)


@dataclass
class ScoreSummary(BaseSchema):
    """Aggregate of N sampled risk scores in [0, 1]."""

    n: int
    mean: float
    std: float
    entropy: float
    diversity: float
    min_score: float
    max_score: float


def _std(scores: list[float], mean: float) -> float:
    """Population std via E[x^2]-E[x]^2, mirroring deviance_variance's reuse of
    the mean aggregator on squared values (clamped to avoid negative roundoff)."""
    mean_sq = aggregate([s * s for s in scores], AggregationMethod.MEAN)
    return math.sqrt(max(mean_sq - mean * mean, 0.0))


def _spread(scores: list[float]) -> tuple[float, float]:
    """Entropy and Hill diversity of the score *distribution*.

    normalize requires a non-negative, non-empty vector; a single score has a
    degenerate (zero-entropy) distribution, so short-circuit it.
    """
    if len(scores) < 2:
        return 0.0, 1.0
    probs = normalize([max(s, 0.0) for s in scores])
    logprobs = probs_to_logprobs(probs)
    return float(shannon_entropy(logprobs)), float(q_diversity(logprobs, 1.0))


def summarize_scores(scores: list[float]) -> ScoreSummary:
    """Collapse sampled risk scores into a ScoreSummary (NaN-safe on empty)."""
    if not scores:
        return ScoreSummary(0, float("nan"), 0.0, 0.0, 0.0, float("nan"), float("nan"))
    mean = aggregate(scores, AggregationMethod.MEAN)
    entropy, diversity = _spread(scores)
    return ScoreSummary(
        n=len(scores),
        mean=mean,
        std=_std(scores, mean),
        entropy=entropy,
        diversity=diversity,
        min_score=min(scores),
        max_score=max(scores),
    )
