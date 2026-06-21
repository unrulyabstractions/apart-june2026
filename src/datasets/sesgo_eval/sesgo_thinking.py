"""Thinking SESGO summary: mean/std of the per-draw role indicators.

The thinking level samples N free-form draws and parses the role each committed
to. We summarize those parsed picks as a per-role mean (fraction of draws that
chose the role) and a per-role std (population std of the one-hot indicator
across draws). Both are length-3 vectors in canonical order [TARGET, OTHER,
UNKNOWN]; `sample_size` is the number of PARSED draws backing them.

Kept a clean BaseSchema (two length-3 vectors + a count), so it roundtrips
cheaply and carries honest dispersion alongside the central tendency.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.common.math import AggregationMethod, aggregate
from src.datasets.sesgo import SesgoLabel

# Canonical role order every vector follows; argmax ties resolve to UNKNOWN.
_ROLE_ORDER = (SesgoLabel.TARGET, SesgoLabel.OTHER, SesgoLabel.UNKNOWN)


@dataclass
class SesgoThinking(BaseSchema):
    """Per-role mean/std over parsed thinking draws, ordered [TARGET, OTHER, UNKNOWN]."""

    mean: list[float]  # fraction of draws that chose each role
    std: list[float]  # population std of the per-draw one-hot indicator
    sample_size: int  # number of PARSED draws (Nones dropped by caller)

    @property
    def predicted(self) -> SesgoLabel:
        """Argmax-mean role; a non-unique max → UNKNOWN (indecisive)."""
        top = max(self.mean)
        if self.mean.count(top) > 1:
            return SesgoLabel.UNKNOWN
        return _ROLE_ORDER[self.mean.index(top)]


def summarize_labels(labels: list[SesgoLabel]) -> SesgoThinking:
    """Summarize parsed per-draw picks into per-role mean/std + sample size.

    ``labels`` is one parsed role per draw (caller already dropped the Nones).
    For each role we build the one-hot indicator across draws; its mean is the
    pick fraction and its std is the honest POPULATION std of those indicators
    (== sqrt(p*(1-p)) for a Bernoulli, but computed directly so it stays honest
    even if the harness ever changes). Empty input → sample_size 0, zero vectors.
    """
    n = len(labels)
    if n == 0:
        return SesgoThinking(mean=[0.0, 0.0, 0.0], std=[0.0, 0.0, 0.0], sample_size=0)

    mean: list[float] = []
    std: list[float] = []
    for role in _ROLE_ORDER:
        indicators = [1.0 if pick is role else 0.0 for pick in labels]
        m = aggregate(indicators, AggregationMethod.MEAN)  # reuse src.common.math
        # Population variance of the indicator, then sqrt → honest population std.
        var = sum((x - m) ** 2 for x in indicators) / n
        mean.append(m)
        std.append(var**0.5)

    return SesgoThinking(mean=mean, std=std, sample_size=n)
