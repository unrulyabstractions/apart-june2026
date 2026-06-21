"""Thinking SESGO summary: mean/std of the per-draw role indicators.

The thinking level samples N free-form draws and parses the role each committed
to. We summarize those parsed picks as a per-role mean (fraction of draws that
chose the role) and a per-role std (population std of the one-hot indicator
across draws). Both are length-3 vectors in canonical order [TARGET, OTHER,
UNKNOWN]; `sample_size` is the number of PARSED draws backing them.

Alongside the role distribution we keep ``vocab_entropies``: ONE value per draw
(over ALL draws, parsed or not) holding that generation's MEAN next-token Shannon
entropy (nats). The divergence study reads its distribution across the ~100 CoT
draws to relate a draw's per-token uncertainty to which outcome it commits to.

Kept a clean BaseSchema (two length-3 vectors, a per-draw entropy list, and a
count), so it roundtrips cheaply and carries honest dispersion alongside the
central tendency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema
from src.common.math import AggregationMethod, aggregate
from src.datasets.sesgo import SesgoLabel

# Canonical role order every vector follows; argmax ties resolve to UNKNOWN.
_ROLE_ORDER = (SesgoLabel.TARGET, SesgoLabel.OTHER, SesgoLabel.UNKNOWN)


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Population mean and std of a flat float list (empty → (0, 0))."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = aggregate(values, AggregationMethod.MEAN)  # reuse src.common.math
    var = sum((x - m) ** 2 for x in values) / n
    return float(m), float(var**0.5)


@dataclass
class SesgoThinking(BaseSchema):
    """Per-role mean/std over parsed thinking draws, ordered [TARGET, OTHER, UNKNOWN]."""

    mean: list[float]  # fraction of draws that chose each role
    std: list[float]  # population std of the per-draw one-hot indicator
    sample_size: int  # number of PARSED draws (Nones dropped by caller)
    # ONE mean next-token Shannon entropy (nats) per draw, over ALL draws (parsed
    # or not). Empty when the sampler did not capture per-token entropy.
    vocab_entropies: list[float] = field(default_factory=list)

    @property
    def predicted(self) -> SesgoLabel:
        """Argmax-mean role; a non-unique max → UNKNOWN (indecisive)."""
        top = max(self.mean)
        if self.mean.count(top) > 1:
            return SesgoLabel.UNKNOWN
        return _ROLE_ORDER[self.mean.index(top)]

    @property
    def mean_vocab_entropy(self) -> float:
        """Mean per-token vocab entropy across the draws (0 when none captured)."""
        return _mean_std(self.vocab_entropies)[0]

    @property
    def std_vocab_entropy(self) -> float:
        """Population std of per-draw vocab entropy — its spread across draws."""
        return _mean_std(self.vocab_entropies)[1]


def summarize_labels(
    labels: list[SesgoLabel], vocab_entropies: list[float] | None = None
) -> SesgoThinking:
    """Summarize parsed per-draw picks into per-role mean/std + sample size.

    ``labels`` is one parsed role per draw (caller already dropped the Nones).
    For each role we build the one-hot indicator across draws; its mean is the
    pick fraction and its std is the honest POPULATION std of those indicators
    (== sqrt(p*(1-p)) for a Bernoulli, but computed directly so it stays honest
    even if the harness ever changes). Empty input → sample_size 0, zero vectors.

    ``vocab_entropies`` (optional) is the per-draw mean next-token entropy over
    ALL draws — carried through verbatim so the ~N-draw entropy distribution is
    available downstream (its length is the TOTAL draw count, not sample_size).
    """
    ents = list(vocab_entropies or [])
    n = len(labels)
    if n == 0:
        return SesgoThinking(
            mean=[0.0, 0.0, 0.0], std=[0.0, 0.0, 0.0], sample_size=0,
            vocab_entropies=ents,
        )

    mean: list[float] = []
    std: list[float] = []
    for role in _ROLE_ORDER:
        indicators = [1.0 if pick is role else 0.0 for pick in labels]
        m = aggregate(indicators, AggregationMethod.MEAN)  # reuse src.common.math
        # Population variance of the indicator, then sqrt → honest population std.
        var = sum((x - m) ** 2 for x in indicators) / n
        mean.append(m)
        std.append(var**0.5)

    return SesgoThinking(mean=mean, std=std, sample_size=n, vocab_entropies=ents)
