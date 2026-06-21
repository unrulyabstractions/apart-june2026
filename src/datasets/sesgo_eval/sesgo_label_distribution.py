"""A model's 3-way distribution over the SESGO answer roles for one prompt.

Both eval levels collapse to the same shape: probability mass over the three
*meanings* (target / other / unknown), not the displayed positions. Non-thinking
gets this from a single teacher-forced softmax remapped through the prompt's
position->role tuple (n=1); thinking gets it by counting parsed free-form draws
and normalizing (n = #parsed). Kept a clean BaseSchema (three scalars + a count)
so it serializes and roundtrips cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.common.math import probs_to_logprobs, shannon_entropy
from src.datasets.sesgo import SesgoLabel


@dataclass
class SesgoLabelDistribution(BaseSchema):
    """P(target), P(other), P(unknown) plus the #draws backing it."""

    p_target: float
    p_other: float
    p_unknown: float
    n: int  # draws behind the distribution (1 for non-thinking, #parsed thinking)

    @property
    def as_tuple(self) -> tuple[float, float, float]:
        """(p_target, p_other, p_unknown), the canonical role order."""
        return (self.p_target, self.p_other, self.p_unknown)

    @property
    def predicted_label(self) -> SesgoLabel:
        """Argmax role; ties resolve to UNKNOWN (the ambiguous-context gold)."""
        probs = self.as_tuple
        top = max(probs)
        # A non-unique max is not a decisive bias signal, so fall back to UNKNOWN.
        if probs.count(top) > 1:
            return SesgoLabel.UNKNOWN
        return (SesgoLabel.TARGET, SesgoLabel.OTHER, SesgoLabel.UNKNOWN)[probs.index(top)]

    @property
    def entropy(self) -> float:
        """Shannon entropy (nats) of the 3-way role distribution.

        shannon_entropy takes logprobs, so convert the probs back first
        (consistent with the rest of src.common.math).
        """
        return float(shannon_entropy(probs_to_logprobs(list(self.as_tuple))))

    @classmethod
    def from_label_probs(
        cls, p_target: float, p_other: float, p_unknown: float, n: int
    ) -> SesgoLabelDistribution:
        """Build from an already-normalized 3-way distribution (non-thinking)."""
        return cls(p_target=p_target, p_other=p_other, p_unknown=p_unknown, n=n)

    @classmethod
    def from_counts(
        cls, c_target: int, c_other: int, c_unknown: int, n: int
    ) -> SesgoLabelDistribution:
        """Build from empirical role counts (thinking), normalizing by ``n``.

        ``n`` is the #parsed draws (== the count sum); a zero ``n`` means no draw
        parsed, so we emit an all-zero distribution the caller reads as "absent".
        """
        if n <= 0:
            return cls(0.0, 0.0, 0.0, 0)
        return cls(c_target / n, c_other / n, c_unknown / n, n)
