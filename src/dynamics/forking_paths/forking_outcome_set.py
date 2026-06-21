"""The categorical OUTCOME set a forking-paths analysis maps rollouts onto.

In forking-paths (Bigelow 2024) the outcome function R maps a rollout to a
one-hot over the answer set. For an ambiguous SESGO item the answers are the
three roles [TARGET, OTHER, UNKNOWN] plus a catch-all OTHER_OUTCOME bucket for
unparseable / off-set draws (the paper's "Other" category). This object fixes
the label order so every O_t histogram in a trajectory is the same length and
indexes the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema
from src.datasets.sesgo import SesgoLabel

# Catch-all bucket label for rollouts whose answer cannot be parsed to a role.
UNPARSEABLE_OUTCOME = "unparseable"


@dataclass
class ForkOutcomeSet(BaseSchema):
    """Ordered categorical outcome labels + the index of each in the histogram.

    ``labels`` is the canonical order; the SESGO roles come first (TARGET, OTHER,
    UNKNOWN) followed by the catch-all UNPARSEABLE bucket. ``dim`` is the vector
    length every OutcomeHistogram in the trajectory uses.
    """

    labels: list[str] = field(
        default_factory=lambda: [
            SesgoLabel.TARGET.value,
            SesgoLabel.OTHER.value,
            SesgoLabel.UNKNOWN.value,
            UNPARSEABLE_OUTCOME,
        ]
    )

    @property
    def dim(self) -> int:
        """Number of outcome categories (histogram length)."""
        return len(self.labels)

    def index_of(self, label: str) -> int:
        """Histogram index of an outcome label (UNPARSEABLE bucket if unknown)."""
        if label in self.labels:
            return self.labels.index(label)
        return self.labels.index(UNPARSEABLE_OUTCOME)

    def one_hot(self, label: str | None) -> list[float]:
        """One-hot R-vector for a parsed outcome label (None -> UNPARSEABLE)."""
        vec = [0.0] * self.dim
        key = label if label is not None else UNPARSEABLE_OUTCOME
        vec[self.index_of(key)] = 1.0
        return vec
