"""A single rendered risk-assessment prompt and everything to interpret it.

This object is self-contained: a downstream querier never re-derives anything.
For CATEGORIZE it has the `labels` to pass to `runner.choose(text, choice_prefix,
labels)` plus `positive_idx` (which label index means "at risk"). For SCORE it
has `scale_high` (whether 1 or 0 is the at-risk end) so a parsed number can be
oriented. `gold_risk` is the subject's true risk for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from .risk_task_type import RiskTaskType


@dataclass
class RiskPromptSample(BaseSchema):
    """One prompt in the risk grid, with its parsing metadata."""

    sample_idx: int
    text: str
    subject_id: str
    disorder: str
    gold_risk: float | None
    framing: str
    task_type: RiskTaskType
    language: str
    labels: tuple[str, str] | None = None  # categorize only: (labelA, labelB)
    positive_idx: int | None = None  # which label index is the at-risk answer
    choice_prefix: str = ""
    scale_high: bool | None = None  # score only: True if 1 == at risk
    label_flipped: bool = False  # whether option order was flipped

    @property
    def positive_label(self) -> str | None:
        """The label token mapped to the at-risk answer (categorize only)."""
        if self.labels is None or self.positive_idx is None:
            return None
        return self.labels[self.positive_idx]

    @property
    def negative_label(self) -> str | None:
        """The label token mapped to the not-at-risk answer (categorize only)."""
        if self.labels is None or self.positive_idx is None:
            return None
        return self.labels[1 - self.positive_idx]
