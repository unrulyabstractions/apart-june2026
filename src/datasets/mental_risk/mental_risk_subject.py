"""A MentalRiskES subject: their message timeline plus gold risk labels."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema
from .mental_risk_disorder import Disorder
from .mental_risk_message import MentalRiskMessage
from .risk_label_collapse import collapse_risk


@dataclass
class MentalRiskSubject(BaseSchema):
    """One subject's data: ordered messages and the collapsed-risk gold labels.

    `labels` holds the raw gold columns (e.g. `bs`, `rbs`); `risk` collapses
    them into a single score so callers need not know the column scheme.
    """

    subject_id: str
    disorder: Disorder
    edition: str = "master"
    split: str = "train"
    task: str = ""
    messages: list[MentalRiskMessage] = field(default_factory=list)
    labels: dict[str, float] = field(default_factory=dict)

    @property
    def transcript(self) -> str:
        """Full timeline as text, messages ordered by their `id_message`."""
        ordered = sorted(self.messages, key=lambda m: m.id_message)
        return "\n".join(m.message for m in ordered)

    @property
    def risk(self) -> float | None:
        """Single continuous risk score in [0, 1], or None if unlabelled."""
        return collapse_risk(self.labels)

    @property
    def n_messages(self) -> int:
        """Number of messages in this subject's timeline."""
        return len(self.messages)
