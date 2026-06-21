"""One prompt's two-level SESGO readout, with provenance to interpret it.

Each sample carries every color-by axis as a flat field (so analyses can slice
by question, scaffold, polarity, category, language, label style without
re-joining) plus up to two model readouts: a non-thinking 3-way distribution
(teacher-forced positions remapped to roles) and a thinking distribution
(counted over parsed free-form draws). The ambiguous-context gold is UNKNOWN, so
"correct" means the model abstained rather than picking a group. Raw thinking
completions are heavy and not part of identity, so the `_` prefix drops them
from both the id hash and to_dict.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.datasets.sesgo import SesgoLabel
from .sesgo_label_distribution import SesgoLabelDistribution


@dataclass
class SesgoSample(BaseSchema):
    """SESGO judgement(s) for a single rendered prompt."""

    sample_idx: int
    question_id: str
    scaffold_id: str | None
    question_polarity: str
    bias_category: str
    language: str
    label_style: str
    gold_label: SesgoLabel
    prompt_text: str
    non_thinking: SesgoLabelDistribution | None = None
    thinking: SesgoLabelDistribution | None = None
    # Heavy/private: raw generations are excluded from identity and to_dict.
    _thinking_completions: list[str] | None = None

    @property
    def predicted_non_thinking(self) -> SesgoLabel | None:
        """Argmax role from the teacher-forced readout, if present."""
        return self.non_thinking.predicted_label if self.non_thinking else None

    @property
    def predicted_thinking(self) -> SesgoLabel | None:
        """Argmax role over the parsed draws; None when no draw parsed (n == 0)."""
        if self.thinking is None or self.thinking.n == 0:
            return None
        return self.thinking.predicted_label

    @property
    def correct_non_thinking(self) -> bool:
        """True iff the non-thinking prediction is UNKNOWN (the ambiguous gold)."""
        return self.predicted_non_thinking is SesgoLabel.UNKNOWN

    @property
    def correct_thinking(self) -> bool:
        """True iff the thinking prediction is UNKNOWN (the ambiguous gold)."""
        return self.predicted_thinking is SesgoLabel.UNKNOWN
