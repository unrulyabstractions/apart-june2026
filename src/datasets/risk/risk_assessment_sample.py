"""One prompt's full two-level risk readout, with provenance to interpret it.

Each sample pairs the prompt's identity (subject, disorder, framing, gold risk)
with up to two model readouts: a non-thinking calibrated probability and a
thinking summary over many sampled generations. The raw thinking completions
are kept only for debugging — they are large and not part of the sample's
identity, so the `_` prefix drops them from both the id hash and to_dict.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.datasets.prompt import RiskTaskType
from .non_thinking_result import NonThinkingResult
from .score_summary import ScoreSummary


@dataclass
class RiskAssessmentSample(BaseSchema):
    """Risk judgement(s) for a single rendered prompt."""

    sample_idx: int
    subject_id: str
    disorder: str
    framing: str
    language: str
    task_type: RiskTaskType
    gold_risk: float | None
    prompt_text: str
    non_thinking: NonThinkingResult | None = None
    thinking: ScoreSummary | None = None
    # Heavy/private: raw generations are excluded from identity and to_dict.
    _thinking_completions: list[str] | None = None

    @property
    def predicted_risk_non_thinking(self) -> float | None:
        """Calibrated P(at risk) from the teacher-forced readout, if present."""
        return self.non_thinking.predicted_risk if self.non_thinking else None

    @property
    def predicted_risk_thinking(self) -> float | None:
        """Mean of the sampled reasoning-based risk scores.

        None when no draw parsed (``n == 0`` ⇒ NaN mean), so callers can drop
        the sample instead of poisoning aggregates with NaN.
        """
        if self.thinking is None or self.thinking.n == 0:
            return None
        return self.thinking.mean
