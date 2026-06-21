"""One prompt's risk readout PLUS the residual-stream snapshots taken on it.

This is the geometry-half analogue of RiskAssessmentSample: it carries the same
flat color-by axes (subject/disorder/framing/language) and the same two model
readouts (non_thinking / thinking), and adds ``activations`` — the structural
residual snapshots captured while greedily decoding the binary answer. The tensors
themselves live on disk (see RiskGeometryActivation.path); this schema only
references them, so it stays light and serializes through BaseSchema.

Unlike SESGO (gold is always UNKNOWN), the risk gold is a continuous
``gold_risk`` and the color-by axis of interest is ``framing`` (the intervention
axis), so there is no abstention-accuracy property here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema
from src.datasets.prompt import RiskTaskType
from src.datasets.risk import NonThinkingResult, ScoreSummary
from .risk_geometry_activation import RiskGeometryActivation


@dataclass
class RiskGeometrySample(BaseSchema):
    """A risk judgement plus its captured residual geometry for one prompt."""

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
    activations: list[RiskGeometryActivation] = field(default_factory=list)

    @property
    def predicted_risk_non_thinking(self) -> float | None:
        """Calibrated P(at risk) from the teacher-forced readout, if present."""
        return self.non_thinking.predicted_risk if self.non_thinking else None

    @property
    def predicted_risk_thinking(self) -> float | None:
        """Mean of the sampled reasoning-based risk scores; None when no draw."""
        if self.thinking is None or self.thinking.n == 0:
            return None
        return self.thinking.mean
