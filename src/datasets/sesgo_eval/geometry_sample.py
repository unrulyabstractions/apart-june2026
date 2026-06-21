"""One prompt's SESGO readout PLUS the residual-stream snapshots taken on it.

This is the geometry-half analogue of SesgoSample: it carries the same flat
color-by axes and the same two model readouts (non_thinking / thinking), and
adds ``activations`` — the list of structural residual snapshots captured while
teacher-forcing the answer. The tensors themselves live on disk (see
GeometryActivation.path); this schema only references them, so it stays light
and serializes through BaseSchema like every other sample type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema
from src.datasets.sesgo import SesgoLabel
from .geometry_activation import GeometryActivation
from .sesgo_non_thinking import SesgoNonThinking
from .sesgo_thinking import SesgoThinking


@dataclass
class GeometrySample(BaseSchema):
    """A SESGO judgement plus its captured residual geometry for one prompt."""

    sample_idx: int
    question_id: str
    scaffold_id: str | None
    bias_category: str
    question_polarity: str
    language: str
    gold_label: SesgoLabel
    prompt_text: str
    # Every per-sample colour-by axis the geometry viz funnels the representation
    # through: label_style + provenance (bbq) + the literal social-group strings.
    label_style: str = ""
    bbq: bool = False
    target_identity: str = ""
    other_identity: str = ""
    non_thinking: SesgoNonThinking | None = None
    thinking: SesgoThinking | None = None
    activations: list[GeometryActivation] = field(default_factory=list)

    @property
    def predicted_non_thinking(self) -> SesgoLabel | None:
        """Argmax role from the teacher-forced readout, if present."""
        return self.non_thinking.predicted if self.non_thinking else None

    @property
    def predicted_thinking(self) -> SesgoLabel | None:
        """Argmax role over parsed draws; None when no draw parsed (n == 0)."""
        if self.thinking is None or self.thinking.sample_size == 0:
            return None
        return self.thinking.predicted

    @property
    def correct_non_thinking(self) -> bool:
        """True iff the non-thinking prediction is UNKNOWN (the ambiguous gold)."""
        return self.predicted_non_thinking is SesgoLabel.UNKNOWN
