"""The non-thinking (teacher-forced) risk readout for one prompt.

BinaryChoiceRunner.choose teacher-forces past an empty <think></think> block,
so this captures the model's *immediate* preference between the two answer
options with no reasoning. We keep both raw divergent logprobs and the labels
they correspond to so the calibrated probability can be re-derived or audited
without re-running the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema


@dataclass
class NonThinkingResult(BaseSchema):
    """Calibrated P(at risk) plus the raw two-way logprob evidence."""

    predicted_risk: float  # softmax P(at-risk option) in [0, 1]
    choice_idx: int  # 0/1 of the chosen label, or -1 if tied
    logprob_positive: float  # logprob of the at-risk option
    logprob_negative: float  # logprob of the not-at-risk option
    positive_label: str  # label token scored as "at risk"
    negative_label: str  # label token scored as "not at risk"
