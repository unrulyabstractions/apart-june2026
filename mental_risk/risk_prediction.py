"""Pick the single comparable predicted risk off a RiskAssessmentSample.

The two readout levels are not both available for every prompt: SCORE prompts
have no binary non-thinking readout (the first token of a free number is a
useless anchor), so only their thinking score exists; CATEGORIZE prompts have the
cheap non-thinking calibrated risk. The stability and selection visualizers need
ONE number per sample to compare, so they share this resolver: prefer the cheap
non-thinking risk when present, else fall back to the sampled thinking mean. Used
by every per-study visualizer instead of re-deriving the precedence each time.
"""

from __future__ import annotations

from src.datasets.risk import RiskAssessmentSample


def effective_risk(sample: RiskAssessmentSample) -> float | None:
    """The comparable predicted risk: non-thinking if present, else thinking."""
    if sample.predicted_risk_non_thinking is not None:
        return sample.predicted_risk_non_thinking
    return sample.predicted_risk_thinking
