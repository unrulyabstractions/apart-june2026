"""The two ways we elicit a risk judgement from a model."""

from __future__ import annotations

from enum import Enum


class RiskTaskType(Enum):
    """How the model is asked to express risk.

    SCORE asks for a continuous number; CATEGORIZE asks for a discrete
    choice between two labels. They drive different downstream parsing
    (number extraction vs. label selection), so they are first-class.
    """

    SCORE = "score"
    CATEGORIZE = "categorize"
