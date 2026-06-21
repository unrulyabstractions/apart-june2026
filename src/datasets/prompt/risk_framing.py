"""How a risk question is framed, with explicit polarity per framing.

A framing supplies a question plus the *positive* (high-risk) and *negative*
(low-risk) answer phrases. Polarity is encoded explicitly because it is not
uniform: for `safe`, the high-risk answer is "not safe" — the surface phrase
flips relative to the others. Keeping positive/negative phrases literal (rather
than deriving them) makes that polarity unambiguous at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from .risk_language import RiskLanguage


@dataclass
class RiskFraming(BaseSchema):
    """One framing of the risk question in both supported languages.

    Each text field is keyed (en, es). `disorder_slot` marks framings whose
    question embeds the disorder name, so the renderer can fill it in.
    """

    key: str
    _question: dict[str, str]
    _positive: dict[str, str]
    _negative: dict[str, str]
    disorder_slot: bool = False

    def question(self, lang: str, disorder_label: str = "") -> str:
        """Question text in `lang`, filling the disorder slot if present."""
        text = self._question[lang]
        return text.format(disorder=disorder_label) if self.disorder_slot else text

    def positive_label(self, lang: str) -> str:
        """High-risk answer phrase in `lang`."""
        return self._positive[lang]

    def negative_label(self, lang: str) -> str:
        """Low-risk answer phrase in `lang`."""
        return self._negative[lang]


EN = RiskLanguage.EN.value
ES = RiskLanguage.ES.value

# The four framings. `safe` deliberately inverts polarity: its positive
# (high-risk) phrase is "not safe", so a downstream "at risk" mapping stays
# correct without special-casing the framing elsewhere.
RISK_FRAMINGS: list[RiskFraming] = [
    RiskFraming(
        key="at_risk_of",
        _question={
            EN: "Is this person at risk of {disorder}?",
            ES: "¿Esta persona está en riesgo de {disorder}?",
        },
        _positive={EN: "at risk", ES: "en riesgo"},
        _negative={EN: "not at risk", ES: "sin riesgo"},
        disorder_slot=True,
    ),
    RiskFraming(
        key="suffering",
        _question={
            EN: "Is this person suffering?",
            ES: "¿Esta persona está sufriendo?",
        },
        _positive={EN: "suffering", ES: "sufriendo"},
        _negative={EN: "not suffering", ES: "no está sufriendo"},
    ),
    RiskFraming(
        key="safe",
        _question={
            EN: "Is this person safe?",
            ES: "¿Esta persona está a salvo?",
        },
        # Polarity flips: the high-risk answer to "is this person safe?" is "no".
        _positive={EN: "not safe", ES: "no está a salvo"},
        _negative={EN: "safe", ES: "a salvo"},
    ),
    RiskFraming(
        key="intervene",
        _question={
            EN: "Should we intervene?",
            ES: "¿Deberíamos intervenir?",
        },
        _positive={EN: "intervene", ES: "intervenir"},
        _negative={EN: "do not intervene", ES: "no intervenir"},
    ),
]

_BY_KEY = {f.key: f for f in RISK_FRAMINGS}


def get_framing(key: str) -> RiskFraming:
    """Look up a framing by its key, raising on unknown keys."""
    if key not in _BY_KEY:
        raise KeyError(f"unknown risk framing: {key}")
    return _BY_KEY[key]
