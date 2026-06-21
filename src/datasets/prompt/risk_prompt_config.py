"""Configuration selecting which slices of the risk grid to generate.

Every axis defaults to "all", so a bare `RiskPromptConfig(name=...)` produces
the full cross product. Toggles are stored as explicit value lists (not flags)
so the config both documents and reproduces a dataset deterministically. Source
selection (disorders, limit) is recorded here for provenance; loading the
subjects themselves stays the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema
from .risk_framing import RISK_FRAMINGS
from .risk_label_style import get_risk_label_styles
from .risk_language import RiskLanguage
from .risk_task_type import RiskTaskType


def _all_framings() -> list[str]:
    return [f.key for f in RISK_FRAMINGS]


def _all_task_types() -> list[RiskTaskType]:
    return list(RiskTaskType)


def _all_label_styles() -> list[tuple[str, str]]:
    return get_risk_label_styles()


def _all_languages() -> list[str]:
    return [lang.value for lang in RiskLanguage]


@dataclass
class RiskPromptConfig(BaseSchema):
    """Which framings, tasks, label styles, languages and flips to include."""

    name: str
    framings: list[str] = field(default_factory=_all_framings)
    task_types: list[RiskTaskType] = field(default_factory=_all_task_types)
    label_styles: list[tuple[str, str]] = field(default_factory=_all_label_styles)
    languages: list[str] = field(default_factory=_all_languages)
    order_flips: list[bool] = field(default_factory=lambda: [False, True])
    # SCORE scale directions: scale_high (1 = at risk) and scale_low (0 = at risk).
    scale_highs: list[bool] = field(default_factory=lambda: [True, False])
    choice_prefix_en: str = "Answer: "
    choice_prefix_es: str = "Respuesta: "
    # Source provenance only; the caller loads subjects with these.
    disorders: list[str] = field(default_factory=list)
    limit: int | None = None

    def get_filename(self) -> str:
        """Filename for saving a dataset built from this config."""
        return f"{self.name}_{self.get_id()}.json"

    def choice_prefix(self, lang: str) -> str:
        """The localized answer prefix used to cue the model's choice."""
        return self.choice_prefix_es if lang == RiskLanguage.ES.value else self.choice_prefix_en
