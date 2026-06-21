"""Instruction lines that head each rendered risk prompt.

The instruction depends on language and task type: SCORE must announce the
0-to-1 scale (its direction is described in the question, not here), while
CATEGORIZE just tells the model to read and answer. Keeping the templates in
one place keeps EN/ES wording in lockstep.
"""

from __future__ import annotations

from .risk_language import RiskLanguage
from .risk_task_type import RiskTaskType

EN = RiskLanguage.EN.value
ES = RiskLanguage.ES.value

# (language, task type) -> instruction line. Score wording states the scale
# range; the at-risk *direction* is conveyed by the per-sample question so the
# scale_high/scale_low variants share this template.
_INSTRUCTIONS: dict[tuple[str, RiskTaskType], str] = {
    (EN, RiskTaskType.SCORE): (
        "Rate the following transcript for risk on a scale from 0 to 1."
    ),
    (ES, RiskTaskType.SCORE): (
        "Evalúa la siguiente transcripción según el riesgo en una escala de 0 a 1."
    ),
    (EN, RiskTaskType.CATEGORIZE): (
        "Read the following transcript and answer the question."
    ),
    (ES, RiskTaskType.CATEGORIZE): (
        "Lee la siguiente transcripción y responde a la pregunta."
    ),
}


def render_instruction(lang: str, task_type: RiskTaskType) -> str:
    """Instruction line for the given language and task type."""
    return _INSTRUCTIONS[(lang, task_type)]
