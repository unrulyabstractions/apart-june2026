"""Parse a [0, 1] risk score out of a free-form thinking generation.

The thinking level lets the model reason naturally, so its reply is prose, not
a forced token. We strip the reasoning block first (the score lives in the
final answer, not the scratch-work) then parse per task type: SCORE yields a
number we orient by the prompt's scale direction; CATEGORIZE yields which of
the two labels the model landed on, mapped to the at-risk endpoint. Returns
None when nothing parseable is present so the caller can drop the draw.
"""

from __future__ import annotations

import re

from src.datasets.prompt import RiskPromptSample, RiskTaskType

# A decimal in [0, 1]: optional leading 0, a fraction, or a bare 0/1.
_NUMBER = re.compile(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b")


def _answer_after_thinking(text: str) -> str | None:
    """Return the model's answer = whatever follows the closed reasoning block.

    A reasoning model's score lives in the final answer, never the scratch-work,
    so we must read only what comes after ``</think>``. Returns None when a
    ``<think>`` block was opened but never closed: the model ran out of budget
    mid-reasoning and emitted no answer, so the draw must be dropped rather than
    mined for a stray number in the reasoning.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return None  # truncated mid-thought; no answer to parse
    return text  # non-reasoning model: the whole reply is the answer


def _parse_score(answer: str, sample: RiskPromptSample) -> float | None:
    """First number in [0, 1], inverted when the scale was flipped (0 == at risk)."""
    for match in _NUMBER.finditer(answer):
        value = float(match.group(1))
        if 0.0 <= value <= 1.0:
            # scale_high is False ⇒ 0 is the at-risk end, so flip to a risk score.
            return 1.0 - value if sample.scale_high is False else value
    return None


def _parse_categorize(answer: str, sample: RiskPromptSample) -> float | None:
    """Detect the chosen label; 1.0 if it is the at-risk option else 0.0.

    Match the label token (e.g. "(A)") or its phrase, case-insensitively; the
    earliest mention wins so a trailing restatement cannot override the answer.
    """
    pos, neg = sample.positive_label, sample.negative_label
    if pos is None or neg is None:
        return None
    text = answer.lower()
    pos_at = min((text.find(t.lower()) for t in (pos,) if t.lower() in text), default=-1)
    neg_at = min((text.find(t.lower()) for t in (neg,) if t.lower() in text), default=-1)
    if pos_at == -1 and neg_at == -1:
        return None
    if neg_at == -1 or (pos_at != -1 and pos_at <= neg_at):
        return 1.0
    return 0.0


def parse_risk_score(generated_text: str, sample: RiskPromptSample) -> float | None:
    """Extract an oriented risk score in [0, 1] from one thinking generation."""
    answer = _answer_after_thinking(generated_text)
    if not answer or not answer.strip():
        return None
    if sample.task_type is RiskTaskType.SCORE:
        return _parse_score(answer.strip(), sample)
    return _parse_categorize(answer.strip(), sample)
