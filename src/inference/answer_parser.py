"""Parse a model's response into the answer it COMMITTED to — (label, choice, offset).

label  = the option MARKER the model chose (e.g. "c)"), or "" if it never committed.
choice = the role that marker maps to via position_labels (target/other/unknown),
         or "invalid" when the model gave no parseable final answer.
offset = char offset of the committed marker in response_text (for the readout), or -1.

Design (the whole point): the model is INSTRUCTED to end with an explicit final-answer
cue ("Respuesta final: <letra>"). We read the marker right after the LAST such cue.
We do NOT scan for the first option marker — verbose models echo the option list while
reasoning, and a thinking model that runs out of tokens mid-thought has NO answer at
all. In both cases the correct result is `invalid`, never a guess.
"""

from __future__ import annotations

import re

INVALID = "invalid"
_THINK_CLOSE = "</think>"

# Final-answer cues, strongest first. Matched case-insensitively. The marker (or bare
# letter) immediately following the LAST occurrence is the committed answer.
_CUES = [
    r"respuesta\s+final\s*:?",
    r"final\s+answer\s*:?",
    r"respuesta\s+seleccionada\s*:?",
    r"respuesta\s+correcta\s+es\s*:?",
    r"the\s+answer\s+is\s*:?",
    r"\brespuesta\s*:",
    r"\banswer\s*:",
]


def answer_segment(response_text: str) -> str:
    """The part of the response AFTER any reasoning block (where the answer lives)."""
    i = response_text.rfind(_THINK_CLOSE)
    return response_text[i + len(_THINK_CLOSE):] if i != -1 else response_text


def _marker_at(text: str, start: int, option_labels) -> tuple[str, int]:
    """Earliest option marker in text[start:start+40], as (marker, abs_offset) or ("", -1)."""
    window = text[start:start + 40]
    best: tuple[str, int] | None = None
    for m in option_labels:
        p = window.find(m)
        if p != -1 and (best is None or p < best[1]):
            best = (m, p)
    return (best[0], start + best[1]) if best else ("", -1)


def find_committed_answer(response_text: str, option_labels) -> tuple[str, int]:
    """The marker after the LAST final-answer cue, as (marker, char_offset). ("", -1) if
    the model never committed (no cue, or a cue with no option marker after it)."""
    seg = answer_segment(response_text)
    base = len(response_text) - len(seg)
    # Try cues strongest-first; within a cue, use its LAST occurrence (the final word).
    for cue in _CUES:
        last = None
        for mtch in re.finditer(cue, seg, flags=re.IGNORECASE):
            last = mtch
        if last is not None:
            marker, off = _marker_at(seg, last.end(), option_labels)
            if marker:
                return marker, base + off
    # No cue at all: accept ONLY a terse direct answer — a short reply (no rambling CoT)
    # whose marker sits at the very start, e.g. "c)" or "c) No hay…". Otherwise invalid.
    terse = seg.strip()
    if len(terse) <= 60:
        marker, off = _marker_at(terse, 0, option_labels)
        if marker and off <= 4:
            return marker, base + seg.find(terse) + off
    return "", -1


def parse_answer(response_text: str, option_labels, position_labels) -> tuple[str, str, int]:
    """Return (label, choice, char_offset). 'invalid'/-1 when no answer was committed."""
    label, off = find_committed_answer(response_text, option_labels)
    if not label:
        return "", INVALID, -1
    role = position_labels[list(option_labels).index(label)]
    choice = role.value if hasattr(role, "value") else role
    return label, choice, off
