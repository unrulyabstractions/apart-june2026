"""Parse a model's greedy multiple-choice response into (label, choice).

Generic over the option markers + role mapping a caller passes in:
  label  = the option MARKER the model picked (e.g. "c)"), or "" if none recognised.
  choice = the role that marker maps to via `position_labels` (e.g. target / other /
           unknown), or "invalid" when no marker is found.

The model is instructed to answer with the option letter, so we look in the
post-thinking answer segment for the earliest option marker and decode it through
position_labels. Robust to surrounding markdown ("**c)**"), a leading answer cue
("Respuesta: c)"), and reasoning blocks (search after the last </think>).
"""

from __future__ import annotations

INVALID = "invalid"
_THINK_CLOSE = "</think>"


def answer_segment(response_text: str) -> str:
    """The part of the response AFTER any reasoning block (where the answer lives)."""
    i = response_text.rfind(_THINK_CLOSE)
    return response_text[i + len(_THINK_CLOSE):] if i != -1 else response_text


def find_label(response_text: str, option_labels) -> tuple[str, int]:
    """Earliest option marker in the answer segment + its char offset in the FULL
    response_text. Returns ("", -1) when no marker appears."""
    seg = answer_segment(response_text)
    base = len(response_text) - len(seg)
    best: tuple[str, int] | None = None
    for marker in option_labels:
        pos = seg.find(marker)
        if pos != -1 and (best is None or pos < best[1]):
            best = (marker, pos)
    if best is None:
        return "", -1
    return best[0], base + best[1]


def parse_answer(response_text: str, option_labels, position_labels) -> tuple[str, str, int]:
    """Return (label, choice, char_offset). choice is decoded via position_labels;
    'invalid' (offset -1) when no option marker is recognised."""
    label, off = find_label(response_text, option_labels)
    if not label:
        return "", INVALID, -1
    role = position_labels[list(option_labels).index(label)]
    choice = role.value if hasattr(role, "value") else role
    return label, choice, off
