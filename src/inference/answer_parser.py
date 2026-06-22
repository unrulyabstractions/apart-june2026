"""Parse a model's response into the answer it committed to — (label, choice, offset).

NOTHING about the prompt format is hardcoded here. The caller passes the EXPECTED answer
prefix (`answer_cue` — exactly the final-answer prefix the prompt instructed the model to
emit, carried on each dataset record) and the option markers. The committed answer is the
option marker after the LAST occurrence of that cue in the ANSWER segment (everything after
the final </think>) — searched whole, since the model may state the answer first and then
explain. A cue mentioned mid-reasoning is excluded because the </think> split drops the
reasoning; a CoT cut off mid-thought (no answer segment / no cue) yields invalid.
"""

from __future__ import annotations

INVALID = "invalid"
# Reasoning-scratch-pad close tokens across families: Qwen uses </think>, Mistral
# (Ministral/Magistral) uses [/THINK]. The answer lives after the LAST of either.
_THINK_CLOSE_MARKERS = ("</think>", "[/THINK]")


def answer_segment(response_text: str) -> str:
    """The part of the response AFTER any reasoning block (where the answer lives)."""
    best_end = -1
    for marker in _THINK_CLOSE_MARKERS:
        i = response_text.rfind(marker)
        if i != -1:
            best_end = max(best_end, i + len(marker))
    return response_text[best_end:] if best_end != -1 else response_text


def _marker_at(text: str, start: int, option_labels) -> tuple[str, int]:
    """Earliest option marker in text[start:start+40], as (marker, abs_offset) or ("", -1)."""
    window = text[start:start + 40]
    best: tuple[str, int] | None = None
    for m in option_labels:
        p = window.find(m)
        if p != -1 and (best is None or p < best[1]):
            best = (m, p)
    return (best[0], start + best[1]) if best else ("", -1)


def find_committed_answer(response_text: str, option_labels, answer_cue: str) -> tuple[str, int]:
    """Option marker after the LAST occurrence of `answer_cue` in the ANSWER segment (after
    any </think>), as (marker, char_offset). The answer may lead the response and be followed
    by an explanation, so we search the whole answer segment — not just the tail. Mid-
    reasoning mentions are already excluded by the </think> stripping (and the Qwen
    force-close). Falls back to a terse direct answer; ("", -1) when nothing was committed."""
    seg = answer_segment(response_text)
    base = len(response_text) - len(seg)

    idx = seg.lower().rfind(answer_cue.lower())
    if idx != -1:
        marker, off = _marker_at(seg, idx + len(answer_cue), option_labels)
        if marker:
            return marker, base + off
    # Terse direct answer: a short segment whose marker is at the very start ("c)" / "c) …").
    terse = seg.strip()
    if len(terse) <= 60:
        marker, off = _marker_at(terse, 0, option_labels)
        if marker and off <= 4:
            return marker, base + seg.find(terse) + off
    return "", -1


def parse_answer(response_text: str, option_labels, position_labels, answer_cue: str) -> tuple[str, str, int]:
    """Return (label, choice, char_offset). 'invalid'/-1 when no committed answer in the
    expected format. `answer_cue` is the prompt's instructed final-answer prefix."""
    label, off = find_committed_answer(response_text, option_labels, answer_cue)
    if not label:
        return "", INVALID, -1
    role = position_labels[list(option_labels).index(label)]
    choice = role.value if hasattr(role, "value") else role
    return label, choice, off
