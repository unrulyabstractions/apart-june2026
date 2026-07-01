"""Parse a model's response into the answer it committed to — (label, choice, offset).

NOTHING about the prompt format is hardcoded here. The caller passes the EXPECTED answer
prefix (`answer_cue` — exactly the final-answer prefix the prompt instructed the model to
emit), the option markers, and (optionally) the option TEXTS. The committed answer is the
option marker after the LAST occurrence of that cue in the ANSWER segment (everything after
the final </think>) — searched whole, since the model may state the answer first and then
explain. A cue mentioned mid-reasoning is excluded because the </think> split drops the
reasoning; a CoT cut off mid-thought (no answer segment / no cue) yields invalid.

Marker matching is tolerant of the ways models actually write the choice: the option letter
is matched case-insensitively, with or without its closing paren ("Y)", "z", "**Z)**"), and
as a fallback a response that simply STATES an option's text ("No hay suficiente informacion
...") is credited to that option.
"""

from __future__ import annotations

import re

INVALID = "invalid"
# Reasoning-scratch-pad close tokens across families: Qwen uses </think>, Mistral
# (Ministral/Magistral) uses [/THINK]. The answer lives after the LAST of either.
_THINK_CLOSE_MARKERS = ("</think>", "[/THINK]")

# Natural-language answer lead-ins to try when the model committed to an option but did NOT
# use the prompt's exact instructed prefix (e.g. answered "La respuesta es a) ..." instead of
# "Respuesta final: a)"). We take the option marker after the LAST such lead-in.
_ANSWER_LEADINS = (
    "respuesta final:", "respuesta correcta:", "la respuesta correcta es",
    "la respuesta es", "la mejor respuesta es", "la opción correcta es",
    "la opcion correcta es", "la opción es", "la opcion es", "mi respuesta es",
    "respuesta:", "the correct answer is", "the answer is", "final answer:", "answer:",
)
_LETTERS = "A-Za-zÁÉÍÓÚÑáéíóúñ"          # letter class incl. Spanish accents (for word boundaries)
_LEADING_JUNK = re.compile(r"^[\s*#>\-•\.]+")  # markdown / bullet prefix before a stated option


def answer_segment(response_text: str) -> str:
    """The part of the response AFTER the reasoning block (where the answer lives).

    We split at the FIRST close-think marker, not the last: a model that degenerates into
    repeating ``</think> Respuesta final: b) ...`` would otherwise leave only the trailing
    (empty) tail after the LAST ``</think>``, hiding a clearly committed answer."""
    best_end = -1
    for marker in _THINK_CLOSE_MARKERS:
        i = response_text.find(marker)
        if i != -1:
            end = i + len(marker)
            best_end = end if best_end == -1 else min(best_end, end)
    return response_text[best_end:] if best_end != -1 else response_text


def _marker_at(text: str, start: int, option_labels) -> tuple[str, int]:
    """Earliest option marker in text[start:start+40], case-insensitive, as (marker, abs_offset).

    Matches the option letter either with its closing paren/bracket ("a)", "Y)", "z]") or as a
    bare terminal token ("Respuesta final: c" -> "c" before end / EOS tag / newline), so an
    uppercased or paren-less choice is still recognised. ("", -1) when none is found."""
    window = text[start:start + 40]
    best: tuple[str, int] | None = None
    for m in option_labels:
        letter = re.escape(m.rstrip(").]").strip())
        if not letter:
            continue
        # (a) letter + closing paren/bracket, OR (b) bare letter as the terminal token.
        pat = re.compile(rf"(?<![{_LETTERS}])(?:{letter}[)\]]|{letter}(?=\s*(?:$|[<\n])))",
                         re.IGNORECASE)
        mm = pat.search(window)
        if mm and (best is None or mm.start() < best[1]):
            best = (m, mm.start())
    return (best[0], start + best[1]) if best else ("", -1)


def _distinct_markers(text: str, option_labels) -> int:
    """How many DISTINCT option markers (letter + paren, case-insensitive) appear in `text`.
    Used to tell a real leading answer ("z) ...") from an option-list echo ("a) .. b) .. c) ..")."""
    low = text.lower()
    n = 0
    for m in option_labels:
        letter = re.escape(m.rstrip(").]").strip())
        if letter and re.search(rf"(?<![{_LETTERS}]){letter}[)\]]", low):
            n += 1
    return n


def _leading_option(seg: str, option_labels, option_texts) -> str:
    """The option whose TEXT the answer segment leads with ("No hay suficiente informacion ...")
    -- the model stated its choice in prose. Requires an unambiguous, sufficiently long match."""
    s = _LEADING_JUNK.sub("", seg.strip()).lower()
    hit = ""
    for label, txt in zip(option_labels, option_texts or []):
        t = (txt or "").strip().lower()
        if len(t) >= 10 and s.startswith(t):
            if hit:
                return ""  # two options match -> ambiguous, decline
            hit = label
    return hit


def find_committed_answer(response_text: str, option_labels, answer_cue: str) -> tuple[str, int]:
    """Option marker after the LAST occurrence of `answer_cue` (or a natural-language lead-in)
    in the ANSWER segment, as (marker, char_offset); a terse marker-first answer as a fallback.
    ("", -1) when nothing was committed."""
    seg = answer_segment(response_text)
    base = len(response_text) - len(seg)
    low = seg.lower()

    # 1. The prompt's instructed cue takes precedence (its LAST occurrence).
    if answer_cue:
        idx = low.rfind(answer_cue.lower())
        if idx != -1:
            marker, off = _marker_at(seg, idx + len(answer_cue), option_labels)
            if marker:
                return marker, base + off
    # 2. Natural-language lead-ins ("La respuesta es a) …"): the LATEST lead-in occurrence
    #    that is actually followed by a marker. We scan EVERY occurrence, not just the last,
    #    because a model may repeat "La respuesta correcta es ..." with only the first copy
    #    carrying the letter (the rest followed by a quoted option name).
    best_pos, best = -1, ("", -1)
    for cue in _ANSWER_LEADINS:
        start = 0
        while True:
            i = low.find(cue, start)
            if i == -1:
                break
            marker, off = _marker_at(seg, i + len(cue), option_labels)
            if marker and i > best_pos:
                best_pos, best = i, (marker, base + off)
            start = i + 1
    if best[0]:
        return best
    # 3. The model led with its choice: a marker at the very start of the answer segment
    #    ("z) No hay …"), unless the start is an option-list echo (>1 distinct marker up front).
    lead = seg.lstrip()
    marker, off = _marker_at(lead, 0, option_labels)
    if marker and off <= 4 and _distinct_markers(lead[:160], option_labels) <= 1:
        return marker, base + (len(seg) - len(lead)) + off
    return "", -1


def parse_answer(response_text: str, option_labels, position_labels, answer_cue: str,
                 option_texts=None) -> tuple[str, str, int]:
    """Return (label, choice, char_offset). 'invalid'/-1 when no committed answer is found.
    `answer_cue` is the prompt's instructed final-answer prefix; `option_texts` (optional,
    aligned to `option_labels`) lets a prose answer that states an option's text be credited."""
    label, off = find_committed_answer(response_text, option_labels, answer_cue)
    if not label and option_texts:
        seg = answer_segment(response_text)
        lab = _leading_option(seg, option_labels, option_texts)
        if lab:
            label, off = lab, len(response_text) - len(seg)
    if not label:
        return "", INVALID, -1
    role = position_labels[list(option_labels).index(label)]
    choice = role.value if hasattr(role, "value") else role
    return label, choice, off
