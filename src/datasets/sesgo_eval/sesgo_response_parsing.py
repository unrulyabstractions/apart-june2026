"""Parse which SESGO role a free-form thinking generation chose.

The thinking level lets the model reason naturally, so its reply is prose, not
a forced token. We strip the reasoning block first (the answer lives after
``</think>``, never in the scratch-work) then find which displayed OPTION the
model committed to: its position marker (e.g. "b)"), the bare letter/number
without the paren, or — when no marker appears — the option's own text. The
chosen position is decoded into a SesgoLabel through the prompt's
``position_labels`` tuple. Returns None when nothing is detectable so the caller
can drop the draw.
"""

from __future__ import annotations

import re

from src.datasets.prompt import SesgoPromptSample
from src.datasets.sesgo import SesgoLabel


def _answer_after_thinking(text: str) -> str | None:
    """Return the answer = whatever follows the closed reasoning block.

    Returns None when ``<think>`` opened but never closed: the model ran out of
    budget mid-reasoning and emitted no answer, so the draw must be dropped
    rather than mined for a stray marker in the reasoning.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return None  # truncated mid-thought; no answer to parse
    return text  # non-reasoning model: the whole reply is the answer


def _option_texts(sample: SesgoPromptSample) -> tuple[str, str, str]:
    """Per-position authored option text, parsed from the rendered ``##options``.

    The generator renders one line ``{marker} {text}`` per displayed position in
    position order, so line i is position i — the same index ``position_labels``
    uses. Falls back to empty strings if the block can't be located.
    """
    texts: list[str] = []
    for marker in sample.option_labels:
        # Anchor on the exact marker so we lift only that position's option text.
        m = re.search(re.escape(marker) + r"\s*(.+)", sample.text)
        texts.append(m.group(1).strip() if m else "")
    return (texts[0], texts[1], texts[2])


def _bare_marker(marker: str) -> str:
    """The marker's letter/number with surrounding punctuation stripped (e.g. "b")."""
    return marker.strip().strip("().:]>").strip()


def _chosen_position(answer: str, sample: SesgoPromptSample) -> int | None:
    """Index (0..2) of the earliest-mentioned option, or None if undetectable.

    Tries the full marker first, then the bare letter/number as a standalone
    token, then the authored option text — the earliest hit across positions
    wins so a trailing restatement cannot override the committed answer.
    """
    low = answer.lower()
    texts = _option_texts(sample)
    best_pos, best_at = None, len(low) + 1
    for i, marker in enumerate(sample.option_labels):
        at = -1
        if marker.lower() in low:
            at = low.find(marker.lower())
        else:
            bare = _bare_marker(marker).lower()
            # \b...\b avoids matching the letter inside an unrelated word.
            bm = re.search(rf"(?<![a-z0-9]){re.escape(bare)}(?![a-z0-9])", low) if bare else None
            if bm:
                at = bm.start()
            elif texts[i] and texts[i].lower() in low:
                at = low.find(texts[i].lower())
        if at != -1 and at < best_at:
            best_pos, best_at = i, at
    return best_pos


def parse_chosen_label(
    generated_text: str, sample: SesgoPromptSample
) -> SesgoLabel | None:
    """Decode the role (target/other/unknown) one thinking draw committed to."""
    answer = _answer_after_thinking(generated_text)
    if not answer or not answer.strip():
        return None
    pos = _chosen_position(answer.strip(), sample)
    if pos is None:
        return None
    return sample.position_labels[pos]
