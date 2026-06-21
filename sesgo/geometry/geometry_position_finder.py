"""Locate the fine-grained structural token positions in a forced sequence.

The geometry probe captures the residual stream at chat-template boundary tokens.
Splitting the coarse "turn" boundary into its constituent tokens and the coarse
"answer" into prefix+label lets the viz differentiate, for example, the role
token from the line break that follows it. Markers are MODEL-AWARE via
``runner.structural_markers`` (Qwen <|im_start|>/<think>, Llama
<|start_header_id|>, Gemma <start_of_turn>, Mistral [/INST]); any marker a family
lacks is simply omitted so the caller can log + skip that position.

Locating is robust to multi-token role words (matched by their first token after
the turn opener) and to leading-space BPE merges at the answer (the ``label`` and
``answer_prefix`` indices are taken directly from ``answer_start`` rather than
re-tokenizing the answer span).
"""

from __future__ import annotations

from src.ternary_choice import TernaryChoiceRunner


def _single_token_id(runner: TernaryChoiceRunner, text: str) -> int | None:
    """Token id for ``text`` iff it encodes to exactly one (special) token."""
    ids = runner.encode_ids(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def _last_index(ids: list[int], target: int | None) -> int | None:
    """Index of the LAST occurrence of ``target`` in ``ids`` (None if absent)."""
    if target is None:
        return None
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == target:
            return i
    return None


def _marker_position(runner: TernaryChoiceRunner, ids: list[int], marker: str) -> int | None:
    """Last index of the single-token ``marker`` in ``ids`` (None if absent/multi)."""
    if not marker:
        return None
    return _last_index(ids, _single_token_id(runner, marker))


def _first_after(ids: list[int], start: int | None, target: int | None) -> int | None:
    """First index > ``start`` holding ``target`` (None if either is absent)."""
    if start is None or target is None:
        return None
    for i in range(start + 1, len(ids)):
        if ids[i] == target:
            return i
    return None


def _assistant_role_position(
    runner: TernaryChoiceRunner, ids: list[int], turn: int | None, role: str
) -> int | None:
    """First token index of the assistant role word, searched AFTER the turn opener.

    The role word ("assistant"/"model") may tokenize to >1 token, so we match its
    FIRST token id appearing after ``turn``. None when the family has no role word
    or it cannot be located.
    """
    if not role:
        return None
    role_ids = runner.encode_ids(role, add_special_tokens=False)
    return _first_after(ids, turn, role_ids[0] if role_ids else None)


def find_positions(
    runner: TernaryChoiceRunner, ids: list[int], answer_start: int
) -> dict[str, int]:
    """Locate the fine-grained structural token positions in the forced sequence.

    ``im_start`` is the LAST assistant-turn opener; ``im_end`` the previous turn's
    closer; ``assistant`` the role word and ``newline`` the line break, both AFTER
    the opener. think_open/close exist only for reasoning models. The answer
    marker is appended LAST: its first token is ``label`` (at ``answer_start``) and
    the token before it is ``answer_prefix``. Missing positions are omitted.
    """
    markers = runner.structural_markers
    found: dict[str, int] = {}

    turn = _marker_position(runner, ids, markers.turn_marker)
    if turn is not None:
        found["im_start"] = turn

    for ptype, idx in (
        ("im_end", _marker_position(runner, ids, markers.turn_end)),
        ("newline", _first_after(ids, turn, _single_token_id(runner, "\n"))),
        ("assistant", _assistant_role_position(runner, ids, turn, markers.assistant_role)),
        ("think_open", _marker_position(runner, ids, markers.think_open)),
        ("think_close", _marker_position(runner, ids, markers.think_close)),
    ):
        if idx is not None:
            found[ptype] = idx

    if 0 < answer_start <= len(ids):
        found["answer_prefix"] = answer_start - 1
    if 0 <= answer_start < len(ids):
        found["label"] = answer_start
    return found


def answer_start_for(runner: TernaryChoiceRunner, head: str, forced: str) -> int:
    """First index where the realized sequence diverges from the prefix-only one."""
    ids = runner.encode_ids(forced, add_special_tokens=True)
    head_ids = runner.encode_ids(head, add_special_tokens=True)
    return next(
        (i for i in range(min(len(head_ids), len(ids))) if head_ids[i] != ids[i]),
        len(head_ids),
    )
