"""Snapshot residual-stream geometry along a risk prompt's answer path.

The reusable engine behind the geometry driver: given a BinaryChoiceRunner forced
onto the HuggingFace backend (the only one exposing run_with_cache), it follows
the model's GREEDY NON-THINKING answer (greedy decode past an empty
<think></think> block) and snapshots the FULL per-layer residual stream at four
structural token positions (turn / think_open / think_close / answer).

This mirrors SESGO's in-driver capture_activations, but lives in the library so
the driver stays thin and the position-finding logic is shared. It is risk-aware
only in that it takes a RiskPromptSample's text + choice_prefix; the residual
geometry itself is task-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.binary_choice.binary_choice_runner import BinaryChoiceRunner
from src.datasets.prompt import RiskPromptSample
from .risk_geometry_activation import RiskGeometryActivation

# Structural positions we look for, in capture order. think_* exist only for
# reasoning models (skip-thinking prefix); missing ones are logged and skipped.
POSITION_TYPES = ("turn", "think_open", "think_close", "answer")

# Tokens to greedily decode for the non-thinking answer path. We only need the
# answer token (the first generated token); a few extra give a stable sequence.
_GREEDY_TOKENS = 24


def _single_token_id(runner: BinaryChoiceRunner, text: str) -> int | None:
    """Token id for ``text`` iff it encodes to exactly one (special) token.

    Special markers like <|im_start|> / <think> are single tokens in the Qwen
    vocab; we encode WITHOUT special tokens so we read the literal id and can
    search for it in the forced sequence. Returns None when it isn't single.
    """
    ids = runner.encode_ids(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def _resid_filter(layers: list[int] | None):
    """names_filter selecting per-layer residual-stream (resid_post) hooks."""

    def keep(name: str) -> bool:
        if "resid_post" not in name:
            return False
        if layers is None:
            return True
        return any(name == f"blocks.{layer}.hook_resid_post" for layer in layers)

    return keep


def _stack_resid(cache: dict, layers: list[int], pos: int) -> torch.Tensor:
    """Stack the residual vector at token ``pos`` across layers -> [n_layers, d_model]."""
    vecs = []
    for layer in layers:
        # cache tensors are [batch, seq, d_model]; batch is always 1 here.
        act = cache[f"blocks.{layer}.hook_resid_post"]
        vecs.append(act[0, pos].detach().float().cpu())
    return torch.stack(vecs, dim=0)


def _last_index(ids: list[int], target: int | None) -> int | None:
    """Index of the LAST occurrence of ``target`` in ``ids`` (None if absent)."""
    if target is None:
        return None
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == target:
            return i
    return None


def find_positions(
    runner: BinaryChoiceRunner, ids: list[int], answer_start: int
) -> dict[str, int]:
    """Locate the four structural token positions in the forced id sequence.

    The assistant turn is the LAST <|im_start|>; think_open/close are the
    <think>/</think> tokens. The answer marker is appended LAST, so its first
    token sits at ``answer_start`` (the length of the prompt+prefix prefix). Missing
    positions are omitted so the caller can log + skip them.
    """
    found: dict[str, int] = {}
    turn = _last_index(ids, _single_token_id(runner, "<|im_start|>"))
    if turn is not None:
        found["turn"] = turn
    open_i = _last_index(ids, _single_token_id(runner, "<think>"))
    if open_i is not None:
        found["think_open"] = open_i
    close_i = _last_index(ids, _single_token_id(runner, "</think>"))
    if close_i is not None:
        found["think_close"] = close_i
    if 0 <= answer_start < len(ids):
        found["answer"] = answer_start
    return found


def _greedy_answer_ids(
    runner: BinaryChoiceRunner, prompt: RiskPromptSample
) -> tuple[list[int], int]:
    """Build the forced id sequence on the greedy answer path + the answer index.

    Returns (ids, answer_start): the chat-templated prompt + skip-thinking prefix
    + choice prefix + the model's own deterministic (temperature 0) continuation,
    with answer_start the first index where it diverges from the prefix-only ids
    (robust to leading-space BPE merges, " a" != "a").
    """
    prefix = prompt.choice_prefix or "Answer: "
    head = runner.apply_chat_template(prompt.text) + runner.skip_thinking_prefix + prefix
    greedy = runner.generate(
        prompt.text,
        max_new_tokens=_GREEDY_TOKENS,
        temperature=0.0,
        prefilling=runner.skip_thinking_prefix + prefix,
    )
    ids = runner.encode_ids(head + greedy, add_special_tokens=True)
    head_ids = runner.encode_ids(head, add_special_tokens=True)
    answer_start = next(
        (i for i in range(min(len(head_ids), len(ids))) if head_ids[i] != ids[i]),
        len(head_ids),
    )
    return ids, answer_start


def capture_activations(
    runner: BinaryChoiceRunner,
    prompt: RiskPromptSample,
    layers: list[int],
    sample_dir: Path,
    rel_root: Path,
) -> tuple[list[RiskGeometryActivation], list[str]]:
    """Snapshot residuals along the model's GREEDY NON-THINKING answer path.

    Runs one forward pass over the realized greedy sequence and snapshots the
    per-layer residual stream at the structural positions. Returns the captured
    RiskGeometryActivations plus the position types NOT found.
    """
    ids, answer_start = _greedy_answer_ids(runner, prompt)
    _, cache = runner._backend.run_with_cache(
        torch.tensor([ids], device=runner.device), names_filter=_resid_filter(layers)
    )
    positions = find_positions(runner, ids, answer_start)
    captured: list[RiskGeometryActivation] = []
    for ptype in POSITION_TYPES:
        pos = positions.get(ptype)
        if pos is None:
            continue
        resid = _stack_resid(cache, layers, pos)  # [n_layers, d_model]
        fname = f"sample_{prompt.sample_idx}_{ptype}.pt"
        torch.save(resid, sample_dir / fname)
        captured.append(
            RiskGeometryActivation(
                position_type=ptype,
                token_position=pos,
                token_text=runner.decode_ids([ids[pos]]),
                path=str((sample_dir / fname).relative_to(rel_root)),
            )
        )
    missing = [p for p in POSITION_TYPES if p not in positions]
    return captured, missing
