"""Shared residual-geometry capture primitives (single-sample + batched paths).

Locating structural token positions and snapshotting the per-layer residual
stream is identical whether one prompt or a chunk is processed; both the
single-sample driver and the batched driver import these so the captured geometry
is byte-identical across paths. Markers are MODEL-AWARE via
``runner.structural_markers`` (Qwen <|im_start|>/<think>, Llama
<|start_header_id|>, Gemma <start_of_turn>, Mistral [/INST]).
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.datasets.prompt import SesgoPromptSample
from src.datasets.sesgo_eval import GeometryActivation
from src.ternary_choice import TernaryChoiceRunner

# Structural positions we look for, in capture order. think_* exist only for
# reasoning models (skip-thinking prefix); missing ones are logged and skipped.
_POSITION_TYPES = ("turn", "think_open", "think_close", "answer")

# Tokens to greedily decode for the non-thinking answer path. We only need the
# answer token (the first generated token); a few extra give a stable sequence.
_GREEDY_TOKENS = 24


def _single_token_id(runner: TernaryChoiceRunner, text: str) -> int | None:
    """Token id for ``text`` iff it encodes to exactly one (special) token."""
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


def _marker_position(runner: TernaryChoiceRunner, ids: list[int], marker: str) -> int | None:
    """Last index of the single-token ``marker`` in ``ids`` (None if absent/multi)."""
    if not marker:
        return None
    return _last_index(ids, _single_token_id(runner, marker))


def find_positions(
    runner: TernaryChoiceRunner, ids: list[int], answer_start: int
) -> dict[str, int]:
    """Locate the structural token positions in the forced id sequence.

    The assistant turn is the LAST turn marker; think_open/close are present only
    for reasoning models. The answer marker is appended LAST, so its first token
    sits at ``answer_start`` (we use that index directly rather than re-tokenizing,
    which would miss leading-space BPE merges). Missing positions are omitted so
    the caller can log + skip.
    """
    markers = runner.structural_markers
    found: dict[str, int] = {}

    turn = _marker_position(runner, ids, markers.turn_marker)
    if turn is not None:
        found["turn"] = turn

    open_i = _marker_position(runner, ids, markers.think_open)
    if open_i is not None:
        found["think_open"] = open_i
    close_i = _marker_position(runner, ids, markers.think_close)
    if close_i is not None:
        found["think_close"] = close_i

    if 0 <= answer_start < len(ids):
        found["answer"] = answer_start
    return found


def answer_start_for(runner: TernaryChoiceRunner, head: str, forced: str) -> int:
    """First index where the realized sequence diverges from the prefix-only one."""
    ids = runner.encode_ids(forced, add_special_tokens=True)
    head_ids = runner.encode_ids(head, add_special_tokens=True)
    return next(
        (i for i in range(min(len(head_ids), len(ids))) if head_ids[i] != ids[i]),
        len(head_ids),
    )


def save_positions(
    runner: TernaryChoiceRunner,
    ids: list[int],
    cache: dict,
    positions: dict[str, int],
    layers: list[int],
    sample_idx: int,
    sample_dir: Path,
    rel_root: Path,
) -> tuple[list[GeometryActivation], list[str]]:
    """Persist each located structural residual stack; return captures + missing."""
    captured: list[GeometryActivation] = []
    for ptype in _POSITION_TYPES:
        pos = positions.get(ptype)
        if pos is None:
            continue
        resid = _stack_resid(cache, layers, pos)  # [n_layers, d_model]
        fname = f"sample_{sample_idx}_{ptype}.pt"
        torch.save(resid, sample_dir / fname)
        captured.append(
            GeometryActivation(
                position_type=ptype,
                token_position=pos,
                token_text=runner.decode_ids([ids[pos]]),
                path=str((sample_dir / fname).relative_to(rel_root)),
            )
        )
    missing = [p for p in _POSITION_TYPES if p not in positions]
    return captured, missing


def capture_activations(
    runner: TernaryChoiceRunner,
    prompt: SesgoPromptSample,
    layers: list[int],
    sample_dir: Path,
    rel_root: Path,
) -> tuple[list[GeometryActivation], list[str]]:
    """Snapshot residuals along ONE prompt's greedy non-thinking answer path."""
    prefix = prompt.choice_prefix or "Answer: "
    head = runner.apply_chat_template(prompt.text) + runner.skip_thinking_prefix + prefix
    greedy = runner.generate(
        prompt.text,
        max_new_tokens=_GREEDY_TOKENS,
        temperature=0.0,
        prefilling=runner.skip_thinking_prefix + prefix,
    )
    forced = head + greedy
    ids = runner.encode_ids(forced, add_special_tokens=True)
    answer_start = answer_start_for(runner, head, forced)

    _, cache = runner._backend.run_with_cache(
        torch.tensor([ids], device=runner.device), names_filter=_resid_filter(layers)
    )
    positions = find_positions(runner, ids, answer_start)
    return save_positions(
        runner, ids, cache, positions, layers, prompt.sample_idx, sample_dir, rel_root
    )
