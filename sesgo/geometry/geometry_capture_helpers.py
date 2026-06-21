"""Shared residual-geometry capture primitives (single-sample + batched paths).

Snapshotting the residual stream at the structural token positions is identical
whether one prompt or a chunk is processed; both the single-sample driver and the
batched driver import these so the captured geometry is byte-identical across
paths. Locating the positions themselves lives in ``geometry_position_finder``.

POSITIONS captured (fine-grained chat-template boundary, in sequence order):

    im_end        - the special token CLOSING the previous (user) turn
                    (Qwen <|im_end|>, Llama <|eot_id|>, Gemma <end_of_turn>)
    newline       - the literal "\\n" token right after the assistant opener
    im_start      - the assistant-turn opener (Qwen <|im_start|>, Llama
                    <|start_header_id|>, Gemma <start_of_turn>, Mistral [/INST])
    assistant     - the role word the template emits ("assistant"/"model")
    think_open    - the <think> token (reasoning models only)
    think_close   - the </think> token (reasoning models only)
    answer_prefix - the token JUST BEFORE the emitted answer (end of the forced
                    "Answer: " prefix; i.e. answer_start - 1)
    label         - the first GENERATED answer-label token (answer_start)

Markers a family lacks (think_* for non-reasoning models, im_end/assistant for
Mistral) are simply omitted so the caller can log + skip them.

For each located position we save the residual stream PER LAYER as a separate
.pt tensor (keyed by both position AND layer), so downstream can differentiate
every captured layer (middle->last) rather than only a layer mean.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.datasets.prompt import SesgoPromptSample
from src.datasets.sesgo_eval import GeometryActivation
from src.ternary_choice import TernaryChoiceRunner
from .geometry_position_finder import answer_start_for, find_positions

# Structural positions we look for, in capture (sequence) order. think_* exist
# only for reasoning models; im_end/assistant only for families whose template
# emits them. Missing ones are logged and skipped.
_POSITION_TYPES = (
    "im_end",
    "newline",
    "im_start",
    "assistant",
    "think_open",
    "think_close",
    "answer_prefix",
    "label",
)

# Tokens to greedily decode for the non-thinking answer path. We only need the
# answer token (the first generated token); a few extra give a stable sequence.
_GREEDY_TOKENS = 24


def _resid_filter(layers: list[int] | None):
    """names_filter selecting per-layer residual-stream (resid_post) hooks."""

    def keep(name: str) -> bool:
        if "resid_post" not in name:
            return False
        if layers is None:
            return True
        return any(name == f"blocks.{layer}.hook_resid_post" for layer in layers)

    return keep


def _layer_resid(cache: dict, layer: int, pos: int) -> torch.Tensor:
    """Residual vector at token ``pos`` for ONE layer -> [d_model] float32 CPU."""
    # cache tensors are [batch, seq, d_model]; batch is always 1 here.
    act = cache[f"blocks.{layer}.hook_resid_post"]
    return act[0, pos].detach().float().cpu()


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
    """Persist each (position, layer) residual as its OWN tensor; return captures.

    Saving per-(position, LAYER) — not one stacked tensor per position — keys the
    geometry by layer so downstream can differentiate every captured layer
    (middle->last) independently. ``layers`` is the middle->last layer subset the
    caller selected; one .pt of shape [d_model] is written per (position, layer).
    """
    captured: list[GeometryActivation] = []
    for ptype in _POSITION_TYPES:
        pos = positions.get(ptype)
        if pos is None:
            continue
        token_text = runner.decode_ids([ids[pos]])
        for layer in layers:
            resid = _layer_resid(cache, layer, pos)  # [d_model]
            fname = f"sample_{sample_idx}_{ptype}_L{layer}.pt"
            torch.save(resid, sample_dir / fname)
            captured.append(
                GeometryActivation(
                    position_type=ptype,
                    token_position=pos,
                    token_text=token_text,
                    path=str((sample_dir / fname).relative_to(rel_root)),
                    layer=layer,
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
