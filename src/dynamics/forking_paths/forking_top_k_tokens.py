"""Extract the per-position alternate tokens from the base path's logits.

Stage 1 of forking-paths records, at each base-path position t, the top-K most
probable next tokens w with p(x_t=w | x*_{<t}) >= a probability floor. These are
the only branch points worth Monte-Carlo sampling (Stage 2). We read them from
the base trajectory's full-vocab logits (one softmax per position); the base
(greedy) token is always included and flagged so the survival test can compare
each alternate's outcome against the base token's.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Paper hyperparameters (verbatim): top k <= 10 tokens each with p >= 5%.
DEFAULT_TOP_K = 10
DEFAULT_PROB_FLOOR = 0.05


@dataclass
class AltToken:
    """One alternate next-token candidate w at a position: id, text, probability."""

    token_id: int
    token_text: str
    prob: float
    is_base: bool  # the greedy/base-path token w* at this position


def alternates_at_position(
    logits_row: torch.Tensor,
    base_token_id: int,
    decode_id,
    top_k: int = DEFAULT_TOP_K,
    prob_floor: float = DEFAULT_PROB_FLOOR,
) -> list[AltToken]:
    """Top-k tokens with p >= floor at one position, base token always included.

    ``logits_row`` is the full-vocab logits that PREDICT this position's token;
    ``decode_id`` maps a token id to its display text. The base token is appended
    (flagged) even if it falls below the floor, so the branch set always contains
    the path the model actually took.
    """
    probs = torch.softmax(logits_row.float(), dim=-1)
    k = min(top_k, probs.numel())
    top_probs, top_ids = torch.topk(probs, k)

    alts: list[AltToken] = []
    seen: set[int] = set()
    for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
        if p < prob_floor:
            continue
        alts.append(AltToken(tid, decode_id([tid]), float(p), tid == base_token_id))
        seen.add(tid)

    # Guarantee the base token is present (the path the model committed to).
    if base_token_id not in seen:
        alts.append(
            AltToken(
                base_token_id,
                decode_id([base_token_id]),
                float(probs[base_token_id]),
                True,
            )
        )
    return alts
