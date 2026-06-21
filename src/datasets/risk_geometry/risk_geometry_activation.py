"""One residual-stream snapshot at a structurally meaningful token position.

The risk-geometry half asks "where, in the model's representation, does a framing
move the risk judgement". To answer that we capture the full per-layer residual
stream ([n_layers, d_model]) at a handful of STRUCTURAL token positions in the
greedy non-thinking answer sequence — the assistant turn boundary, the <think>
open and close, and the answer (option-marker) token. Each position is one of
these.

The actual tensor is heavy and per-position, so it is saved to disk (torch.save)
and this schema keeps only the relative ``path`` to it — never the tensor — so
the RiskGeometryDataset json stays small and roundtrips cheaply via BaseSchema.
This mirrors SESGO's GeometryActivation exactly; the name is suffixed _risk so it
stays globally unique alongside the bias-role version.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema


@dataclass
class RiskGeometryActivation(BaseSchema):
    """Pointer to a saved residual snapshot at one structural token position."""

    position_type: str  # "turn" | "think_open" | "think_close" | "answer"
    token_position: int  # index of this token in the forced id sequence
    token_text: str  # decoded surface text of that token (for sanity checks)
    path: str  # relative path to the saved .pt tensor, shape [n_layers, d_model]
