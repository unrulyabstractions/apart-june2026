"""One residual-stream snapshot at a structurally meaningful token position.

The geometry half asks "where, in the model's representation, does a scaffold
move the answer". To answer that we capture the full per-layer residual stream
([n_layers, d_model]) at a handful of STRUCTURAL token positions in the
teacher-forced answer sequence — the assistant turn boundary, the <think> open
and close, and the answer (option-marker) token. Each position is one of these.

The actual tensor is heavy and per-position, so it is saved to disk (torch.save)
and this schema keeps only the relative ``path`` to it — never the tensor — so
the GeometryDataset json stays small and roundtrips cheaply via BaseSchema.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema


@dataclass
class GeometryActivation(BaseSchema):
    """Pointer to a saved residual snapshot at one structural token position."""

    position_type: str  # "turn" | "think_open" | "think_close" | "answer"
    token_position: int  # index of this token in the forced id sequence
    token_text: str  # decoded surface text of that token (for sanity checks)
    path: str  # relative path to the saved .pt tensor, shape [n_layers, d_model]
