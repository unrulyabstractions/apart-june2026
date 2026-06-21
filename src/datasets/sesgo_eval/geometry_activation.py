"""One residual-stream snapshot at a structural token position AND layer.

The geometry half asks "where, in the model's representation, does a scaffold
move the answer". To answer that we capture the residual stream at a handful of
fine-grained STRUCTURAL token positions in the teacher-forced answer sequence —
the chat-template boundary tokens (im_end / newline / im_start / assistant), the
<think> open and close, and the answer (answer_prefix / label) tokens — for each
transformer layer from MIDDLE to LAST.

The capture is keyed by (position_type, LAYER): each saved tensor is a single
layer's [d_model] residual at one position, so downstream can differentiate
EVERY captured layer (not just a layer mean). The tensor is saved to disk
(torch.save) and this schema keeps only the relative ``path`` to it — never the
tensor — so the GeometryDataset json stays small and roundtrips cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema


@dataclass
class GeometryActivation(BaseSchema):
    """Pointer to a saved residual snapshot at one (structural position, layer)."""

    position_type: str  # im_end|newline|im_start|assistant|think_*|answer_prefix|label
    token_position: int  # index of this token in the forced id sequence
    token_text: str  # decoded surface text of that token (for sanity checks)
    path: str  # relative path to the saved .pt tensor, shape [d_model]
    layer: int = -1  # transformer layer this single-layer residual came from
