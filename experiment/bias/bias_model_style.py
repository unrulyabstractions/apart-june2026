"""Per-model display style for the bias-alignment figure: the family, size, mode and
colour a renderer needs to draw one model's marker/line/label without re-parsing names.

Kept separate from ``BiasSegment`` (the data reduction) so the plot layer carries its
own presentation concern. One ``ModelStyle`` per model dir, keyed by ``group_key``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema


@dataclass
class ModelStyle(BaseSchema):
    """How one model draws: family colour, size (for marker area + label), mode (shape)."""

    group_key: str
    family: str
    size_b: float
    is_thinking: bool
    color: str        # family base colour (hex)
    size_label: str   # short size tag, e.g. "0.8B", "27B"
