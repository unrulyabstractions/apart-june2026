"""Spread crowded segment text labels apart in y so none overlap a neighbour.

The bias-alignment figure stacks many horizontal segments; when several share a
similar accuracy their text labels collide. This computes a non-overlapping y for
each label by sweeping bottom-to-top and pushing any too-close label up by a fixed
minimum gap, returning where each label should sit (so the renderer can draw a
thin leader line from the segment to its nudged label). One concern, one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema


@dataclass
class LabelSlot(BaseSchema):
    """A label's natural anchor and its de-collided draw position (data coords)."""

    index: int  # position in the caller's segment list (to recover colour/text)
    y_natural: float  # the segment's true accuracy (leader-line target)
    y_label: float  # the nudged y where the label text is drawn


def spread_labels(ys: list[float], min_gap: float, hi: float) -> list[LabelSlot]:
    """Push labels up so consecutive ones differ by >= ``min_gap``, capped at ``hi``.

    Sweeps in ascending accuracy: each label starts at its true y and is lifted to
    at least ``min_gap`` above the previous one. A final top-down pass pulls the
    stack back down under ``hi`` so a dense cluster stays inside the axes.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed: dict[int, float] = {}
    last = float("-inf")
    for i in order:
        y = max(ys[i], last + min_gap)
        placed[i] = y
        last = y
    # If the stack overflowed the top, slide the whole tail down to fit.
    overflow = last - hi
    if overflow > 0:
        for i in order:
            placed[i] = max(ys[i] - overflow, placed[i] - overflow)
        # Re-resolve any collisions the downward slide reintroduced.
        last = float("-inf")
        for i in order:
            placed[i] = max(placed[i], last + min_gap)
            last = placed[i]
    return [LabelSlot(i, ys[i], placed[i]) for i in range(len(ys))]
