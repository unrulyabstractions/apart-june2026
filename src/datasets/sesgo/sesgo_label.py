"""The three-way answer label of a SESGO item.

Every SESGO question offers exactly three options whose *roles* are fixed by the
positional convention in `answer_info` (ans0=OTHER, ans1=TARGET, ans2=UNKNOWN).
We name the roles rather than the positions so downstream bias scoring can talk
about "did the model pick the TARGET group" independent of option ordering.
"""

from __future__ import annotations

from enum import Enum


class SesgoLabel(Enum):
    """Role of a SESGO answer option.

    TARGET  - the stereotyped/marginalized group the bias is probed against.
    OTHER   - the contrasting (non-target) group.
    UNKNOWN - the "not enough information" option (gold for ambiguous contexts).
    """

    TARGET = "target"
    OTHER = "other"
    UNKNOWN = "unknown"
