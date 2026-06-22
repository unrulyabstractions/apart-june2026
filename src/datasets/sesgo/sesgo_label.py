"""The three-way answer label of a SESGO item.

Every SESGO question offers exactly three options whose *roles* are fixed by the
positional convention in `answer_info` (ans0=OTHER, ans1=TARGET, ans2=UNKNOWN).
We name the roles rather than the positions so downstream bias scoring can talk
about "did the model pick the TARGET group" independent of option ordering.
"""

from __future__ import annotations

from enum import Enum


# Corpus positional convention for the `label` / `answer_info` index:
# ans0 == OTHER, ans1 == TARGET, ans2 == UNKNOWN. Single source of truth for
# decoding a disambiguated item's ground-truth `label` integer into a role.
_INDEX_TO_LABEL = ("other", "target", "unknown")


class SesgoLabel(Enum):
    """Role of a SESGO answer option.

    TARGET  - the stereotyped/marginalized group the bias is probed against.
    OTHER   - the contrasting (non-target) group.
    UNKNOWN - the "not enough information" option (gold for ambiguous contexts).
    """

    TARGET = "target"
    OTHER = "other"
    UNKNOWN = "unknown"

    @classmethod
    def from_answer_index(cls, index: int) -> "SesgoLabel":
        """Decode a corpus answer index (ans0/ans1/ans2) into its role.

        The SESGO `label` column gives the index of the correct answer for a
        DISAMBIGUATED item; ans0=OTHER, ans1=TARGET, ans2=UNKNOWN by convention.
        """
        return cls(_INDEX_TO_LABEL[int(index)])
