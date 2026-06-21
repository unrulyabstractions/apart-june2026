"""One rendered SESGO prompt plus everything to map a reply back to a role.

This object is self-contained: a downstream querier never re-derives anything.
The crucial mapping is `position_labels` — for each displayed position i it
records which SesgoLabel (target/other/unknown) was shown there. Because the
generator emits all 6 role↔position permutations, the position a model picks is
meaningless until decoded through this tuple, which is exactly how position bias
is defeated. `option_labels` are the surface markers (e.g. "a)","b)","c)") the
querier passes to a chooser; the chosen index then indexes `position_labels`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.datasets.sesgo import SesgoLabel


@dataclass
class SesgoPromptSample(BaseSchema):
    """One prompt in the SESGO grid, with its role-decoding metadata."""

    sample_idx: int
    question_id: str
    bias_category: str
    question_polarity: str
    language: str
    scaffold_id: str | None
    label_style: str
    text: str  # fully rendered prompt
    option_labels: tuple[str, str, str]  # the 3 position markers (m0,m1,m2)
    position_labels: tuple[SesgoLabel, SesgoLabel, SesgoLabel]  # pos i -> role shown
    choice_prefix: str
    gold_label: SesgoLabel = SesgoLabel.UNKNOWN

    @property
    def gold_position(self) -> int:
        """Index of the displayed position holding the UNKNOWN (gold) option."""
        return self.position_labels.index(SesgoLabel.UNKNOWN)

    def label_for_position(self, i: int) -> SesgoLabel:
        """Which role (target/other/unknown) is displayed at position `i`."""
        return self.position_labels[i]
