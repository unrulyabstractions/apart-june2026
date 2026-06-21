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
    """One prompt in the SESGO grid, with its role-decoding metadata.

    Carries TWO rendered forms: the 3-option prompt (`text`, target+other+UNKNOWN)
    and a 2-option forced-choice prompt (`text_2opt`, target+other only, no
    UNKNOWN). Each has its own option markers and position->role decoding tuple;
    the 2-option form drops the UNKNOWN role entirely.
    """

    sample_idx: int
    question_id: str
    bias_category: str
    question_polarity: str
    context_condition: str  # "ambig" or "disambig"
    language: str
    scaffold_id: str | None
    label_style: str
    text: str  # fully rendered 3-option prompt
    option_labels: tuple[str, str, str]  # the 3 position markers (m0,m1,m2)
    position_labels: tuple[SesgoLabel, SesgoLabel, SesgoLabel]  # pos i -> role shown
    choice_prefix: str
    # 2-option forced-choice rendering (target+other only, no UNKNOWN). Its two
    # markers and position->role tuple decode a binary readout the same way.
    text_2opt: str = ""
    option_labels_2opt: tuple[str, str] = ("a)", "b)")
    position_labels_2opt: tuple[SesgoLabel, SesgoLabel] = (
        SesgoLabel.OTHER,
        SesgoLabel.TARGET,
    )
    gold_label: SesgoLabel = SesgoLabel.UNKNOWN
    # Provenance / social-group axes carried so any downstream record can colour
    # by them without re-joining the source item. bbq: original vs BBQ-adapted;
    # target_identity (ans1) / other_identity (ans0) are the literal group strings.
    bbq: bool = False
    target_identity: str = ""
    other_identity: str = ""

    @property
    def gold_position(self) -> int | None:
        """Displayed position holding the gold role, or None if not shown.

        For ambiguous items the gold is UNKNOWN; for disambiguated items it is the
        ground-truth role. Returns None when that role is absent from the 3-option
        display (should not happen for the 3-option form).
        """
        if self.gold_label in self.position_labels:
            return self.position_labels.index(self.gold_label)
        return None

    def label_for_position(self, i: int) -> SesgoLabel:
        """Which role (target/other/unknown) is displayed at position `i`."""
        return self.position_labels[i]
