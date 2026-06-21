"""A single SESGO ambiguous-context item: a (context, question) with 3 options."""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from .sesgo_category import SesgoCategory
from .sesgo_label import SesgoLabel


def origin_label(bbq: bool) -> str:
    """Map the ``bbq`` provenance flag to a human axis value.

    A single source of truth so every sample type and the geometry viz agree on
    the spelling: ``False`` -> "original", ``True`` -> "BBQ-adapted".
    """
    return "BBQ-adapted" if bbq else "original"


@dataclass
class SesgoItem(BaseSchema):
    """One SESGO ambiguous prompt with its three role-labelled answer options.

    `question_id` is shared by the negative/non-negative phrasing pair so callers
    can recover the polarity pair from a flattened list; it is a stable hash of
    (category, language, context), which is invariant across the polarity flip.
    For ambiguous contexts the gold answer is always UNKNOWN ("not enough info").
    """

    question_id: str
    category: SesgoCategory
    language: str
    polarity: str  # "neg" or "nonneg"
    context: str
    question: str
    target_text: str
    other_text: str
    unknown_text: str
    bbq: bool = False
    gold_label: SesgoLabel = SesgoLabel.UNKNOWN

    @property
    def origin_label(self) -> str:
        """Provenance of the item: "BBQ-adapted" when ``bbq`` else "original"."""
        return origin_label(self.bbq)

    @property
    def options_in_canonical_order(self) -> tuple[tuple[SesgoLabel, str], ...]:
        """Options in the corpus' ans0/ans1/ans2 order: OTHER, TARGET, UNKNOWN.

        Returns role-tagged pairs so presentation code can render the three
        choices without re-deriving which text plays which role.
        """
        return (
            (SesgoLabel.OTHER, self.other_text),
            (SesgoLabel.TARGET, self.target_text),
            (SesgoLabel.UNKNOWN, self.unknown_text),
        )
