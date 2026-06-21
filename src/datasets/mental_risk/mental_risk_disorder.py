"""Disorder categories of the MentalRiskES corpus.

Enum values are the on-disk directory names; the English/Spanish labels are
phrasing-ready fragments meant to be dropped into prompt sentences such as
"this person suffers from {label}".
"""

from __future__ import annotations

from enum import Enum

# Directory name -> (English label, Spanish label). Kept beside the enum so the
# two phrasings stay in lockstep with the canonical directory names.
_LABELS: dict[str, tuple[str, str]] = {
    "Anxiety": ("anxiety", "ansiedad"),
    "Depress": ("depression", "depresión"),
    "ED": ("an eating disorder", "un trastorno alimentario"),
}


class Disorder(Enum):
    """A MentalRiskES disorder, identified by its corpus directory name."""

    ANXIETY = "Anxiety"
    DEPRESSION = "Depress"
    EATING_DISORDER = "ED"

    @property
    def label(self) -> str:
        """English phrasing fragment (e.g. 'anxiety', 'an eating disorder')."""
        return _LABELS[self.value][0]

    @property
    def label_es(self) -> str:
        """Spanish phrasing fragment (e.g. 'ansiedad')."""
        return _LABELS[self.value][1]
