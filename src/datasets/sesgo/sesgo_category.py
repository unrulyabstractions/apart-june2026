"""Bias categories of the SESGO benchmark.

Enum values are the Spanish file stems used in the prompt filenames
(`prompts_<stem>_<lang>.xlsx`), since the filename is the one identifier that is
consistent across the corpus. The data's own `category` column is *not* reliable
— casing and spelling vary by file (e.g. "Xenophoby", lowercase "gender",
"clasismo", "SES") — so `from_data_value` tolerates every observed variant.
"""

from __future__ import annotations

from enum import Enum

# Stem -> English display label. Kept beside the enum so the English phrasing
# stays in lockstep with the canonical Spanish stems.
_ENGLISH: dict[str, str] = {
    "racismo": "Racism",
    "xenofobia": "Xenophobia",
    "clasismo": "Classism",
    "genero": "Gender",
}

# Lowercased data-column aliases -> stem. Maps every spelling actually seen in
# the corpus' `category` column onto the canonical category.
_DATA_ALIASES: dict[str, str] = {
    "racism": "racismo",
    "racismo": "racismo",
    "xenophobia": "xenofobia",
    "xenophoby": "xenofobia",
    "xenofobia": "xenofobia",
    "classism": "clasismo",
    "clasismo": "clasismo",
    "ses": "clasismo",
    "gender": "genero",
    "genero": "genero",
}


class SesgoCategory(Enum):
    """A SESGO bias category, identified by its Spanish prompt-file stem."""

    RACISM = "racismo"
    XENOPHOBIA = "xenofobia"
    CLASSISM = "clasismo"
    GENDER = "genero"

    @property
    def english(self) -> str:
        """English display label (e.g. 'Racism', 'Xenophobia')."""
        return _ENGLISH[self.value]

    @classmethod
    def from_data_value(cls, s: str) -> SesgoCategory:
        """Map a raw `category` cell to the enum, tolerating spelling variants.

        Case-insensitive and alias-aware because the corpus is inconsistent
        ("Racism", "Xenophoby", "gender", "clasismo", "SES", ...).
        """
        return cls(_DATA_ALIASES[s.strip().lower()])
