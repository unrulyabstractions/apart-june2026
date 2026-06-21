"""A debiasing scaffold: a bilingual preamble prepended to a SESGO prompt.

A scaffold nudges the model toward answering "unknown" on ambiguous items. It is
authored in both languages because the prompt it precedes is language-fixed per
item, and the preamble must match that language to read as one coherent text.
We keep `text_en`/`text_es` literal (not templated) so each translation can be
authored idiomatically rather than mechanically derived from the other.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema


@dataclass
class Scaffold(BaseSchema):
    """One debiasing preamble in English and Spanish, picked by item language."""

    scaffold_id: str
    text_en: str
    text_es: str

    def text(self, language: str) -> str:
        """The scaffold text in `language` (Spanish for "es", else English)."""
        return self.text_es if language == "es" else self.text_en
