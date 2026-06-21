"""Configuration selecting which slices of the SESGO prompt grid to generate.

Every axis defaults to its full setting, so a bare `SesgoPromptConfig(name=...)`
emits the complete cross product: all 6 role↔position permutations, all label
styles, and a no-scaffold condition alongside each supplied scaffold. Toggles are
stored as explicit value lists (not implicit flags) so the config both documents
and reproduces a dataset deterministically. Source selection (categories,
languages, limit) is recorded here for provenance; loading the SesgoItems
themselves stays the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema
from .sesgo_label_style import get_sesgo_label_styles


def _all_label_styles() -> list[tuple[str, str, str]]:
    return get_sesgo_label_styles()


@dataclass
class SesgoPromptConfig(BaseSchema):
    """Which label styles, permutations and scaffold conditions to include."""

    name: str
    label_styles: list[tuple[str, str, str]] = field(default_factory=_all_label_styles)
    # all_permutations=True crosses all 6 role->position assignments (defeats
    # position bias); False keeps only the canonical (identity) ordering.
    all_permutations: bool = True
    # Whether to also emit the no-scaffold (baseline) condition per item.
    include_no_scaffold: bool = True
    # Optional grid-wide override for the answer cue. Left None so the generator
    # derives it from each item's language ("Answer: " / "Respuesta: "); set a
    # string only to force one cue across both languages.
    choice_prefix: str | None = None
    # Source provenance only; the caller loads items with these.
    categories: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    limit: int | None = None

    def get_filename(self) -> str:
        """Filename for saving a dataset built from this config."""
        return f"{self.name}_{self.get_id()}.json"
