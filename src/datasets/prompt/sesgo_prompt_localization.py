"""Language-localized structural scaffolding for the SESGO prompt render.

A SESGO item is authored entirely in one language ("es"/"en"); only the prompt's
structural skeleton — the section markers and the trailing answer cue — is fixed
by us rather than the corpus. We localize that skeleton so a Spanish item reads as
one coherent Spanish prompt (and likewise English), instead of mixing English
markers into Spanish text. The non-thinking querier teacher-forces `choice_prefix`
verbatim, so it must match the item language for the model to continue naturally.

One source of truth lives here: `sesgo_markers` and `sesgo_choice_prefix` both
key off the same language convention used elsewhere ("es" -> Spanish, else
English), so the generator never spells the localized strings itself.
"""

from __future__ import annotations

# Section markers and answer cue per language. English is the fallback for any
# code other than "es", matching `Scaffold.text`'s convention.
_SECTION_MARKERS = {
    "en": ("##context", "##question", "##options"),
    "es": ("##contexto", "##pregunta", "##opciones"),
}
_CHOICE_PREFIXES = {
    "en": "Answer: ",
    "es": "Respuesta: ",
}


def sesgo_markers(language: str) -> tuple[str, str, str]:
    """The (context, question, options) section markers for `language`."""
    return _SECTION_MARKERS["es"] if language == "es" else _SECTION_MARKERS["en"]


def sesgo_choice_prefix(language: str) -> str:
    """The trailing answer cue for `language` (teacher-forced by the querier)."""
    return _CHOICE_PREFIXES["es"] if language == "es" else _CHOICE_PREFIXES["en"]
