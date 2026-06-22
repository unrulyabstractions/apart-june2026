"""Display names + ordering + scaffold hues for the canonical bias figure (B).

The canonical bias-alignment figure plots each group (model or scaffold) as one
point. This module supplies the short display name and the canonical order for
both group kinds, plus the scaffold colour:

  * MODELS (baseline) -- a bare output-dir name like ``Llama-3.2-1B-Instruct``,
    shortened to ``Llama-3.2 1B`` and ordered by size via sesgo_model_sizing.
  * SCAFFOLDS (selection) -- a scaffold id, named via plain_language_labels and
    ordered in the canonical SCAFFOLD_ORDER (no-scaffold first).
"""

from __future__ import annotations

import re

from sesgo.baseline.sesgo_model_sizing import params_b
from sesgo.common.plain_language_labels import (
    SCAFFOLD_LABEL,
    SCAFFOLD_ORDER,
    normalize_scaffold,
)

# Strip the instruct/chat suffix and collapse the size token to ``Family-x NB``.
_INSTRUCT_SUFFIX = re.compile(r"[-_](Instruct|it|chat)\b.*$", re.IGNORECASE)

# Okabe-Ito hue per scaffold (no-scaffold baseline is the neutral grey-black).
_SCAFFOLD_COLORS: dict[str | None, str] = {
    None: "#000000",
    "interpretive_direction": "#0072B2",
    "intent_and_register_respect": "#009E73",
    "prior_dominance_warning": "#CC79A7",
}


def scaffold_color(scaffold_id: str | None) -> str:
    """Okabe-Ito hue for a scaffold (grey fallback for unknown ids)."""
    return _SCAFFOLD_COLORS.get(normalize_scaffold(scaffold_id), "#555555")


def model_display_name(bare_name: str) -> str:
    """``Llama-3.2-1B-Instruct`` -> ``Llama-3.2 1B``; size made the trailing token."""
    stem = _INSTRUCT_SUFFIX.sub("", bare_name)
    parts = stem.rsplit("-", 1)
    return f"{parts[0]} {parts[1]}" if len(parts) == 2 else stem


def scaffold_display_name(scaffold_id: str | None) -> str:
    """Compact plain-language scaffold name (drops the redundant 'scaffold' word)."""
    full = SCAFFOLD_LABEL.get(normalize_scaffold(scaffold_id), str(scaffold_id))
    return full[: -len(" scaffold")] if full.endswith(" scaffold") else full


def model_sort_key(bare_name: str) -> tuple[float, str]:
    """Size-then-name key so baseline points order smallest-first by params."""
    return (params_b(bare_name) or 0.0, bare_name)


def scaffold_sort_key(scaffold_id: str | None) -> int:
    """Index into the canonical scaffold order (unknown ids sort last)."""
    norm = normalize_scaffold(scaffold_id)
    return SCAFFOLD_ORDER.index(norm) if norm in SCAFFOLD_ORDER else len(SCAFFOLD_ORDER)
