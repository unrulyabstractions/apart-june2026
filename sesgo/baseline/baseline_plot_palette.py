"""Shared plain-language palette + helpers for the single-model baseline figures.

One source of truth for the Okabe-Ito hues, plain-language slice/role legends, the
random-guess references, and the stable category display order used by BOTH the
baseline accuracy figure and the role-probability figure. Keeps each figure file
small and guarantees the two plots stay visually and verbally consistent.
"""

from __future__ import annotations

from sesgo.common import CATEGORY_ORDER, READOUT_LABEL

# Short headline per answering mode (accuracy-figure row title).
COND_TITLE: dict[str, str] = {
    "non_thinking": READOUT_LABEL["non_thinking"],
    "non_thinking_2opt": READOUT_LABEL["non_thinking_2opt"],
    "greedy_thinking": READOUT_LABEL["greedy_thinking"],
}

# Okabe-Ito hues: one per accuracy slice (shared with the role legend palette).
SLICE_COLORS: dict[str, str] = {
    "ambig": "#0072B2", "disambig-target": "#E69F00", "disambig-other": "#56B4E9",
}
SLICE_LABELS: dict[str, str] = {
    "ambig": "Abstains (ambiguous)",
    "disambig-target": "Stereotyped group",
    "disambig-other": "Other group",
}
ROLE_NAMES: tuple[str, ...] = ("target", "other", "unknown")
ROLE_COLORS: dict[str, str] = {"target": "#E69F00", "other": "#56B4E9", "unknown": "#009E73"}

# Random-guess reference differs by mode: the forced two-way choice picks 1 of 2,
# the three-option modes pick 1 of 3.
CHANCE: dict[str, float] = {"non_thinking": 1 / 3, "non_thinking_2opt": 1 / 2, "greedy_thinking": 1 / 3}


def ordered_categories(cats: list[str]) -> list[str]:
    """Categories in the stable plain-language display order (unknowns trail)."""
    return [c for c in CATEGORY_ORDER if c in cats] + [c for c in cats if c not in CATEGORY_ORDER]
