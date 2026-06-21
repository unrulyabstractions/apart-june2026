"""Shared plain-language palette + helpers for the single-model baseline figures.

One source of truth for the Okabe-Ito hues, plain-language slice/role legends, the
random-guess references, and the stable category display order used by BOTH the
baseline accuracy figure and the role-probability figure. Keeps each figure file
small and guarantees the two plots stay visually and verbally consistent.
"""

from __future__ import annotations

from sesgo.common import CATEGORY_ORDER, READOUT_LABEL

# Plain-language headline per answering mode (accuracy-figure row title). The
# forced two-way row only offers two answers, so its option count differs.
COND_TITLE: dict[str, str] = {
    "non_thinking": f"{READOUT_LABEL['non_thinking']}  -  "
    "three options offered, including 'unknown'",
    "non_thinking_2opt": f"{READOUT_LABEL['non_thinking_2opt']}  -  "
    "only the two named groups offered, no 'unknown'",
    "greedy_thinking": f"{READOUT_LABEL['greedy_thinking']}  -  "
    "writes reasoning, then answers from three options",
}

# Okabe-Ito hues: one per accuracy slice (shared with the role legend palette).
SLICE_COLORS: dict[str, str] = {
    "ambig": "#0072B2", "disambig-target": "#E69F00", "disambig-other": "#56B4E9",
}
SLICE_LABELS: dict[str, str] = {
    "ambig": "Correctly abstains on ambiguous questions (correct answer is 'unknown')",
    "disambig-target": "Correct when the stereotyped group is the answer",
    "disambig-other": "Correct when the other group is the answer",
}
ROLE_NAMES: tuple[str, ...] = ("target", "other", "unknown")
ROLE_COLORS: dict[str, str] = {"target": "#E69F00", "other": "#56B4E9", "unknown": "#009E73"}

# Random-guess reference differs by mode: the forced two-way choice picks 1 of 2,
# the three-option modes pick 1 of 3.
CHANCE: dict[str, float] = {"non_thinking": 1 / 3, "non_thinking_2opt": 1 / 2, "greedy_thinking": 1 / 3}


def ordered_categories(cats: list[str]) -> list[str]:
    """Categories in the stable plain-language display order (unknowns trail)."""
    return [c for c in CATEGORY_ORDER if c in cats] + [c for c in cats if c not in CATEGORY_ORDER]
