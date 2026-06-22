"""Plain-language display labels shared by every SESGO plot.

ONE source of truth so figures speak human, not pipeline jargon. Internal keys
(scaffold ids, readout names, Spanish category codes, polarity flags, roles) map
to short, accessible English a non-expert reads at a glance. Import from here
instead of hardcoding ``3-opt`` / ``non-thinking`` / ``p(UNKNOWN)`` / ``chance``
anywhere. Plots are read by reviewers, not pipeline authors.
"""

from __future__ import annotations

# How the model was read out -> plain words, plus a one-line caption gloss.
READOUT_LABEL: dict[str, str] = {
    "non_thinking": "Without thinking",
    "greedy_thinking": "With thinking",
    "non_thinking_2opt": "Forced two-way choice",
    "thinking": "Free-form thinking",
}
READOUT_GLOSS: dict[str, str] = {
    "non_thinking": "answers directly, no reasoning",
    "greedy_thinking": "reasons step by step, then answers",
    "non_thinking_2opt": "only two options offered, no 'unknown'",
    "thinking": "writes free-form reasoning, then answers",
}

# Debiasing scaffold (one-sentence preamble) -> plain words. None / "None" is the
# no-scaffold baseline. Short two-line variants are for crowded x-axes.
SCAFFOLD_LABEL: dict[str | None, str] = {
    None: "No scaffold",
    "None": "No scaffold",
    "interpretive_direction": "Interpretive-direction scaffold",
    "intent_and_register_respect": "Intent & register scaffold",
    "prior_dominance_warning": "Prior-dominance-warning scaffold",
}
SCAFFOLD_SHORT: dict[str | None, str] = {
    None: "No\nscaffold",
    "None": "No\nscaffold",
    "interpretive_direction": "Interpretive\ndirection",
    "intent_and_register_respect": "Intent &\nregister",
    "prior_dominance_warning": "Prior-dominance\nwarning",
}
SCAFFOLD_ORDER: tuple[str | None, ...] = (
    None,
    "interpretive_direction",
    "intent_and_register_respect",
    "prior_dominance_warning",
)

# Spanish bias-category codes -> English social-group axis (stable display order).
CATEGORY_LABEL: dict[str, str] = {
    "clasismo": "Classism",
    "racismo": "Racism",
    "xenofobia": "Xenophobia",
    "genero": "Gender",
}
CATEGORY_ORDER: tuple[str, ...] = ("clasismo", "racismo", "xenofobia", "genero")

# Question wording -> plain words (neutral framing vs negatively-loaded framing).
POLARITY_LABEL: dict[str, str] = {"neg": "Negative wording", "nonneg": "Neutral wording"}
POLARITY_ORDER: tuple[str, ...] = ("nonneg", "neg")

# Where the only correct answer should land.
CONTEXT_LABEL: dict[str, str] = {
    "ambig": "Ambiguous question (no clear answer)",
    "disambig": "Clear question (answer is stated)",
}
# Which group the model picked / the gold role.
ROLE_LABEL: dict[str, str] = {
    "target": "Stereotyped group",
    "other": "Other group",
    "unknown": "Abstains ('unknown')",
}

# Reusable plain phrases for axis titles / reference lines.
ABSTENTION_AXIS = "Abstention rate (answers 'unknown')"
RANDOM_GUESS_LABEL = "random guessing"


def scaffold_label(value: str | None) -> str:
    """Full plain-language scaffold name; unknown ids pass through verbatim."""
    return SCAFFOLD_LABEL.get(value, str(value))


def normalize_scaffold(value: str | None) -> str | None:
    """Collapse both Python ``None`` and the string ``"None"`` to the baseline."""
    return None if value in (None, "None") else value
