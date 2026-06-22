"""Position-marker triples for SESGO's three-option layout.

Reuses the label-style idea from `formatting_variation`, but SESGO always shows
exactly three options, so we need triples (not pairs). Each triple's markers
start with distinct characters so a runner can score a single token, and the set
spans alphabetic / numeric / late-alphabet styles to vary surface form without
changing meaning — part of defeating shallow format-driven answer bias.
"""

from __future__ import annotations

# Canonical grid order. Three styles is enough to probe format sensitivity;
# every marker's leading character is unique within and across triples.
SESGO_LABEL_STYLES: list[tuple[str, str, str]] = [
    ("a)", "b)", "c)"),
    ("1)", "2)", "3)"),
    ("x)", "y)", "z)"),
]


def get_sesgo_label_styles() -> list[tuple[str, str, str]]:
    """All three-option marker triples, in canonical grid order."""
    return SESGO_LABEL_STYLES.copy()
