"""Label-token pairs for the CATEGORIZE task.

These reuse the label-style concept from `formatting_variation` but the
risk grid needs a small, fixed, language-neutral set whose first characters
are distinct (so a runner can score a single distinguishing token). "Yes"/"No"
are intentionally included because they read naturally as binary answers.
"""

from __future__ import annotations

from .formatting.formatting_variation import SIMPLE_LABEL_STYLES

# ("a)","b)") already lives in the shared style list; we reuse it rather than
# redeclaring the a)/b) concept, and add the numeric + word variants the risk
# grid requires. Order matters: it is the canonical grid order.
RISK_LABEL_STYLES: list[tuple[str, str]] = [
    SIMPLE_LABEL_STYLES[0],  # ("a)", "b)")
    ("1)", "2)"),
    ("Yes", "No"),
]


def get_risk_label_styles() -> list[tuple[str, str]]:
    """All categorize label-token pairs, in canonical grid order."""
    return RISK_LABEL_STYLES.copy()
