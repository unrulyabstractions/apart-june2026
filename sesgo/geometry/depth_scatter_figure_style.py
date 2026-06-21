"""Shared house-style chrome for the geometry depth-scatter panels (2D + 3D).

One place that owns the colourblind-safe scaffold palette, the plain-language
legend text, the depth/layer caption phrasing, and the separation-score box, so
the 2D and 3D renderers stay tiny and read identically. The score box uses the
corrected reading: 1 = the two groups sit fully apart, 0 = they overlap, below 0
= they are intermixed.
"""

from __future__ import annotations

from sesgo.common import SCAFFOLD_LABEL

# Okabe-Ito, colourblind-safe: blue = no scaffold, orange = with scaffold.
SCAFFOLD_COLOR: dict[str, str] = {"baseline": "#0072B2", "scaffold": "#E69F00"}


def scaffold_legend(n_base: int, n_scaf: int) -> list[tuple[str, str, str]]:
    """(colour, plain label with n) for the two scaffold groups, in legend order."""
    return [
        (SCAFFOLD_COLOR["baseline"], "baseline", f"{SCAFFOLD_LABEL[None]}  (n={n_base})"),
        (SCAFFOLD_COLOR["scaffold"], "scaffold",
         f"{SCAFFOLD_LABEL['interpretive_direction']}  (n={n_scaf})"),
    ]


def depth_phrase(depth: float) -> str:
    """Plain words for how deep through the network a depth fraction sits."""
    if depth < 0.6:
        return "halfway through the network"
    if depth < 0.8:
        return "about two thirds of the way through"
    return "near the top of the network"


def panel_title(kind: str, depth: float, layer: str, n_layers: int) -> str:
    """Plain two-line headline naming the colour-by, the depth %, and the layer."""
    return (
        f"The model's internal state, coloured by whether a debiasing "
        f"instruction was given\n"
        f"{kind} view at depth {depth * 100:.0f}%  "
        f"(layer {layer} of {n_layers}, {depth_phrase(depth)})"
    )


HOW_TO_READ = (
    "How to read this: each dot is one question's internal state at the answer token; "
    "the further\napart the two colours sit, the more the one-line instruction reshaped "
    "what the model represents."
)


def separation_text(silhouette: float, ci_low: float, ci_high: float) -> str:
    """The score box: corrected 1/0/below-0 reading of the separation score."""
    return (
        f"Separation score = {silhouette:.2f}  (95% CI {ci_low:.2f}-{ci_high:.2f})\n"
        "1 = the two groups sit fully apart    0 = they overlap    "
        "below 0 = they are intermixed"
    )
