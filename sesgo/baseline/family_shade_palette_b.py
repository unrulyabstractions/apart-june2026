"""Family-hue x size-shade colours for the redesigned bias-alignment figure.

Each model is one POINT whose colour encodes two things at once:

  * FAMILY -> HUE: the Okabe-Ito family colour (Qwen / Llama / Gemma / Mistral),
    reusing ``FAMILY_COLORS`` so every cross-model figure agrees on hue.
  * SIZE -> SHADE: lightness set by parameter count -- smaller models lighter,
    larger models darker. Shading is computed PER FAMILY (each family's own size
    range maps onto the same lightness band) so within a family the size ordering
    reads cleanly, and across families the hue still separates them.

The shade is produced by blending the family hue toward white (small) or toward a
darkened version of itself (large) in RGB, which keeps the hue recognisable while
walking lightness. Returns a flat ``{model_name -> hex}`` map.
"""

from __future__ import annotations

from matplotlib.colors import to_hex, to_rgb

from sesgo.baseline.cross_model_plot_styles import FAMILY_COLORS
from sesgo.baseline.sesgo_model_sizing import family_of, params_b

import math

# Lightness band: smallest model in a family blends this far toward white; the
# largest blends this far toward (a darkened) hue. Kept off the extremes so even
# the lightest dot stays legible against the pastel triangle and the white edge.
_LIGHT_MAX = 0.55  # fraction toward white for the smallest model
_DARK_MAX = 0.32  # fraction toward black for the largest model
_GREY = "#555555"


def _blend(rgb: tuple[float, float, float], toward: float, amount: float) -> tuple:
    """Blend ``rgb`` a fraction ``amount`` toward a grey of level ``toward``."""
    return tuple(c + (toward - c) * amount for c in rgb)


def _shade(hue_hex: str, frac: float) -> str:
    """Hue at a size fraction in [0, 1] (0 = smallest/lightest, 1 = largest/darkest)."""
    rgb = to_rgb(hue_hex)
    if frac <= 0.5:
        # Lower half: lighten toward white, strongest at frac = 0.
        amount = (0.5 - frac) / 0.5 * _LIGHT_MAX
        return to_hex(_blend(rgb, 1.0, amount))
    # Upper half: darken toward black, strongest at frac = 1.
    amount = (frac - 0.5) / 0.5 * _DARK_MAX
    return to_hex(_blend(rgb, 0.0, amount))


def _size_fraction(size: float, lo: float, hi: float) -> float:
    """Map a (log) param count onto [0, 1] within a family's size range."""
    if hi <= lo:
        return 0.5  # single model in the family -> the mid (base hue) shade
    return (math.log(size) - math.log(lo)) / (math.log(hi) - math.log(lo))


def model_colors(models: list[str]) -> dict[str, str]:
    """``{model -> hex}`` with family hue shaded by size (per-family normalised)."""
    sizes = {m: params_b(m) for m in models}
    families = {m: family_of(m) for m in models}
    by_family: dict[str, list[str]] = {}
    for m in models:
        by_family.setdefault(families[m] or "", []).append(m)
    out: dict[str, str] = {}
    for fam, members in by_family.items():
        hue = FAMILY_COLORS.get(fam, _GREY)
        valid = [m for m in members if sizes[m]]
        lo = min((sizes[m] for m in valid), default=1.0)
        hi = max((sizes[m] for m in valid), default=1.0)
        for m in members:
            size = sizes[m]
            frac = _size_fraction(size, lo, hi) if size else 0.5
            out[m] = _shade(hue, frac)
    return out
