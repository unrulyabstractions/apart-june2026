"""Shared style + ordering helpers for the cross-model DISTRIBUTION figures.

One Okabe-Ito palette (per family and per role), one size-ordering of models, and
one compact x-tick labeller, so every distribution figure agrees on colour, order
and labels. Partial models (fewer items than the full 2310-item run) get a ``*``
appended to their tick label and an ``n=`` annotation, never a silent omission.
"""

from __future__ import annotations

from sesgo.baseline.cross_model_distribution_stats import ModelDistribution

# Okabe-Ito colorblind-safe family hues (matches cross_model_plotting.py).
FAMILY_COLORS: dict[str, str] = {
    "Qwen": "#0072B2", "Llama": "#D55E00", "Gemma": "#009E73", "Mistral": "#CC79A7",
}
# Okabe-Ito role hues for the stacked outcome-distribution bars.
ROLE_COLORS: dict[str, str] = {
    "target": "#D55E00", "other": "#56B4E9", "unknown": "#009E73",
}
# Full SESGO baseline run size; models below it are flagged partial (``*``).
_FULL_N = 2310


def order_by_size(models: list[ModelDistribution]) -> list[ModelDistribution]:
    """Models sorted by parameter count (then name) — the canonical x-order."""
    return sorted(models, key=lambda m: (m.params_b, m.model))


def is_partial(m: ModelDistribution) -> bool:
    """True if this model's run is smaller than a complete baseline sweep."""
    return (m.n_ambig + m.n_disambig) < _FULL_N


def tick_label(m: ModelDistribution) -> str:
    """Compact single-line tick: ``size · model`` (``*`` appended if partial).

    Size leads so the left-to-right size ordering stays obvious even when the
    long model name dominates; a trailing ``*`` flags a partial run.
    """
    star = "*" if is_partial(m) else ""
    return f"{m.params_b:g}B · {m.model}{star}"


def family_color(m: ModelDistribution) -> str:
    """Okabe-Ito hue for a model's family (grey fallback for unknowns)."""
    return FAMILY_COLORS.get(m.family, "#555555")


def partial_note(models: list[ModelDistribution]) -> str:
    """Footnote naming any partial models, or '' when every run is complete."""
    partials = [f"{m.model} (n={m.n_ambig + m.n_disambig})"
                for m in models if is_partial(m)]
    if not partials:
        return ""
    return "* partial run (fewer items): " + ", ".join(partials)
