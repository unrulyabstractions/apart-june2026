"""Map a bare model name (output-dir name) to its size in B params and family.

The cross-model size sweep plots accuracy vs parameter count, colored by family,
so every ``out/sesgo/baseline/<bare_name>/`` dir needs two derived facts: how big
the model is (x-axis, log scale) and which family it belongs to (one line/color).

Param counts mirror cloud/fleet_sizing.py's ladder (the same fleet that produces
these dirs); family is read off the bare name's leading token. New models the
fleet adds are picked up automatically as long as they follow these conventions —
unknown names resolve to ``None`` size so the caller can skip them gracefully.
"""

from __future__ import annotations

# Bare-name -> billions of parameters. Single source of truth for the x-axis,
# kept in lock-step with cloud/fleet_sizing._DEFAULT_MODELS so the plotted sizes
# match the launched fleet exactly.
_PARAMS_B: dict[str, float] = {
    # Qwen3 dense ladder
    "Qwen3-0.6B": 0.6,
    "Qwen3-1.7B": 1.7,
    "Qwen3-4B": 4.0,
    "Qwen3-8B": 8.0,
    "Qwen3-14B": 14.0,
    "Qwen3-32B": 32.0,
    # Llama 3.x
    "Llama-3.2-1B-Instruct": 1.2,
    "Llama-3.2-3B-Instruct": 3.2,
    "Llama-3.1-8B-Instruct": 8.0,
    # Gemma 2
    "gemma-2-2b-it": 2.6,
    "gemma-2-9b-it": 9.2,
    "gemma-2-27b-it": 27.0,
    # Mistral
    "Mistral-7B-Instruct-v0.3": 7.2,
    "Mistral-Small-24B-Instruct-2501": 24.0,
}

# Lower-cased name prefix -> human family label (color/marker grouping). Checked
# longest-first so e.g. "mistral" wins before any shorter accidental prefix.
_FAMILY_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("qwen", "Qwen"),
    ("llama", "Llama"),
    ("gemma", "Gemma"),
    ("mistral", "Mistral"),
)


def params_b(bare_name: str) -> float | None:
    """Billions of parameters for a bare model name, or ``None`` if unknown."""
    return _PARAMS_B.get(bare_name)


def family_of(bare_name: str) -> str | None:
    """Family label (Qwen/Llama/Gemma/Mistral) for a bare name, else ``None``."""
    lowered = bare_name.lower()
    for prefix, label in _FAMILY_BY_PREFIX:
        if lowered.startswith(prefix):
            return label
    return None
