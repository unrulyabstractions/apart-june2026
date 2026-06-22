"""Detect a degenerate model response (a repetition loop / garbage, e.g. a backend
silently mis-generating on an unsupported arch, or a CoT stuck re-deriving the same
paragraph until the token cap). Shared by the HF/MLX and vLLM readout runners so the
same notion of "this is not data" is applied identically on every backend."""

from __future__ import annotations


def is_degenerate(text: str, min_len: int = 40) -> bool:
    """True if `text` is a short repetition loop / near-no-information garbage."""
    t = text.strip()
    if len(t) < min_len:
        return False
    if len(set(t)) <= 3:  # almost no unique characters
        return True
    for period in range(1, 11):  # a short cycle that explains >=85% of the text
        unit = t[:period]
        if unit and t.count(unit) * period >= 0.85 * len(t):
            return True
    # A long-block repetition loop (a CoT stuck re-deriving the same paragraph until the
    # token cap): many non-empty lines but few DISTINCT ones.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 12 and len(set(lines)) < len(lines) * 0.5:
        return True
    return False
