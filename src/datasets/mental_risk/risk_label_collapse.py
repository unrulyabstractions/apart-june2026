"""Collapse the MentalRiskES gold label set into a single risk score.

The corpus annotates each subject with several binary/fractional label columns.
For downstream use we want one continuous risk score in [0, 1]. We prefer the
annotator-agreement fraction `rbs` ("ratio of annotators marking suffers") and
fall back to the binary `bs` flag when the fraction is unavailable.
"""

from __future__ import annotations


def collapse_risk(labels: dict[str, float]) -> float | None:
    """Collapse a gold label dict to a single risk score in [0, 1].

    Preference order: `rbs` (annotator fraction) -> `bs` (binary suffers flag).
    Returns None when neither is present so callers can distinguish "unlabelled"
    from a genuine zero risk.
    """
    for key in ("rbs", "bs"):
        if key in labels and labels[key] is not None:
            # Clamp defensively: fractions should already be in range, but a
            # malformed gold file must never leak out-of-range scores.
            return max(0.0, min(1.0, float(labels[key])))
    return None
