"""Reduce the multivariate {O_t} to the univariate semantic-drift series y_t.

Change-point detection runs on a 1-D signal, so forking-paths collapses the
outcome time series to y_t = d(o_0, o_t) — the L2 distance between the prior
outcome distribution o_0 and the position-t outcome distribution o_t (drift away
from the starting belief). Tiny Gaussian noise (variance 0.03) is added before
fitting to suppress spurious change points when the overall drift is small (BEAST
re-normalizes y internally, so flat regions otherwise register false positives).
"""

from __future__ import annotations

import numpy as np

from src.common.math import l2_distance

# Paper hyperparameter: Gaussian noise of variance 0.03 added to y_t.
DEFAULT_NOISE_VAR = 0.03


def outcome_drift_series(
    prior_histogram: list[float],
    outcome_histograms: list[list[float]],
) -> list[float]:
    """y_t = L2(o_0, o_t) for each base-path position t (no noise)."""
    return [l2_distance(prior_histogram, o_t) for o_t in outcome_histograms]


def add_drift_noise(
    drift_series: list[float], noise_var: float = DEFAULT_NOISE_VAR, seed: int = 0
) -> list[float]:
    """Add zero-mean Gaussian noise (variance ``noise_var``) to the drift series.

    Deterministic for a fixed ``seed`` so re-runs of the analysis are reproducible.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_var**0.5, size=len(drift_series))
    return [float(y + n) for y, n in zip(drift_series, noise)]
