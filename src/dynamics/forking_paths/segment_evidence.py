"""Marginal likelihood of a piecewise-LINEAR-trend segmentation of y_t.

The forking-paths CPD model (y_t ~ Normal(beta_i*t + delta_i, sigma)) splits y
into m+1 contiguous segments, each a degree-1 linear regression. We score a
segmentation by its BAYESIAN marginal likelihood under a conjugate
Normal-Inverse-Gamma prior on (coefficients, variance): integrating BOTH out
gives a closed-form Student-t evidence whose built-in Occam factor resists
over-segmentation (unlike a plain BIC, a segment that fits exactly cannot earn
unbounded reward because the variance prior keeps sigma away from 0). A Bayesian
sampler (bayesian_change_point) explores configurations using this evidence + a
prior on the number of change points to recover p(tau=t|y) and p(m|y) — no Rbeast.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

# Normal-Inverse-Gamma hyperpriors. a0/b0 set a weak prior on the noise scale
# (floor ~ b0/a0); g controls the coefficient prior spread (Zellner g-prior).
_A0 = 1.0
_B0 = 0.005  # noise-variance prior scale (calibrated: clean step BF>9, flat m=0)
_G = 10.0


def _segment_log_evidence(y: np.ndarray, t0: int, t1: int) -> float:
    """Log marginal likelihood of segment y[t0:t1] under a Bayesian linear trend.

    Conjugate Normal-Inverse-Gamma evidence for y ~ N(X beta, sigma^2 I) with a
    g-prior on beta and an Inverse-Gamma(a0, b0) on sigma^2. Closed-form Student-t
    marginal; the variance prior gives every segment a finite reward even when it
    fits exactly, so adding change points is genuinely penalized.
    """
    n = t1 - t0
    if n <= 0:
        return 0.0
    ts = np.arange(t0, t1, dtype=float)
    ys = y[t0:t1]
    X = np.column_stack([ts, np.ones(n)])  # [slope, intercept] design

    xtx = X.T @ X + (1.0 / _G) * np.eye(2)
    xty = X.T @ ys
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        beta = np.zeros(2)
    resid = ys - X @ beta
    # Posterior IG shape/scale after seeing the segment.
    a_n = _A0 + 0.5 * n
    b_n = _B0 + 0.5 * (float(resid @ resid) + (1.0 / _G) * float(beta @ beta))

    sign, logdet = np.linalg.slogdet(xtx)
    # log p(y) = const + a0*log b0 - a_n*log b_n + lgamma(a_n) - lgamma(a0)
    #            - 0.5*log|XtX + I/g| - 0.5*log|I/g|  (the prior-precision det)
    log_ev = (
        -0.5 * n * math.log(2.0 * math.pi)
        + _A0 * math.log(_B0)
        - a_n * math.log(b_n)
        + gammaln(a_n)
        - gammaln(_A0)
        - 0.5 * logdet
        + 0.5 * 2.0 * math.log(1.0 / _G)
    )
    return float(log_ev)


def configuration_log_evidence(y: np.ndarray, change_points: list[int]) -> float:
    """Total log evidence of a segmentation defined by interior change points.

    ``change_points`` are interior boundaries (1..T-1, sorted, unique). Segments
    are [0, c1), [c1, c2), ..., [c_m, T). Boundaries 0 and T are fixed and
    excluded from the change-point set (paper boundary handling).
    """
    n = len(y)
    bounds = [0] + sorted(set(change_points)) + [n]
    return sum(
        _segment_log_evidence(y, bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
    )
