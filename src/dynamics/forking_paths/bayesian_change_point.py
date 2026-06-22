"""Bayesian multiple-change-point detection on the semantic-drift series y_t.

Faithful to the forking-paths CPD model (piecewise-linear trend, Gaussian
observation noise) without the external Rbeast dependency: a reversible-jump MCMC
proposes birth / death / move of interior change points, accepts by the
Metropolis ratio of the configuration evidence (segment_evidence) times a prior
that penalizes additional change points, and accumulates the posteriors
p(tau=t|y) (visit frequency of each interior position) and p(m|y) (frequency of
each change-point count). The Bayes factor p(m>=1|y)/p(m=0|y) > 9 declares a
forking token, localized at argmax_t p(tau=t|y). Boundaries 0 and T are fixed and
excluded from the posteriors (paper boundary handling).
"""

from __future__ import annotations

import math

import numpy as np

from .forking_path_types import ChangePointResult
from .segment_evidence import configuration_log_evidence

# Thresholds hardcoded from the paper.
BAYES_FACTOR_THRESHOLD = 9.0
# Geometric prior penalty per change point (favours parsimony, controls false +).
# Calibrated so a clean outcome step clears BF>9 while a flat series stays at m=0.
DEFAULT_CHANGE_PENALTY = 2.0


def _log_prior(n_changes: int, penalty: float) -> float:
    """Geometric prior on the number of change points: p(m) ∝ exp(-penalty*m)."""
    return -penalty * n_changes


def _propose(rng, current: list[int], n: int) -> list[int]:
    """Birth / death / move proposal over interior change-point positions (1..n-1)."""
    interior = list(range(1, n))
    if not interior:
        return current
    move = rng.integers(0, 3)
    cur = set(current)
    if move == 0 or not cur:  # birth
        free = [p for p in interior if p not in cur]
        if free:
            cur.add(int(rng.choice(free)))
    elif move == 1:  # death
        cur.discard(int(rng.choice(list(cur))))
    else:  # move one change point to a free slot
        free = [p for p in interior if p not in cur]
        if free and cur:
            cur.discard(int(rng.choice(list(cur))))
            cur.add(int(rng.choice(free)))
    return sorted(cur)


def detect_change_points(
    drift_series: list[float],
    n_iter: int = 4000,
    burn_in: int = 1000,
    penalty: float = DEFAULT_CHANGE_PENALTY,
    seed: int = 0,
) -> ChangePointResult:
    """Run RJ-MCMC and return change-point posteriors over the drift series."""
    y = np.asarray(drift_series, dtype=float)
    n = len(y)
    tau_hist = [0.0] * n
    max_m = max(1, n - 1)
    m_hist = [0.0] * (max_m + 1)

    if n < 3:  # too short to host an interior change point
        return ChangePointResult(
            drift_series=list(drift_series),
            tau_posterior=[0.0] * n,
            num_changepoints_posterior=[1.0] + [0.0] * max_m,
            bayes_factor=0.0,
            forking_token_index=-1,
            significant=False,
        )

    rng = np.random.default_rng(seed)
    current: list[int] = []
    cur_score = configuration_log_evidence(y, current) + _log_prior(0, penalty)
    kept = 0
    for it in range(n_iter):
        prop = _propose(rng, current, n)
        prop_score = configuration_log_evidence(y, prop) + _log_prior(len(prop), penalty)
        if math.log(rng.random() + 1e-300) < prop_score - cur_score:
            current, cur_score = prop, prop_score
        if it >= burn_in:
            kept += 1
            m_hist[len(current)] += 1.0
            for c in current:
                tau_hist[c] += 1.0

    denom = max(kept, 1)
    tau_post = [h / denom for h in tau_hist]
    m_post = [h / denom for h in m_hist]
    p_zero = max(m_post[0], 1e-9)
    bayes_factor = (1.0 - m_post[0]) / p_zero
    significant = bayes_factor > BAYES_FACTOR_THRESHOLD
    fork_idx = int(np.argmax(tau_post)) if significant and max(tau_post) > 0 else -1
    return ChangePointResult(
        drift_series=list(drift_series),
        tau_posterior=tau_post,
        num_changepoints_posterior=m_post,
        bayes_factor=float(bayes_factor),
        forking_token_index=fork_idx,
        significant=significant,
    )
