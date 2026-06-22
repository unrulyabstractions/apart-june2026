"""Per-position diversity scores measured on the O_t outcome distribution.

Diversity at a prefix is read off the SAME sampled continuation set used for O_t
(Rios-Sialer Sec. 7), so each base-path position yields:

  balance      ρ_b = H(O_t)                    barycenter entropy (want high)
  disruption   ρ_s = ||O_t - O_0||_θ           default moved off the prior baseline
  mean_dev     E[∂_n]                          mean scalar deviance over rollouts
  var_dev      Var[∂_n]                        deviance spread over rollouts

Homogenization is E[∂_n] -> 0 AND Var[∂_n] -> 0 (mass collapses onto the default);
high diversity is high barycenter entropy with high deviance mean+variance. We
reuse the shared structure-aware estimators so the deviance norm matches the rest
of the dynamics pipeline (dimension-normalized L2).
"""

from __future__ import annotations

from src.common.math import (
    deviance_variance,
    expected_deviance,
    probs_to_logprobs,
    shannon_entropy,
)
from src.dynamics import normalized_norm

from .forking_outcome_set import ForkOutcomeSet
from .forking_path_types import DiversitySeries, ForkPosition


def _rollout_one_hots(
    position: ForkPosition, outcome_set: ForkOutcomeSet
) -> list[list[float]]:
    """One-hot R-vector per individual rollout across all of a position's alternates."""
    one_hots: list[list[float]] = []
    for alt in position.alternates:
        for label in alt.rollout_labels:
            one_hots.append(outcome_set.one_hot(label))
    return one_hots


def compute_diversity_series(
    prior_histogram: list[float],
    positions: list[ForkPosition],
    outcome_set: ForkOutcomeSet,
) -> DiversitySeries:
    """Balance / disruption / deviance-mean / deviance-var per base-path position."""
    balance: list[float] = []
    disruption: list[float] = []
    mean_dev: list[float] = []
    var_dev: list[float] = []

    for pos in positions:
        o_t = pos.outcome_histogram
        # Balance: Shannon entropy of the (already-normalized) barycenter O_t.
        balance.append(float(shannon_entropy(probs_to_logprobs(o_t))))
        # Disruption: how far the default moved off the prior baseline o_0.
        disruption.append(normalized_norm([a - b for a, b in zip(o_t, prior_histogram)]))
        # Deviance spread of the rollouts around their own barycenter O_t.
        one_hots = _rollout_one_hots(pos, outcome_set)
        mean_dev.append(expected_deviance(one_hots, o_t) if one_hots else 0.0)
        var_dev.append(deviance_variance(one_hots, o_t) if one_hots else 0.0)

    return DiversitySeries(
        balance=balance,
        disruption=disruption,
        mean_deviance=mean_dev,
        var_deviance=var_dev,
    )
