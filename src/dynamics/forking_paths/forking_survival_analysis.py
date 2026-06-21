"""Survival analysis: counterfactual influence of the un-sampled tokens (Eq. 3).

Complementary to change-point detection: rather than asking where the realized
path's outcome shifts, survival asks how much un-taken alternate-token mass would
have diverted the outcome. The hazard at position t is the next-token probability
mass of alternates w whose conditional outcome o_{t,w} differs from the base
token's o_{t,w*} by more than epsilon (L2):

  h(t) = Σ_w p(x_t=w) · 1[ ||o_{t,w} - o_{t,w*}||_2 > epsilon ]

and the survival is the running product S(t) = Π_{t'<=t} (1 - h(t')) — the
probability the base path "survives" through position t with no outcome-changing
alternate. Thresholds are hardcoded from the paper: epsilon = 0.6, L2 distance.
"""

from __future__ import annotations

from src.common.math import l2_distance

from .forking_path_types import ForkPosition, SurvivalSeries

# Paper hyperparameter: survival hazard divergence threshold (L2).
DEFAULT_EPSILON = 0.6


def _base_conditional(position: ForkPosition) -> list[float] | None:
    """The base token's o_{t,w*} (the flagged alternate), or None if absent."""
    for alt in position.alternates:
        if alt.is_base_token:
            return alt.conditional_histogram
    return None


def _position_hazard(position: ForkPosition, epsilon: float) -> float:
    """h(t): alternate-token prob mass whose o_{t,w} diverges from o_{t,w*}."""
    base = _base_conditional(position)
    if base is None:
        return 0.0
    hazard = 0.0
    for alt in position.alternates:
        if alt.is_base_token:
            continue
        if l2_distance(alt.conditional_histogram, base) > epsilon:
            hazard += alt.token_prob
    return min(hazard, 1.0)


def compute_survival(
    positions: list[ForkPosition], epsilon: float = DEFAULT_EPSILON
) -> SurvivalSeries:
    """Hazard + cumulative survival S(t) along the base path."""
    hazard: list[float] = []
    survival: list[float] = []
    running = 1.0
    for pos in positions:
        h = _position_hazard(pos, epsilon)
        running *= 1.0 - h
        hazard.append(h)
        survival.append(running)
    return SurvivalSeries(hazard=hazard, survival=survival, epsilon=epsilon)
