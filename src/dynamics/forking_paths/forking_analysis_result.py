"""The combined analysis bundle produced from a captured ForkingTrajectory.

Aggregates the four downstream analyses (change-point detection, dynamic states,
diversity series, survival) into ONE serializable record so the analysis driver
writes a single forking_analysis.json that the plotting driver consumes. The
top-level orchestration (analyze_forking_trajectory) reads like a high-level
description: drift -> change points -> states -> diversity -> survival.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.base_schema import BaseSchema

from .bayesian_change_point import detect_change_points
from .forking_diversity_series import compute_diversity_series
from .forking_dynamic_states import compute_dynamic_states
from .forking_path_types import (
    ChangePointResult,
    DiversitySeries,
    DynamicStatesSeries,
    ForkingTrajectory,
    SurvivalSeries,
)
from .forking_survival_analysis import compute_survival
from .semantic_drift_series import add_drift_noise, outcome_drift_series


@dataclass
class ForkingAnalysis(BaseSchema):
    """Change-point + dynamic-state + diversity + survival series for a trajectory."""

    item_question_id: str
    model: str
    change_points: ChangePointResult
    dynamic_states: DynamicStatesSeries
    diversity: DiversitySeries
    survival: SurvivalSeries


def analyze_forking_trajectory(
    traj: ForkingTrajectory,
    n_iter: int = 6000,
    burn_in: int = 1500,
    noise_seed: int = 0,
) -> ForkingAnalysis:
    """Run the full forking-paths analysis over a captured {O_t} trajectory."""
    o_t_series = [p.outcome_histogram for p in traj.positions]

    # Univariate semantic drift y_t = L2(o_0, o_t) (+ tiny noise) -> change points.
    drift = outcome_drift_series(traj.prior_histogram, o_t_series)
    noisy = add_drift_noise(drift, seed=noise_seed)
    change_points = detect_change_points(noisy, n_iter=n_iter, burn_in=burn_in)

    states = compute_dynamic_states(
        traj.prior_histogram, o_t_series, traj.final_histogram
    )
    diversity = compute_diversity_series(
        traj.prior_histogram, traj.positions, traj.outcome_set
    )
    survival = compute_survival(traj.positions)

    return ForkingAnalysis(
        item_question_id=traj.item_question_id,
        model=traj.model,
        change_points=change_points,
        dynamic_states=states,
        diversity=diversity,
        survival=survival,
    )
