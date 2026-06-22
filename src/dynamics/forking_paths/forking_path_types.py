"""BaseSchema records for the forking-paths O_t pipeline (capture -> analysis).

These typed records are the serialization boundary between the three drivers:
``collect_forking_rollouts`` writes a ``ForkingTrajectory`` (one ``ForkPosition``
per base-path token), and ``analyze_forking_dynamics`` reads it back to produce
the change-point / dynamic-state / diversity / survival series. Every field is a
flat list/scalar (no nested dict/list) so each roundtrips cleanly via BaseSchema.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema

from .forking_outcome_set import ForkOutcomeSet


@dataclass
class AltTokenRollouts(BaseSchema):
    """The rollouts taken after forcing one alternate token w at a position.

    ``token_prob`` is p(x_t=w | x*_{<t}) (the next-token weight for w). Each
    rollout is parsed to an outcome label; ``conditional_histogram`` is o_{t,w}
    (Eq. 1) — the prob-weighted outcome histogram over this token's rollouts.
    """

    token_id: int
    token_text: str
    token_prob: float
    is_base_token: bool  # True for the greedy base-path token w*
    rollout_labels: list[str]  # parsed outcome label per sampled continuation
    conditional_histogram: list[float]  # o_{t,w}: prob-weighted outcome vector


@dataclass
class ForkPosition(BaseSchema):
    """One base-path token position t with its branching outcome distribution.

    ``position`` t indexes the base-path token list. ``alternates`` holds every
    sufficiently-probable alternate token w (>= prob floor, top-k) with its
    rollouts. ``outcome_histogram`` is o_t (Eq. 2): the alternates' o_{t,w}
    averaged, weighted by token_prob and renormalized to a distribution.
    """

    position: int
    base_token_id: int
    base_token_text: str
    alternates: list[AltTokenRollouts] = field(default_factory=list)
    outcome_histogram: list[float] = field(default_factory=list)


@dataclass
class ForkingTrajectory(BaseSchema):
    """The full captured {O_t} series for ONE prompt's base thinking path.

    ``base_token_texts`` are the rendered base-path tokens (for the plot's text
    strip); ``prompt_token_count`` separates uncolored prompt tokens from the
    base-path tokens the change-point posterior colors. ``prior_histogram`` is
    o_0 (the N-sample full-resample baseline), the drift reference frame.
    """

    item_question_id: str
    model: str
    outcome_set: ForkOutcomeSet
    prompt_text: str
    base_path_text: str
    base_token_texts: list[str]
    prompt_token_count: int
    prior_histogram: list[float]  # o_0: full-resample prior outcome distribution
    final_histogram: list[float]  # o_T: outcome dist at the end of the base path
    positions: list[ForkPosition] = field(default_factory=list)

    @property
    def outcome_series(self) -> list[list[float]]:
        """[o_0, o_{t=0}, o_{t=1}, ...]: the {O_t} multivariate time series."""
        return [self.prior_histogram] + [p.outcome_histogram for p in self.positions]


@dataclass
class ChangePointResult(BaseSchema):
    """Bayesian multiple-change-point posteriors over the semantic-drift series.

    ``tau_posterior[t]`` is p(tau=t | y) (probability a change point sits at
    base-path position t); ``num_changepoints_posterior[m]`` is p(m | y).
    ``bayes_factor`` is p(m>=1|y)/p(m=0|y); ``forking_token_index`` is the argmax
    of tau_posterior (the detected forking token), -1 when none is significant.
    """

    drift_series: list[float]  # y_t = L2(o_0, o_t) (+ noise), per base-path token
    tau_posterior: list[float]  # p(tau=t | y)
    num_changepoints_posterior: list[float]  # p(m | y), index m = # change points
    bayes_factor: float
    forking_token_index: int
    significant: bool  # bayes_factor > 9


@dataclass
class DynamicStatesSeries(BaseSchema):
    """Pull / drift / potential magnitudes along the O_t trajectory (App. H).

    Each list is indexed by base-path position t. The magnitudes are the
    dimension-normalized L2 norms of the three vector states; ``forking_magnitude``
    is Delta_t = ||O_t - O_{t-1}||_theta (consecutive-barycenter first difference).
    """

    pull: list[float]
    drift: list[float]
    potential: list[float]
    forking_magnitude: list[float]
    most_forking_index: int  # argmax_t forking_magnitude


@dataclass
class DiversitySeries(BaseSchema):
    """Per-position diversity scores measured on the O_t outcome distribution.

    ``balance`` is barycenter entropy H(o_t); ``disruption`` is ||o_t - o_0||_theta;
    ``mean_deviance`` / ``var_deviance`` are E[partial_n] / Var[partial_n] over the
    position's rollouts (homogenization collapses both to 0).
    """

    balance: list[float]
    disruption: list[float]
    mean_deviance: list[float]
    var_deviance: list[float]


@dataclass
class SurvivalSeries(BaseSchema):
    """Hazard + survival along the base path (forking-paths Eq. 3).

    ``hazard[t]`` is the alternate-token probability mass whose o_{t,w} diverges
    (L2 > epsilon) from the base token's o_{t,w*}; ``survival[t]`` is the running
    product prod_{t'<=t} (1 - hazard[t']).
    """

    hazard: list[float]
    survival: list[float]
    epsilon: float
