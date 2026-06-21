"""Per-model DISTRIBUTIONAL summaries of one SESGO baseline model's samples.

The headline size-sweep collapses each model to a few binomial accuracy points;
these cross-model DISTRIBUTION figures need richer per-model statistics — the full
spread of per-item abstention mass, the mean role mass split, per-category
abstention, the target-vs-other bias gap, and readout (3-opt/2-opt/greedy)
agreement. One ``ModelDistribution`` bundles all of that for one model so the plot
layer can compare the *shapes*, not just the means, across the sweep.

All vectors stay flat (lists of floats) and counts stay flat (lists of ints in a
fixed category order) so nothing nests deeper than a 1-D list — every structured
field is a BaseSchema. Missing readouts shrink n rather than crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema
from src.datasets.sesgo import SesgoLabel
from src.datasets.sesgo_eval import SesgoSample

# Canonical role order shared with SesgoNonThinking.prob = [TARGET, OTHER, UNKNOWN].
ROLE_NAMES: tuple[str, ...] = ("target", "other", "unknown")
# Fixed bias-category order for the per-category abstention heatmap rows/cols.
CATEGORY_ORDER: tuple[str, ...] = ("clasismo", "racismo", "xenofobia", "genero")
# English display labels for the category axis (corpus stems -> readable).
CATEGORY_ENGLISH: dict[str, str] = {
    "clasismo": "Classism", "racismo": "Racism",
    "xenofobia": "Xenophobia", "genero": "Gender",
}


@dataclass
class ModelDistribution(BaseSchema):
    """All cross-model DISTRIBUTION statistics for one baseline model."""

    model: str
    family: str
    params_b: float
    n_ambig: int  # ambiguous items with a usable 3-opt readout (the p_unknown n)
    n_disambig: int  # disambiguated items with a usable 3-opt readout
    # Mean 3-opt role mass over ambiguous items, in ROLE_NAMES order (sums ~1).
    mean_role_mass: list[float] = field(default_factory=list)
    # Per-item 3-opt p_unknown on ambiguous items — the abstention SPREAD.
    p_unknown_ambig: list[float] = field(default_factory=list)
    # Per-item 3-opt probability mass on the GOLD role (disambiguated) — a soft,
    # continuous accuracy whose distribution shows confident-correct vs -wrong.
    p_gold_disambig: list[float] = field(default_factory=list)
    # Per-category ambiguous abstention as (successes, total) in CATEGORY_ORDER.
    cat_abstain_succ: list[int] = field(default_factory=list)
    cat_abstain_total: list[int] = field(default_factory=list)
    # Disambiguated 3-opt accuracy split by gold role: TARGET then OTHER.
    target_succ: int = 0
    target_total: int = 0
    other_succ: int = 0
    other_total: int = 0
    # Ambiguous abstention rate per readout: 3-opt, 2-opt, greedy-thinking.
    abstain_3opt_succ: int = 0
    abstain_3opt_total: int = 0
    abstain_2opt_succ: int = 0
    abstain_2opt_total: int = 0
    abstain_greedy_succ: int = 0
    abstain_greedy_total: int = 0


def _p_unknown(sample: SesgoSample) -> float | None:
    """3-opt UNKNOWN probability mass for one item, or None if no readout."""
    nt = sample.non_thinking
    return float(nt.prob[ROLE_NAMES.index("unknown")]) if nt else None


def _p_gold(sample: SesgoSample) -> float | None:
    """3-opt probability mass on this item's GOLD role, or None if no readout."""
    nt = sample.non_thinking
    if nt is None:
        return None
    return float(nt.prob[ROLE_NAMES.index(sample.gold_label.value)])


def _role_mass_means(ambig: list[SesgoSample]) -> list[float]:
    """Mean 3-opt prob mass per role over ambiguous items (ROLE_NAMES order)."""
    vecs = [s.non_thinking.prob for s in ambig if s.non_thinking]
    if not vecs:
        return [float("nan")] * len(ROLE_NAMES)
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(len(ROLE_NAMES))]


def _abstains_greedy(sample: SesgoSample) -> bool | None:
    """Whether the greedy-thinking decode abstained (chose UNKNOWN), else None."""
    pred = sample.predicted_greedy_thinking
    return None if pred is None else pred is SesgoLabel.UNKNOWN


def is_degenerate_readout(ambig: list[SesgoSample], tol: float = 1e-6) -> bool:
    """True if a model's 3-opt readout collapsed to uniform [⅓,⅓,⅓] everywhere.

    A broken teacher-forced run (identical logprobs for all three options) yields
    an all-uniform prob vector with no signal — its mean role mass, abstention and
    bias gap are artifacts, not measurements. We detect and exclude such a model
    rather than plot a misleading flat bar. Healthy models have near-zero uniform
    fraction; a fully-uniform run trips this on every item.
    """
    vecs = [s.non_thinking.prob for s in ambig if s.non_thinking]
    if not vecs:
        return False
    uniform = sum(
        1 for v in vecs if all(abs(p - 1.0 / len(ROLE_NAMES)) < tol for p in v)
    )
    return uniform == len(vecs)
