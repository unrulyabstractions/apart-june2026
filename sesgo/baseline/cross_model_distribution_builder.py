"""Build one ``ModelDistribution`` from a model's SESGO baseline samples.

Splits the model's samples into ambiguous (gold=UNKNOWN -> abstention regime) and
disambiguated (gold=TARGET/OTHER -> accuracy regime) and reduces each into the flat
distributional fields the cross-model figures plot: the per-item abstention spread,
the mean role-mass split, per-category abstention counts, the target-vs-other bias
gap, and the three-readout abstention agreement. Models we can't size/place yield
``None`` so an unexpected output dir is skipped, never crashing the sweep.
"""

from __future__ import annotations

from src.datasets.sesgo import SesgoLabel
from src.datasets.sesgo_eval import SesgoSample
from sesgo.baseline.cross_model_distribution_stats import (
    CATEGORY_ORDER,
    ModelDistribution,
    _abstains_greedy,
    _p_gold,
    _p_unknown,
    _role_mass_means,
    is_degenerate_readout,
)
from sesgo.baseline.sesgo_model_sizing import family_of, params_b


def _count(flags: list[bool | None]) -> tuple[int, int]:
    """Successes and usable n over a flag list (None readouts don't count)."""
    usable = [f for f in flags if f is not None]
    return sum(bool(f) for f in usable), len(usable)


def _category_abstention(ambig: list[SesgoSample]) -> tuple[list[int], list[int]]:
    """Per-category (successes, total) 3-opt abstention in CATEGORY_ORDER."""
    succ, total = [], []
    for cat in CATEGORY_ORDER:
        grp = [s for s in ambig if s.bias_category == cat]
        s, t = _count([s.correct_non_thinking for s in grp])
        succ.append(s)
        total.append(t)
    return succ, total


def _disambig_gap(disambig: list[SesgoSample]) -> tuple[int, int, int, int]:
    """3-opt accuracy split by gold role: (tgt_succ, tgt_tot, oth_succ, oth_tot)."""
    tgt = [s for s in disambig if s.gold_label is SesgoLabel.TARGET]
    oth = [s for s in disambig if s.gold_label is SesgoLabel.OTHER]
    ts, tt = _count([s.correct_non_thinking for s in tgt])
    os_, ot = _count([s.correct_non_thinking for s in oth])
    return ts, tt, os_, ot


def _readout_abstention(ambig: list[SesgoSample]) -> tuple[int, int, int, int, int, int]:
    """Ambiguous abstention counts for (3-opt, 2-opt, greedy-thinking).

    On ambiguous items gold IS unknown, so 3-opt ``correct_non_thinking`` is the
    abstention indicator. 2-opt has no UNKNOWN option, so it can never abstain
    (succ fixed to 0 over its usable n) — a structural zero the plot annotates.
    Greedy uses the parsed reasoning decode's UNKNOWN commitment.
    """
    s3, t3 = _count([s.correct_non_thinking for s in ambig])
    t2 = sum(1 for s in ambig if s.non_thinking_2opt is not None)
    sg, tg = _count([_abstains_greedy(s) for s in ambig])
    return s3, t3, 0, t2, sg, tg


def build_model_distribution(
    model: str, samples: list[SesgoSample]
) -> ModelDistribution | None:
    """Reduce one model's samples into its cross-model distribution bundle."""
    fam, size = family_of(model), params_b(model)
    if fam is None or size is None:
        return None
    ambig = [s for s in samples if s.context_condition == "ambig"]
    disambig = [s for s in samples if s.context_condition == "disambig"]
    if is_degenerate_readout(ambig):  # all-uniform 3-opt logits -> broken run
        return None
    p_unknown = [v for v in (_p_unknown(s) for s in ambig) if v is not None]
    p_gold = [v for v in (_p_gold(s) for s in disambig) if v is not None]
    cat_s, cat_t = _category_abstention(ambig)
    ts, tt, os_, ot = _disambig_gap(disambig)
    s3, t3, s2, t2, sg, tg = _readout_abstention(ambig)
    return ModelDistribution(
        model=model, family=fam, params_b=size,
        n_ambig=len(p_unknown), n_disambig=len(p_gold),
        mean_role_mass=_role_mass_means(ambig),
        p_unknown_ambig=p_unknown, p_gold_disambig=p_gold,
        cat_abstain_succ=cat_s, cat_abstain_total=cat_t,
        target_succ=ts, target_total=tt, other_succ=os_, other_total=ot,
        abstain_3opt_succ=s3, abstain_3opt_total=t3,
        abstain_2opt_succ=s2, abstain_2opt_total=t2,
        abstain_greedy_succ=sg, abstain_greedy_total=tg,
    )
