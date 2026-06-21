"""Aggregate per-model SESGO baseline accuracy into plottable size-sweep points.

One ``out/sesgo/baseline/<bare_name>/response_samples.json`` is one model. For the
headline figure we collapse each model's samples into a handful of binomial
accuracy points — one per (condition, slice) — carrying successes/total so the
plot layer can attach a Wilson CI and annotate n. Two slice families:

  * AMBIGUOUS abstention: ``context_condition == "ambig"`` (gold is UNKNOWN), so
    accuracy == abstention rate.
  * DISAMBIGUATED, split by gold role into TARGET-gold and OTHER-gold (the gap
    between the two is the bias-by-size signal).

Each condition reads a different per-sample correctness property:
  * ``correct_non_thinking``  (3-option teacher-forced),
  * ``correct_2opt``          (2-option forced choice; None on ambiguous items),
  * ``correct_greedy_thinking`` (single greedy reasoning decode).
Missing readouts simply yield smaller n (or an empty point we skip), never a crash.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.common.math import wilson_interval
from src.datasets.sesgo import SesgoLabel
from src.datasets.sesgo_eval import SesgoSample
from sesgo.baseline.sesgo_model_sizing import family_of, params_b

# Condition key -> the SesgoSample correctness attribute it scores. Stable order
# fixes the subplot row order in the figure (3-opt, 2-opt, greedy-thinking).
CONDITIONS: tuple[tuple[str, str], ...] = (
    ("non_thinking", "correct_non_thinking"),
    ("non_thinking_2opt", "correct_2opt"),
    ("greedy_thinking", "correct_greedy_thinking"),
)
# Human-readable subplot titles for each condition, shared across baseline plots
# so the single-model and cross-model figures label their readouts identically.
COND_TITLES: dict[str, str] = {
    "non_thinking": "3-option (non_thinking, teacher-forced)",
    "non_thinking_2opt": "2-option (non_thinking_2opt, forced choice)",
    "greedy_thinking": "greedy_thinking (single reasoning decode)",
}
# Disambiguated gold roles we split accuracy by (the target-vs-other gap signal).
DISAMBIG_GOLDS: tuple[tuple[str, SesgoLabel], ...] = (
    ("target", SesgoLabel.TARGET),
    ("other", SesgoLabel.OTHER),
)


@dataclass
class CrossModelPoint(BaseSchema):
    """One binomial accuracy point: a model at a size, for one condition/slice."""

    model: str  # bare model name (output-dir name)
    family: str  # Qwen / Llama / Gemma / Mistral
    params_b: float  # parameter count in billions (x-axis, log scale)
    condition: str  # non_thinking / non_thinking_2opt / greedy_thinking
    slice_label: str  # ambig / disambig-target / disambig-other
    successes: int  # correct judgements
    total: int  # judgements with a usable readout (the n)

    @property
    def accuracy(self) -> float:
        """Point estimate (Wilson center is for the error bar, not the dot)."""
        return self.successes / self.total if self.total else float("nan")

    @property
    def wilson(self) -> tuple[float, float, float]:
        """``(p_hat, lo, hi)`` Wilson 95% interval over successes/total."""
        return wilson_interval(self.successes, self.total)


def _count(samples: list[SesgoSample], attr: str) -> tuple[int, int]:
    """Successes and usable n for a correctness attr (None readouts don't count)."""
    flags = [getattr(s, attr) for s in samples]
    usable = [f for f in flags if f is not None]
    return sum(bool(f) for f in usable), len(usable)


def _point(
    model: str, fam: str, size: float, cond: str, slice_label: str,
    samples: list[SesgoSample], attr: str,
) -> CrossModelPoint | None:
    """Build a point for one (condition, slice); ``None`` if no usable readouts."""
    successes, total = _count(samples, attr)
    if total == 0:
        return None
    return CrossModelPoint(model, fam, size, cond, slice_label, successes, total)


def points_for_model(model: str, samples: list[SesgoSample]) -> list[CrossModelPoint]:
    """All (condition x slice) accuracy points for one model's samples.

    Returns ``[]`` for models we can't size or place in a family (so an unexpected
    output dir is skipped, never crashes the sweep).
    """
    fam, size = family_of(model), params_b(model)
    if fam is None or size is None:
        return []
    ambig = [s for s in samples if s.context_condition == "ambig"]
    disambig = [s for s in samples if s.context_condition == "disambig"]
    out: list[CrossModelPoint] = []
    for cond, attr in CONDITIONS:
        out += _ambig_points(model, fam, size, cond, attr, ambig)
        out += _disambig_points(model, fam, size, cond, attr, disambig)
    return out


def _ambig_points(model, fam, size, cond, attr, ambig) -> list[CrossModelPoint]:
    """The single ambiguous abstention point for one condition (if any data)."""
    p = _point(model, fam, size, cond, "ambig", ambig, attr)
    return [p] if p is not None else []


def _disambig_points(model, fam, size, cond, attr, disambig) -> list[CrossModelPoint]:
    """Disambiguated points split by gold role (target / other) for one condition."""
    out: list[CrossModelPoint] = []
    for label, gold in DISAMBIG_GOLDS:
        grp = [s for s in disambig if s.gold_label is gold]
        p = _point(model, fam, size, cond, f"disambig-{label}", grp, attr)
        if p is not None:
            out.append(p)
    return out
