"""Reduce a model's SESGO baseline samples to disambiguated accuracy cells.

On CLEAR (disambiguated) items exactly one option is correct, so the direct-answer
readout's ``correct_non_thinking`` is plain accuracy. For figure F2 we collapse
each model into one cell per (bias category x question wording), carrying
successes/total so the plot layer can attach a Wilson 95% CI and annotate n.
Models we cannot size/place (sesgo_model_sizing) yield no cells and are skipped.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.common import BaseSchema
from src.common.math import wilson_err, wilson_interval
from src.datasets.sesgo_eval import SesgoSample
from sesgo.baseline.sesgo_model_sizing import family_of, params_b
from sesgo.common.plain_language_labels import CATEGORY_ORDER, POLARITY_ORDER


@dataclass
class DisambigCell(BaseSchema):
    """Disambiguated direct-answer accuracy for one (model, category, wording)."""

    model: str
    family: str
    params_b: float
    category: str  # Spanish stem (clasismo/racismo/xenofobia/genero)
    polarity: str  # neg / nonneg
    successes: int
    total: int

    @property
    def accuracy(self) -> float:
        """Point estimate; NaN when the slice is empty."""
        return self.successes / self.total if self.total else float("nan")


def cells_for_model(model: str, samples: list[SesgoSample]) -> list[DisambigCell]:
    """All (category x wording) disambiguated accuracy cells for one model.

    Returns ``[]`` for models we cannot size/place so a stray output dir is
    skipped rather than crashing the sweep.
    """
    fam, size = family_of(model), params_b(model)
    if fam is None or size is None:
        return []
    disambig = [s for s in samples if s.context_condition == "disambig"]
    out: list[DisambigCell] = []
    for cat in CATEGORY_ORDER:
        for pol in POLARITY_ORDER:
            grp = [s for s in disambig
                   if s.bias_category == cat and s.question_polarity == pol]
            flags = [s.correct_non_thinking for s in grp]
            if flags:
                out.append(DisambigCell(model, fam, size, cat, pol,
                                        sum(bool(f) for f in flags), len(flags)))
    return out


def pooled_wilson(cells: list[DisambigCell]) -> tuple[float, float, float, int]:
    """Pool cells (e.g. both wordings) into ``(p_hat, below, above, n)``.

    ``below``/``above`` are clamped non-negative Wilson offsets ready for yerr.
    """
    succ = sum(c.successes for c in cells)
    total = sum(c.total for c in cells)
    p, _, _ = wilson_interval(succ, total)
    below, above = wilson_err(succ, total)
    return p, max(0.0, below), max(0.0, above), total


def family_accuracy_series(
    cells: list[DisambigCell], category: str, family: str
) -> tuple[list[float], list[float], list[list[float]]]:
    """One family's accuracy-vs-size line for a category (wordings pooled).

    Returns ``(sizes, accuracies, yerr)`` size-sorted, where ``yerr`` is the
    ``[below, above]`` Wilson offsets list ready to transpose for matplotlib.
    """
    by_size: dict[float, list[DisambigCell]] = defaultdict(list)
    for c in cells:
        if c.category == category and c.family == family:
            by_size[c.params_b].append(c)
    sizes = sorted(by_size)
    stats = [pooled_wilson(by_size[s]) for s in sizes]
    accs = [p for p, _, _, _ in stats]
    yerr = [[b, a] for _, b, a, _ in stats]
    return sizes, accs, yerr


def wording_gap(neg: DisambigCell, non: DisambigCell) -> tuple[float, float]:
    """Signed (negative - neutral) accuracy gap and an independent Wilson band.

    Above 0 means the model is MORE accurate on negatively-worded questions; the
    half-width combines each wording's Wilson error in quadrature.
    """
    p_neg, _, _ = wilson_interval(neg.successes, neg.total)
    p_non, _, _ = wilson_interval(non.successes, non.total)
    en = max(wilson_err(neg.successes, neg.total))
    eo = max(wilson_err(non.successes, non.total))
    return p_neg - p_non, float((en**2 + eo**2) ** 0.5)
