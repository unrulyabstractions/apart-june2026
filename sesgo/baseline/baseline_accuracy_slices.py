"""Per-category SESGO accuracy slices for the single-model baseline figure.

Collapses one model's samples into binomial accuracy points keyed by
(condition, bias_category, slice) — the same three readouts and three slices the
cross-model figure uses, but kept per-category so a single model's bias signal
(disambiguated TARGET-gold vs OTHER-gold gap) reads per social-group axis.

Three accuracy SLICES per (condition, category):
  * ``ambig``           : ambiguous abstention accuracy, gold == UNKNOWN.
  * ``disambig-target`` : disambiguated accuracy on items whose gold is TARGET.
  * ``disambig-other``  : disambiguated accuracy on items whose gold is OTHER.
The 2-option readout has no UNKNOWN, so its ``ambig`` slice is always empty (n=0).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.common import BaseSchema
from src.common.math import wilson_interval
from src.datasets.sesgo_eval import SesgoSample

from sesgo.baseline.cross_model_aggregation import CONDITIONS, DISAMBIG_GOLDS

# Stable slice order fixes the within-category grouped-bar order (and legend).
# Each entry: (slice key, context_condition filter, gold-role key or None).
SLICES: tuple[tuple[str, str, str | None], ...] = (
    ("ambig", "ambig", None),
    *(("disambig-" + name, "disambig", name) for name, _ in DISAMBIG_GOLDS),
)
_GOLD_BY_NAME = dict(DISAMBIG_GOLDS)


@dataclass
class AccuracyCell(BaseSchema):
    """One binomial accuracy cell: successes / usable-n for a slice in a category."""

    condition: str
    category: str
    slice_label: str
    successes: int
    total: int

    @property
    def accuracy(self) -> float:
        """Point estimate, or NaN when the slice has no usable readouts."""
        return self.successes / self.total if self.total else float("nan")

    @property
    def wilson(self) -> tuple[float, float, float]:
        """``(p_hat, lo, hi)`` Wilson 95% interval over successes / total."""
        return wilson_interval(self.successes, self.total)


def _filter(samples: list[SesgoSample], context: str, gold_name: str | None):
    """Samples matching a slice's context condition and (optional) gold role."""
    out = [s for s in samples if s.context_condition == context]
    if gold_name is not None:
        out = [s for s in out if s.gold_label is _GOLD_BY_NAME[gold_name]]
    return out


def _count(samples: list[SesgoSample], attr: str) -> tuple[int, int]:
    """Successes and usable n for a correctness attr (None readouts don't count)."""
    flags = [getattr(s, attr) for s in samples]
    usable = [f for f in flags if f is not None]
    return sum(bool(f) for f in usable), len(usable)


def categories_of(samples: list[SesgoSample]) -> list[str]:
    """Sorted distinct bias categories present in the samples."""
    return sorted({s.bias_category for s in samples})


def cells_for(samples: list[SesgoSample]) -> list[AccuracyCell]:
    """Every (condition x category x slice) accuracy cell for one model's samples.

    Cells with zero usable readouts are still emitted (total == 0) so the plot
    can render an honest empty marker rather than silently dropping the bar.
    """
    by_cat: dict[str, list[SesgoSample]] = defaultdict(list)
    for s in samples:
        by_cat[s.bias_category].append(s)
    out: list[AccuracyCell] = []
    for cond, attr in CONDITIONS:
        for cat in sorted(by_cat):
            for slice_label, context, gold_name in SLICES:
                grp = _filter(by_cat[cat], context, gold_name)
                succ, total = _count(grp, attr)
                out.append(AccuracyCell(cond, cat, slice_label, succ, total))
    return out
