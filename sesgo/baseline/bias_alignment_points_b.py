"""Reduce SESGO samples to one bias-alignment-vs-accuracy POINT per group.

The redesigned figure plots every group (model or scaffold) as a single MARKER at
(x = bias alignment, y = accuracy) per panel -- not a bar. This module computes
that point from one group's teacher-forced non-thinking readout (prob ordered
[target, other, unknown]):

  * BIAS ALIGNMENT x in [-1, 1]: mean over items of the RAW signed displacement
        P(target) - P(other)
    0 == unbiased, +1 == fully toward the stereotyped TARGET, -1 == toward OTHER.
    Raw (not the target/other-renormalised ratio) so the achievable region of
    (x, accuracy) is a clean triangle: |x| <= 1 - accuracy on the ambiguous panel.
  * ACCURACY y: abstention rate on AMBIGUOUS items (gold UNKNOWN) for the ambiguous
    panel; correctness on DISAMBIGUATED items for the disambiguated one. Carried as
    successes / total so the plot can attach a Wilson 95% CI and report n.

One ``BiasPoint`` per (group, panel). Groups with no usable readout are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.common.math import wilson_interval
from src.datasets.sesgo import SesgoLabel
from src.datasets.sesgo_eval import SesgoSample


def _signed_displacement(samples: list[SesgoSample]) -> float | None:
    """Mean raw (P_target - P_other) over readable items; None if none readable."""
    deltas = [
        s.non_thinking.prob[0] - s.non_thinking.prob[1]
        for s in samples
        if s.non_thinking is not None
    ]
    return sum(deltas) / len(deltas) if deltas else None


def _accuracy_count(samples: list[SesgoSample], panel: str) -> tuple[int, int]:
    """Successes / usable-n for the panel's accuracy (abstention vs correctness)."""
    usable = [s for s in samples if s.non_thinking is not None]
    if panel == "ambig":
        succ = sum(s.predicted_non_thinking is SesgoLabel.UNKNOWN for s in usable)
    else:
        succ = sum(s.correct_non_thinking for s in usable)
    return succ, len(usable)


@dataclass
class BiasPoint(BaseSchema):
    """One group's (bias alignment, accuracy) point for one panel."""

    group_key: str  # bare model name (baseline) or scaffold id ("None" == baseline)
    panel: str  # "ambig" or "disambig"
    bias: float  # mean raw signed displacement P_target - P_other, in [-1, 1]
    successes: int  # accuracy numerator (abstentions or correct judgements)
    total: int  # accuracy denominator (usable n)

    @property
    def accuracy(self) -> float:
        """Point accuracy (marker height); NaN with no data."""
        return self.successes / self.total if self.total else float("nan")

    @property
    def wilson(self) -> tuple[float, float, float]:
        """``(p_hat, lo, hi)`` Wilson 95% interval over successes/total."""
        return wilson_interval(self.successes, self.total)


def _point(group_key: str, panel: str, samples: list[SesgoSample]) -> BiasPoint | None:
    """Build one panel's point for a group; None if nothing is readable."""
    bias = _signed_displacement(samples)
    succ, total = _accuracy_count(samples, panel)
    if bias is None or total == 0:
        return None
    return BiasPoint(group_key=group_key, panel=panel, bias=bias,
                     successes=succ, total=total)


def points_for_group(group_key: str, samples: list[SesgoSample]) -> list[BiasPoint]:
    """Both panel points (ambiguous, disambiguated) for one group's samples."""
    ambig = [s for s in samples if s.context_condition == "ambig"]
    disambig = [s for s in samples if s.context_condition == "disambig"]
    out: list[BiasPoint] = []
    for panel, subset in (("ambig", ambig), ("disambig", disambig)):
        pt = _point(group_key, panel, subset)
        if pt is not None:
            out.append(pt)
    return out
