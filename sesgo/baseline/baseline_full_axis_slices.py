"""Slice the full-data baseline samples by the NEW axes (language/origin/scaffold).

The es-original baseline figure slices by bias_category; the full-data baseline_full
study widens the grid to BOTH languages (es/en), BOTH origins (original/BBQ-adapted),
and the scaffold axis (no-scaffold + the three representative debiasing scaffolds).
This module collapses one model's samples into binomial accuracy cells keyed by
(condition, axis, axis_value), where ``axis`` is one of language / origin / scaffold
and the metric is the ambiguous abstention rate — accuracy = fraction predicted
UNKNOWN on the ambiguous items whose gold is UNKNOWN. Abstention is the headline the
scaffolds are meant to move, so each axis cell scores it directly, reusing the same
``AccuracyCell`` (successes / usable-n + Wilson CI) the per-category figure uses.
"""

from __future__ import annotations

from collections import defaultdict

from src.datasets.sesgo import SesgoLabel
from src.datasets.sesgo_eval import SesgoSample

from sesgo.baseline.baseline_accuracy_slices import AccuracyCell, _count
from sesgo.baseline.cross_model_aggregation import CONDITIONS

# The three NEW axes the full-data study adds, each a flat SesgoSample field mapped
# to a human-readable value. Origin reads the bbq provenance flag; scaffold reads the
# scaffold_id (None == the no-scaffold baseline condition).
AXES: tuple[str, ...] = ("language", "origin", "scaffold")


def _axis_value(sample: SesgoSample, axis: str) -> str:
    """Read one sample's value on a NEW axis (language / origin / scaffold)."""
    if axis == "language":
        return sample.language
    if axis == "origin":
        return "BBQ-adapted" if sample.bbq else "original"
    return sample.scaffold_id or "(none)"


def axis_values(samples: list[SesgoSample], axis: str) -> list[str]:
    """Distinct values present on an axis, scaffolds in (none)-first canonical order."""
    present = {_axis_value(s, axis) for s in samples}
    if axis == "scaffold":
        order = ["(none)", "interpretive_direction", "prior_dominance_warning",
                 "intent_and_register_respect", "insufficient_information_unknown"]
        return [v for v in order if v in present] + sorted(present - set(order))
    return sorted(present)


def _ambiguous(samples: list[SesgoSample]) -> list[SesgoSample]:
    """Only the ambiguous items (gold == UNKNOWN); abstention is defined there."""
    return [s for s in samples if s.gold_label is SesgoLabel.UNKNOWN]


def abstention_cells(samples: list[SesgoSample], axis: str) -> list[AccuracyCell]:
    """Per-(condition x axis-value) abstention accuracy on the ambiguous items.

    Cells with zero usable readouts are still emitted (total == 0) so the plot can
    render an honest empty marker rather than silently dropping the bar. The 2-option
    readout has no UNKNOWN, so its abstention is always undefined (n=0) by design.
    """
    ambig = _ambiguous(samples)
    by_value: dict[str, list[SesgoSample]] = defaultdict(list)
    for s in ambig:
        by_value[_axis_value(s, axis)].append(s)
    out: list[AccuracyCell] = []
    for cond, attr in CONDITIONS:
        for value in axis_values(ambig, axis):
            succ, total = _count(by_value[value], attr)
            out.append(AccuracyCell(cond, axis, value, succ, total))
    return out
