"""Stability metrics over a collected SESGO dataset, split by context condition.

A STABILITY group is one semantic item under one polarity: a (question_id,
context_condition, question_polarity) triple. Within a group the gold answer is
FIXED, so every member differs only in superficial FORMAT (label style x role
permutation). Any change in the prediction across a group is therefore pure format
sensitivity — the quantity these helpers measure.

Each metric is returned as a small ``BaseSchema`` carrying its point estimate, the
counts needed for a Wilson / SEM / bootstrap interval, and the sample size, so the
plotting layer never re-derives n. Two readouts are tracked in parallel: the
3-OPTION non-thinking argmax (``predicted_non_thinking``) and the 2-OPTION
forced-choice pick (``picked_2opt``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from src.common import BaseSchema
from src.datasets.sesgo import SesgoLabel
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample
from stability_permutation_grid import (
    assert_permutation_grid,
    permutation_signature,
)

# Non-thinking prob vector is ordered [TARGET, OTHER, UNKNOWN]; index 2 is p_unknown.
_P_UNKNOWN_IDX = 2
# The two superficial axes the stability grid varies (everything else is fixed).
AXES = ("label_style", "permutation")
CONDITIONS = ("ambig", "disambig")


@dataclass
class ConsistencySet(BaseSchema):
    """Per-item consistency / entropy values for one readout under one condition."""

    consistency: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)


@dataclass
class FlipRate(BaseSchema):
    """A mean flip rate (1 - within-group consistency) along one format axis."""

    axis: str
    rate: float
    n_groups: int  # number of (held-fixed) sub-groups averaged
    flips: list[float] = field(default_factory=list)  # per sub-group, for bootstrap


@dataclass
class AccuracyCount(BaseSchema):
    """A correct/total tally (for a Wilson interval) under one label."""

    label: str
    correct: int
    total: int


def _group_key(s: SesgoSample) -> tuple[str, str, str]:
    """The (question_id, context_condition, polarity) a sample belongs to."""
    return (s.question_id, s.context_condition, s.question_polarity)


def _axis_key(s: SesgoSample, axis: str) -> str:
    """The value of one format axis (label_style marker string or permutation)."""
    return s.label_style if axis == "label_style" else permutation_signature(s)


def _modal_fraction(labels: list[SesgoLabel]) -> float:
    """Fraction of labels equal to the modal label (1.0 for <=1 label)."""
    if not labels:
        return float("nan")
    return Counter(l.value for l in labels).most_common(1)[0][1] / len(labels)


def _entropy(labels: list[SesgoLabel]) -> float:
    """Shannon entropy (nats) of the label distribution."""
    if not labels:
        return float("nan")
    c = np.array(list(Counter(l.value for l in labels).values()), dtype=float)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def _labels_by_group(
    dataset: SesgoDataset, condition: str, two_opt: bool
) -> dict[tuple[str, str, str], list[SesgoLabel]]:
    """Map each stability group (in `condition`) to its readout labels."""
    out: dict[tuple[str, str, str], list[SesgoLabel]] = defaultdict(list)
    for s in dataset.samples:
        if s.context_condition != condition:
            continue
        label = s.picked_2opt if two_opt else s.predicted_non_thinking
        if label is not None:
            out[_group_key(s)].append(label)
    return out


def consistency_set(
    dataset: SesgoDataset, condition: str, two_opt: bool
) -> ConsistencySet:
    """Per-group consistency + entropy for one readout under one condition."""
    groups = _labels_by_group(dataset, condition, two_opt)
    cons, ents = [], []
    for labels in groups.values():
        if len(labels) < 2:
            continue
        cons.append(_modal_fraction(labels))
        ents.append(_entropy(labels))
    return ConsistencySet(consistency=cons, entropy=ents)


def flip_rate(dataset: SesgoDataset, axis: str, condition: str) -> FlipRate:
    """Mean per-item flip rate ALONG `axis`, other axis held fixed, in `condition`.

    Within each stability group we further bucket by the OTHER axis (so only `axis`
    varies inside a bucket) and take 1 - modal_fraction. Averaging over all such
    buckets gives how much perturbing `axis` alone moves the 3-option prediction.
    """
    assert_permutation_grid(dataset)  # fail loud on a drifted / incomplete grid
    other = AXES[1 - AXES.index(axis)]
    buckets: dict[tuple, list[SesgoLabel]] = defaultdict(list)
    for s in dataset.samples:
        if s.context_condition != condition or s.predicted_non_thinking is None:
            continue
        buckets[(*_group_key(s), _axis_key(s, other))].append(s.predicted_non_thinking)
    flips = [1.0 - _modal_fraction(v) for v in buckets.values() if len(v) >= 2]
    rate = float(np.mean(flips)) if flips else float("nan")
    return FlipRate(axis=axis, rate=rate, n_groups=len(flips), flips=flips)


def p_unknown_spread(dataset: SesgoDataset, condition: str) -> list[float]:
    """Per-group std of non-thinking p(unknown) across variations (>=2 present)."""
    by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for s in dataset.samples:
        if s.context_condition == condition and s.non_thinking is not None:
            by_group[_group_key(s)].append(float(s.non_thinking.prob[_P_UNKNOWN_IDX]))
    return [float(np.std(v)) for v in by_group.values() if len(v) >= 2]


def accuracy_count(dataset: SesgoDataset, condition: str, two_opt: bool) -> AccuracyCount:
    """Correct/total tally for one readout under one condition (Wilson-ready).

    3-option uses ``correct_non_thinking`` (per-condition gold). 2-option uses
    ``correct_2opt`` which is None for ambiguous items, so the ambiguous 2-option
    tally is empty (total 0) — the plot renders it as N/A.
    """
    correct = total = 0
    for s in dataset.samples:
        if s.context_condition != condition:
            continue
        if two_opt:
            ok = s.correct_2opt
            if ok is None:
                continue
        else:
            if s.predicted_non_thinking is None:
                continue
            ok = s.correct_non_thinking
        total += 1
        correct += int(ok)
    label = ("2-opt " if two_opt else "3-opt ") + condition
    return AccuracyCount(label=label, correct=correct, total=total)
