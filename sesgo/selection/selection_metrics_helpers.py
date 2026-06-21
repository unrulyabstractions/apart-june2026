"""Aggregate a selection SesgoDataset into pooled abstention/accuracy counts.

The selection study probes ambiguous SESGO items (gold = "unknown", so accuracy IS
abstention) read out at several answering styles. This module reduces the raw samples
into flat, typed counts the plot layer renders directly — never passing a nested
dict/list across a boundary.

A ``ScaffoldCount`` is one cell: how many predictions were "correct" (abstained, on
ambiguous items) out of how many were defined at that readout level. Counts can be
pooled by readout level (``counts_by_level``) or split by bias category
(``counts_by_category``); the 2-option forced choice is undefined on ambiguous items.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.common import BaseSchema
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample
from sesgo.common.plain_language_labels import READOUT_GLOSS, READOUT_LABEL

# The four readout levels and the per-sample correctness accessor for each. The
# 2-option level is only defined on disambiguated items (no "unknown" to abstain to).
LEVELS = ("non_thinking", "non_thinking_2opt", "greedy_thinking", "thinking")


def readout_title(level: str) -> str:
    """Plain-language readout name + one-line gloss, e.g. for a panel badge."""
    return f"{READOUT_LABEL[level]} ({READOUT_GLOSS[level]})"


@dataclass
class ScaffoldCount(BaseSchema):
    """Correct-out-of-defined for one (readout level or bias category) cell."""

    scaffold: str  # the cell's key (readout level or bias-category code)
    correct: int
    total: int  # predictions DEFINED at this level (the denominator)

    @property
    def rate(self) -> float:
        """Abstention / accuracy fraction; 0 when nothing was defined."""
        return self.correct / self.total if self.total else 0.0


def _level_flag(sample: SesgoSample, level: str) -> bool | None:
    """Per-condition correctness at a level, or None when undefined for this sample."""
    if level == "non_thinking":
        return sample.correct_non_thinking if sample.predicted_non_thinking else None
    if level == "thinking":
        return sample.correct_thinking if sample.predicted_thinking else None
    if level == "greedy_thinking":
        return (
            sample.correct_greedy_thinking
            if sample.predicted_greedy_thinking
            else None
        )
    if level == "non_thinking_2opt":
        return sample.correct_2opt  # None for ambiguous items by construction
    return None


def _count(flags: list[bool]) -> tuple[int, int]:
    """(#correct, #defined) over a list of correctness flags."""
    return sum(flags), len(flags)


def count_for_level(dataset: SesgoDataset, level: str, context: str) -> ScaffoldCount:
    """One pooled correct-out-of-defined cell at a readout level (all scaffolds)."""
    flags = [
        f for s in dataset.samples if s.context_condition == context
        for f in [_level_flag(s, level)] if f is not None
    ]
    correct, total = _count(flags)
    return ScaffoldCount(scaffold=level, correct=correct, total=total)


def counts_by_level(
    dataset: SesgoDataset, levels: list[str], context: str
) -> dict[str, ScaffoldCount]:
    """Pooled ScaffoldCount per readout level, keeping only levels with data."""
    out = {lvl: count_for_level(dataset, lvl, context) for lvl in levels}
    return {lvl: c for lvl, c in out.items() if c.total > 0}


def counts_by_category(
    dataset: SesgoDataset, level: str, context: str, categories: list[str]
) -> dict[str, ScaffoldCount]:
    """Pooled ScaffoldCount per bias category at a fixed readout level/context."""
    flags: dict[str, list[bool]] = defaultdict(list)
    for s in dataset.samples:
        if s.context_condition != context or s.bias_category not in categories:
            continue
        flag = _level_flag(s, level)
        if flag is not None:
            flags[s.bias_category].append(flag)
    return {
        cat: ScaffoldCount(scaffold=cat, correct=_count(fl)[0], total=_count(fl)[1])
        for cat, fl in flags.items()
    }
