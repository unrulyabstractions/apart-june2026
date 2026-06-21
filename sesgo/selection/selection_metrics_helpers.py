"""Aggregate a selection SesgoDataset into per-scaffold accuracy counts.

The selection study crosses one SESGO item against five scaffold conditions (the
no-scaffold baseline plus four debiasing preambles) at four readout levels and two
context conditions. This module reduces the raw samples into flat, typed counts the
plot layer can render directly — never passing a nested dict/list across a boundary.

A ``ScaffoldCount`` is one (scaffold, level, context, [category]) cell: how many
predictions were correct out of how many were defined at that level. Correctness is
per-condition (ambiguous gold = UNKNOWN i.e. abstention; disambiguated gold = the
ground-truth role); the 2-option forced choice is undefined for ambiguous items.
``rank_scaffolds`` orders scaffolds best→worst on a chosen level/condition and names
the SELECTED one, mirroring the visualizer's original SELECT semantics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.common import BaseSchema
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample

# The no-scaffold condition has scaffold_id == None; label it so it reads clearly.
BASELINE = "(baseline)"
# The four readout levels and the per-sample correctness accessor for each. The
# 2-option level is only defined on disambiguated items (no UNKNOWN to abstain to).
LEVELS = ("non_thinking", "non_thinking_2opt", "greedy_thinking", "thinking")
LEVEL_TITLE = {
    "non_thinking": "non-thinking (3-option)",
    "non_thinking_2opt": "forced-choice (2-option)",
    "greedy_thinking": "greedy-thinking (baseline)",
    "thinking": "thinking (sampled)",
}


@dataclass
class ScaffoldCount(BaseSchema):
    """Correct-out-of-defined for one (scaffold, level, context[, category]) cell."""

    scaffold: str
    correct: int
    total: int  # predictions DEFINED at this level (the denominator)

    @property
    def rate(self) -> float:
        """Accuracy / abstention fraction; 0 when nothing was defined."""
        return self.correct / self.total if self.total else 0.0


@dataclass
class ScaffoldRanking(BaseSchema):
    """Scaffolds ordered best→worst on the SELECT level, plus the chosen one."""

    order: list[str] = field(default_factory=list)
    selected: str | None = None


def scaffold_label(scaffold_id: str | None) -> str:
    """Human label for a scaffold condition; None == the no-scaffold baseline."""
    return scaffold_id or BASELINE


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


def counts_by_scaffold(
    dataset: SesgoDataset, level: str, context: str
) -> dict[str, ScaffoldCount]:
    """Per-scaffold ScaffoldCount at one readout level and context condition."""
    flags: dict[str, list[bool]] = defaultdict(list)
    for s in dataset.samples:
        if s.context_condition != context:
            continue
        flag = _level_flag(s, level)
        if flag is not None:
            flags[scaffold_label(s.scaffold_id)].append(flag)
    out: dict[str, ScaffoldCount] = {}
    for sc, fl in flags.items():
        c, t = _count(fl)
        out[sc] = ScaffoldCount(scaffold=sc, correct=c, total=t)
    return out


def counts_by_scaffold_category(
    dataset: SesgoDataset, level: str, context: str, category: str
) -> dict[str, ScaffoldCount]:
    """Per-scaffold ScaffoldCount for one bias_category at a level and context."""
    flags: dict[str, list[bool]] = defaultdict(list)
    for s in dataset.samples:
        if s.context_condition != context or s.bias_category != category:
            continue
        flag = _level_flag(s, level)
        if flag is not None:
            flags[scaffold_label(s.scaffold_id)].append(flag)
    return {
        sc: ScaffoldCount(scaffold=sc, correct=_count(fl)[0], total=_count(fl)[1])
        for sc, fl in flags.items()
    }


def all_scaffolds(dataset: SesgoDataset) -> list[str]:
    """Every scaffold label present, baseline first then the rest sorted."""
    labels = {scaffold_label(s.scaffold_id) for s in dataset.samples}
    rest = sorted(labels - {BASELINE})
    return ([BASELINE] if BASELINE in labels else []) + rest


def rank_scaffolds(
    counts: dict[str, ScaffoldCount], scaffolds: list[str]
) -> ScaffoldRanking:
    """Order scaffolds best→worst by rate (ties: more data, then non-baseline).

    Scaffolds with no defined predictions sink to the bottom and are never SELECTED.
    The SELECTED scaffold is the top one that actually has data, matching the
    original visualizer's "skip a scaffold with no decodable answer" rule.
    """

    def key(sc: str) -> tuple[float, int, int]:
        c = counts.get(sc)
        if c is None or c.total == 0:
            return (float("-inf"), 0, 0)
        return (c.rate, c.total, 0 if sc == BASELINE else 1)

    order = sorted(scaffolds, key=key, reverse=True)
    selectable = [sc for sc in order if counts.get(sc) and counts[sc].total > 0]
    return ScaffoldRanking(order=order, selected=selectable[0] if selectable else None)
