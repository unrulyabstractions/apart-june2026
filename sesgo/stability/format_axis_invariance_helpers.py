"""Split format-invariance by the TYPE of format change: label style vs role order.

The 18 superficial variants of a stability item are 3 LABEL styles (a)b)c) /
1)2)3) / x)y)z)) crossed with 6 ROLE orders (the permutation the three groups are
listed in). This module reuses the existing per-bucket flip logic
(``flip_rate``) to ask, separately for each axis: holding the OTHER axis fixed,
how often is the 3-option answer fully unchanged as ONLY this axis varies?

Each ``flip_rate`` bucket fixes the other axis, so a bucket with flip == 0 is an
item that is fully invariant along the queried axis. Pooling those binomial
outcomes over both context conditions yields an invariant/total tally that
carries an honest Wilson interval.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.datasets.sesgo_eval import SesgoDataset
from stability_metrics_helpers import AXES, CONDITIONS, flip_rate

# Plain-English name + display order for the two format axes we contrast.
AXIS_LABEL: dict[str, str] = {"label_style": "Label style", "permutation": "Role order"}
AXIS_ORDER = ("label_style", "permutation")


@dataclass
class AxisInvariance(BaseSchema):
    """Invariant/total bucket tally for one (model, format axis), Wilson-ready."""

    model: str
    axis: str
    invariant: int  # buckets whose answer never moved as only this axis varied
    total: int  # buckets with >=2 comparable variants along this axis

    @property
    def rate(self) -> float:
        """Share of buckets fully invariant along this axis (0 when none)."""
        return self.invariant / self.total if self.total else 0.0


def axis_invariance(dataset: SesgoDataset, model: str, axis: str) -> AxisInvariance:
    """Pool both conditions: count buckets fully invariant along `axis`.

    Reuses ``flip_rate`` (other axis held fixed); a per-bucket flip of 0 means the
    answer never changed as only `axis` was perturbed.
    """
    invariant = total = 0
    for cond in CONDITIONS:
        flips = flip_rate(dataset, axis, cond).flips
        invariant += sum(1 for f in flips if f == 0.0)
        total += len(flips)
    return AxisInvariance(model=model, axis=axis, invariant=invariant, total=total)


def axis_invariances(dataset: SesgoDataset, model: str) -> list[AxisInvariance]:
    """Both format axes for one model, in display order."""
    return [axis_invariance(dataset, model, axis) for axis in AXIS_ORDER]
