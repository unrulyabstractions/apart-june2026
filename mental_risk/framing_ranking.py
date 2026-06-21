"""Rank and SELECT the framing that best tracks the gold risk.

The selection study's core: given the per-framing predicted risks and the gold
risk, score each framing by how well it tracks gold and pick the winner. The
SESGO analogue ranks scaffolds by abstention accuracy; here we rank framings by
their gold-correlation (primary) and mean absolute error (tiebreak), since the
risk gold is continuous. Kept in the library so the visualizer just renders the
ranking it returns. Returns a flat list of typed FramingScore so nothing nested
crosses the function boundary (house rule 7).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from mental_risk.risk_prediction import effective_risk
from src.common import BaseSchema
from src.datasets.risk import RiskAssessmentSample


@dataclass
class FramingScore(BaseSchema):
    """How well one framing's predicted risk tracks the gold risk."""

    framing: str
    n: int  # paired (pred, gold) count
    pearson: float | None  # correlation with gold (primary rank key)
    mae: float | None  # mean absolute error from gold (tiebreak)
    mean_risk: float  # mean predicted risk under this framing


def _pearson(preds: list[float], golds: list[float]) -> float | None:
    """Pearson r between paired predictions and golds, or None if degenerate."""
    if len(preds) < 2 or np.std(preds) == 0 or np.std(golds) == 0:
        return None
    return float(np.corrcoef(preds, golds)[0, 1])


def _score_one(framing: str, pairs: list[tuple[float, float]]) -> FramingScore:
    """Collapse one framing's (pred, gold) pairs into a FramingScore."""
    preds = [p for p, _ in pairs]
    golds = [g for _, g in pairs]
    mae = float(np.mean([abs(p - g) for p, g in pairs])) if pairs else None
    mean_risk = float(np.mean(preds)) if preds else 0.0
    return FramingScore(
        framing=framing, n=len(pairs), pearson=_pearson(preds, golds),
        mae=mae, mean_risk=mean_risk,
    )


def score_framings(samples: list[RiskAssessmentSample]) -> list[FramingScore]:
    """Score every framing by gold-tracking; best (highest r, then lowest MAE) first."""
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in samples:
        r = effective_risk(s)
        if r is not None and s.gold_risk is not None:
            pairs[s.framing].append((r, s.gold_risk))
    scores = [_score_one(f, p) for f, p in pairs.items()]
    # Rank: higher correlation first (None last), then lower MAE.
    scores.sort(
        key=lambda fs: (
            -(fs.pearson if fs.pearson is not None else -2.0),
            fs.mae if fs.mae is not None else 2.0,
        )
    )
    return scores


def best_framing(scores: list[FramingScore]) -> str | None:
    """The SELECTED framing: the first of the ranked list, or None if empty."""
    return scores[0].framing if scores else None
