"""Drive the steered SESGO abstention readout across a steering-strength sweep.

This is the thin intervention runner the task asks for: it wraps the model with
the existing add-mode resid_post hook (built once per alpha via ``steering(...)``)
and exposes the SESGO abstention readout under it. It delegates ALL hook
machinery to the inference stack — ``steering()`` builds the Intervention,
``SteeredTernaryChoiceRunner`` routes ``choose3`` through
``compute_trajectories_batch_with_intervention``. No new hooks are written.

``measure_abstention`` runs one alpha over a list of held-out readouts and
aggregates UNKNOWN mass; ``unsteered_reference`` scores the actual-scaffold
prompts with NO intervention (the behaviour +v aims to reproduce).
"""

from __future__ import annotations

import numpy as np

from src.inference.interventions import steering
from .sesgo_abstention_readout import (
    AbstentionReadout,
    is_abstained,
    unknown_probability,
)
from .steered_ternary_runner import SteeredTernaryChoiceRunner
from .steering_test_schema import ScaffoldReference, SweepPoint


def _score_items(
    runner: SteeredTernaryChoiceRunner, readouts: list[AbstentionReadout]
) -> tuple[float, float]:
    """Mean UNKNOWN probability and hard abstain rate over the readouts."""
    probs, abstains = [], []
    for r in readouts:
        choice = runner.choose3(r.prompt, r.choice_prefix, r.labels)
        probs.append(unknown_probability(choice))
        abstains.append(is_abstained(choice))
    n = max(1, len(readouts))
    return float(np.mean(probs)) if probs else 0.0, sum(abstains) / n


def measure_abstention(
    runner: SteeredTernaryChoiceRunner,
    readouts: list[AbstentionReadout],
    layer: int,
    direction: list[float],
    alpha: float,
    normalize: bool,
    baseline_unknown_prob: float | None = None,
) -> SweepPoint:
    """Abstention at one steering strength over the held-out readouts.

    alpha == 0 means NO intervention (the unsteered baseline); any other alpha
    installs ``alpha * v`` on resid_post at ``layer`` for the whole forward.
    """
    runner.active_intervention = (
        None
        if alpha == 0.0
        else steering(
            layer=layer,
            direction=direction,
            strength=alpha,
            component="resid_post",
            normalize=normalize,
        )
    )
    mean_unknown, abstain_rate = _score_items(runner, readouts)
    runner.active_intervention = None
    base = baseline_unknown_prob if baseline_unknown_prob is not None else mean_unknown
    return SweepPoint(
        alpha=alpha,
        n_items=len(readouts),
        mean_unknown_prob=mean_unknown,
        abstain_rate=abstain_rate,
        delta_unknown_prob=mean_unknown - base,
    )


def run_alpha_sweep(
    runner: SteeredTernaryChoiceRunner,
    readouts: list[AbstentionReadout],
    layer: int,
    direction: list[float],
    alphas: list[float],
    normalize: bool,
    log_fn=None,
) -> list[SweepPoint]:
    """Sweep every alpha, anchoring each point's delta on the alpha=0 baseline.

    alpha=0 is scored FIRST so every steered point's ``delta_unknown_prob`` is
    measured against the true unsteered baseline. Returns the points sorted by
    alpha. ``log_fn`` (optional) is called with one line per evaluated point.
    """
    ordered = sorted(alphas, key=lambda a: (a != 0.0, a))
    baseline: float | None = None
    sweep: list[SweepPoint] = []
    for alpha in ordered:
        point = measure_abstention(
            runner, readouts, layer, direction, alpha, normalize, baseline
        )
        if alpha == 0.0:
            baseline = point.mean_unknown_prob
        sweep.append(point)
        if log_fn is not None:
            log_fn(
                f"[test] alpha={alpha:+.2f} unknown_prob={point.mean_unknown_prob:.3f} "
                f"abstain_rate={point.abstain_rate:.3f} "
                f"d_prob={point.delta_unknown_prob:+.3f}"
            )
    return sorted(sweep, key=lambda p: p.alpha)


def unsteered_reference(
    runner: SteeredTernaryChoiceRunner, readouts: list[AbstentionReadout]
) -> ScaffoldReference:
    """Unsteered abstention on the actual-scaffold prompts (the target behaviour)."""
    runner.active_intervention = None
    mean_unknown, abstain_rate = _score_items(runner, readouts)
    return ScaffoldReference(
        n_items=len(readouts),
        mean_unknown_prob=mean_unknown,
        abstain_rate=abstain_rate,
    )
