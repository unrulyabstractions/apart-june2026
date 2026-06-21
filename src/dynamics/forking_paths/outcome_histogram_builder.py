"""Build the prob-weighted outcome histograms o_{t,w} and o_t (Eqs. 1-2).

These are pure aggregations over already-parsed rollout outcome labels. A rollout
is mapped to a one-hot R-vector (via ForkOutcomeSet); o_{t,w} is the mean R over
a token's rollouts (Eq. 1, sampled continuations weighted uniformly since they
are drawn i.i.d. from the same conditional), and o_t averages the o_{t,w} over
the alternate tokens w weighted by their next-token probability p(x_t=w) (Eq. 2).
Both are renormalized to sum to 1 over the outcome categories.
"""

from __future__ import annotations

from src.common.math import normalize

from .forking_outcome_set import ForkOutcomeSet


def conditional_histogram(
    rollout_labels: list[str], outcome_set: ForkOutcomeSet
) -> list[float]:
    """o_{t,w}: mean one-hot outcome over a token's rollouts (Eq. 1).

    Empty rollouts -> a zero vector (the caller drops such alternates). The i.i.d.
    sampled continuations share the same path weight, so the expectation is a plain
    average of their one-hot outcomes.
    """
    if not rollout_labels:
        return [0.0] * outcome_set.dim
    acc = [0.0] * outcome_set.dim
    for label in rollout_labels:
        oh = outcome_set.one_hot(label)
        for i in range(outcome_set.dim):
            acc[i] += oh[i]
    return [c / len(rollout_labels) for c in acc]


def position_histogram(
    conditional_histograms: list[list[float]],
    token_probs: list[float],
    outcome_set: ForkOutcomeSet,
) -> list[float]:
    """o_t: the o_{t,w} averaged over alternates w, weighted by p(x_t=w) (Eq. 2).

    Weights are renormalized over the PRESENT alternates (the top-k >= floor set),
    and the result is renormalized to a valid distribution over outcomes. Empty
    input -> a uniform fallback so downstream L2 distances stay well-defined.
    """
    if not conditional_histograms:
        return normalize([1.0] * outcome_set.dim)
    total_w = sum(token_probs) or 1.0
    acc = [0.0] * outcome_set.dim
    for hist, w in zip(conditional_histograms, token_probs):
        weight = w / total_w
        for i in range(outcome_set.dim):
            acc[i] += weight * hist[i]
    return normalize(acc)
