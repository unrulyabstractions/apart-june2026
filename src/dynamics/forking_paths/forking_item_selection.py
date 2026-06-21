"""Pick the ambiguous SESGO item whose committed outcome is most likely to FLIP.

A forking token only exists if re-sampling can divert the final outcome, so the
best demo item is one whose sampled thinking decodes DISAGREE on the answer (high
outcome entropy). This module pilots a handful of sampled thinking decodes per
candidate item, parses each to an outcome, and scores the item by the Shannon
entropy of its empirical outcome distribution — the driver then keeps the top
item(s). Pure logic (no I/O); the driver owns the model + persistence.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.math import probs_to_logprobs, shannon_entropy
from src.datasets.prompt import SesgoPromptSample
from src.inference import ModelRunner

from .forking_outcome_set import ForkOutcomeSet
from .forking_outcome_mapping import rollout_to_outcome_label
from .outcome_histogram_builder import conditional_histogram


@dataclass
class ItemEntropy:
    """One candidate item's pilot outcome distribution + its entropy score."""

    sample_idx: int
    question_id: str
    histogram: list[float]  # pilot outcome distribution over the outcome set
    entropy: float  # Shannon entropy (nats) of the pilot distribution
    n_parsed: int  # number of pilot draws that parsed to a role


def pilot_item_entropy(
    runner: ModelRunner,
    sample: SesgoPromptSample,
    outcome_set: ForkOutcomeSet,
    n_pilot: int,
    max_new_tokens: int,
    temperature: float,
) -> ItemEntropy:
    """Pilot ``n_pilot`` sampled thinking decodes and score the outcome entropy."""
    rollouts = runner.generate_batch(
        [sample.text] * n_pilot, max_new_tokens=max_new_tokens, temperature=temperature
    )
    labels = [rollout_to_outcome_label(r, sample) for r in rollouts]
    hist = conditional_histogram(labels, outcome_set)
    return ItemEntropy(
        sample_idx=sample.sample_idx,
        question_id=sample.question_id,
        histogram=hist,
        entropy=float(shannon_entropy(probs_to_logprobs(hist))),
        n_parsed=len(labels),
    )


def rank_items_by_entropy(items: list[ItemEntropy]) -> list[ItemEntropy]:
    """Highest-outcome-entropy items first (the most likely to host a fork)."""
    return sorted(items, key=lambda it: it.entropy, reverse=True)
