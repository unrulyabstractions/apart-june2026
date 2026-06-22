"""Compute the forking-paths prior o_0 by full-resampling the bare prompt.

o_0 is the N-sample full-resample baseline outcome distribution — the drift
reference frame every later o_t is measured against. Factored into its own module
(out of ``forking_path_capture``) so BOTH the single-box capture driver AND the
sharded fleet's shard-0 box import the SAME implementation: in the sharded flow
only shard 0 pays this N-draw cost, exactly mirroring the single-box prior.
"""

from __future__ import annotations

from src.datasets.prompt import SesgoPromptSample
from src.inference import ModelRunner

from .forking_outcome_set import ForkOutcomeSet
from .forking_outcome_mapping import rollout_to_outcome_label
from .outcome_histogram_builder import conditional_histogram


def resample_prior(
    runner: ModelRunner,
    sample: SesgoPromptSample,
    outcome_set: ForkOutcomeSet,
    n_prior: int,
    max_new_tokens: int,
    temperature: float,
) -> list[float]:
    """o_0: full-resample prior from the bare prompt (N independent draws)."""
    rollouts = runner.generate_batch(
        [sample.text] * n_prior, max_new_tokens=max_new_tokens, temperature=temperature
    )
    labels = [rollout_to_outcome_label(r, sample) for r in rollouts]
    return conditional_histogram(labels, outcome_set)
