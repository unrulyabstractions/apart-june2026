"""Knobs for how each SESGO prompt is queried at both levels."""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema


@dataclass
class SesgoQueryConfig(BaseSchema):
    """Per-run settings shared across every prompt in a dataset."""

    n_thinking_samples: int = 8  # free-form draws per prompt for the thinking level
    temperature: float = 0.7  # >0 so thinking draws actually vary
    # Reasoning models (e.g. Qwen3) spend most of the budget inside <think>;
    # too small a budget truncates before the answer and the draw is dropped.
    max_new_tokens: int = 512
    do_non_thinking: bool = True  # teacher-forced 3-way softmax over positions
    # The 2-option forced choice (target vs other, NO unknown) is a SECOND
    # teacher-forced readout reusing the same loaded weights — disable it to keep
    # a run to the 3-option readout only.
    do_two_option: bool = True
    # The greedy non-thinking decode is an EXTRA generation on top of the
    # teacher-forced readout — disable it for cheap label-only runs.
    do_greedy: bool = True
    # The greedy-THINKING decode is a SINGLE deterministic (temperature 0)
    # generation WITH reasoning enabled (NO skip-thinking prefix), parsed for the
    # post-</think> answer — the role the model commits to when it reasons
    # greedily. Distinct from the greedy non-thinking decode and the sampled draws.
    do_greedy_thinking: bool = False
    do_thinking: bool = True  # sampled free-form reasoning, parsed per draw
    subsample: float = 1.0  # fraction of prompts to query (1.0 == all)
    # Prompts processed per batched forward pass. 1 == the exact single-sample
    # path (safe fallback); >1 batches choose3 / greedy / thinking draws across
    # prompts for higher throughput. Identical results within fp tolerance.
    batch_size: int = 1
