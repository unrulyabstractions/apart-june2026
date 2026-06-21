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
    do_thinking: bool = True  # sampled free-form reasoning, parsed per draw
    subsample: float = 1.0  # fraction of prompts to query (1.0 == all)
