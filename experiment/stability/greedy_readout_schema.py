"""Minimal per-prompt readout from a single greedy generation.

One record per prompt (mapped back to the prompt dataset via `prompt_id`). The
model always greedy-decodes a full trajectory; we parse its answer into a label
(the option marker) + choice (target/other/unknown/invalid), and measure the
model's commitment at the answer position: `label_prob` (probability of the chosen
label token given the preceding context = exp of its logprob) and `vocab_diversity`
(effective number of next-token choices there = Hill D1 = exp of Shannon entropy).
A thinking model and a non-thinking model are SEPARATE runs (separate files) of the
same prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema


@dataclass
class GreedyReadout(BaseSchema):
    sample_idx: int
    prompt_id: str
    prompt_text: str
    response_text: str
    choice: str          # target / other / unknown / invalid
    label: str           # the option marker picked, e.g. "c)" ("" if none parsed)
    label_prob: float       # probability the model gave the chosen label token | context
    vocab_diversity: float  # effective number of next-token choices (Hill D1 = exp(entropy))
    degenerate: bool = False  # response was a repetition loop / garbage (e.g. a backend that
                              # silently mis-generates) — excluded from analysis, never silent


@dataclass
class GreedyReadoutDataset(BaseSchema):
    """All readouts for one (model, mode, study) run — what response_samples.json holds."""

    study: str
    model: str
    mode: str            # "thinking" or "nonthinking"
    samples: list[GreedyReadout] = field(default_factory=list)
