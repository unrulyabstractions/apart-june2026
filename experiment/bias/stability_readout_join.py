"""Join a greedy-readout slice (`out/<study>/<dir>/response_samples.json`, which carries
only `prompt_id` + `choice` + measurements) back to the prompt dataset metadata it needs
for the bias / accuracy figures — gold role, question polarity, context condition — keyed
by the stable `prompt_id`. Keeping the readout minimal and re-joining here means the
running fleet never has to be re-run to add provenance fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.common import BaseSchema


@dataclass
class EnrichedResponse(BaseSchema):
    """One model response with the provenance needed to score it."""

    group_key: str       # bare model dir name (the figure's per-model series)
    prompt_id: str
    question_id: str
    role: str            # the committed choice: target / other / unknown / invalid
    gold: str            # gold role: unknown (ambig) | target | other (disambig)
    polarity: str        # neg | nonneg
    context: str         # ambig | disambig
    bias_category: str
    label_prob: float
    vocab_diversity: float

    @property
    def correct(self) -> bool:
        return self.role == self.gold


def load_metadata(dataset_path: Path) -> dict[str, dict]:
    """{prompt_id: record} for the prompt dataset (gold/polarity/context/...)."""
    return {r["prompt_id"]: r for r in json.load(dataset_path.open())}


def enrich(samples: list[dict], group_key: str, meta: dict[str, dict]) -> list[EnrichedResponse]:
    """Join readout samples to dataset metadata by prompt_id (drops unmatched)."""
    out: list[EnrichedResponse] = []
    for s in samples:
        m = meta.get(s["prompt_id"])
        if m is None:
            continue
        out.append(EnrichedResponse(
            group_key=group_key, prompt_id=s["prompt_id"], question_id=m["question_id"],
            role=s["choice"], gold=m["gold_label"], polarity=m["question_polarity"],
            context=m["context_condition"], bias_category=m["bias_category"],
            label_prob=s["label_prob"], vocab_diversity=s["vocab_diversity"],
        ))
    return out
