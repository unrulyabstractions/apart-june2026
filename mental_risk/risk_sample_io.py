"""Subsample-aware loading of a RiskPromptDataset for the run-by-path drivers.

SESGO repeats an identical ``load_prompt_dataset`` stride helper in every
collector (stability/geometry/...). We extract the risk version once so the five
mental_risk collectors share it instead of copy-pasting: when subsample < 1 we
json-load the raw dict ONCE, take an evenly-spaced stride over the raw sample
dicts (so the slice still spans every framing/format block, not just the first
subject), and build only the kept RiskPromptSamples. The querier then runs with
subsample=1.0.
"""

from __future__ import annotations

import math
from pathlib import Path

from src.common.file_io import load_json
from src.datasets.prompt import (
    RiskPromptConfig,
    RiskPromptDataset,
    RiskPromptSample,
)


def load_risk_prompt_dataset(path: Path, subsample: float) -> RiskPromptDataset:
    """Load a RiskPromptDataset, striding the RAW json before deserializing."""
    if subsample >= 1.0:
        return RiskPromptDataset.from_json(path)
    # load_json (not raw json) rejoins readable_text line-lists back to strings.
    data = load_json(Path(path))
    raw = data["samples"]
    n = max(1, math.ceil(len(raw) * subsample))
    stride = max(1, len(raw) // n)
    kept = [RiskPromptSample.from_dict(d) for d in raw[::stride][:n]]
    dataset = RiskPromptDataset(
        dataset_id=data["dataset_id"],
        config=RiskPromptConfig.from_dict(data["config"]),
        samples=kept,
    )
    return dataset
