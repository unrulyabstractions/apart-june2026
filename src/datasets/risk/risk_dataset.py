"""A model's risk judgements over one prompt dataset, plus save/load.

Serialization leans entirely on BaseSchema: to_dict/from_dict already
reconstruct the nested config and per-sample results (the heavy raw
completions are dropped via their `_` prefix), so save/load are thin file_io
wrappers with no bespoke parsing to drift out of sync — mirroring
RiskPromptDataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common import BaseSchema
from src.common.file_io import ensure_dir, save_json
from .risk_assessment_sample import RiskAssessmentSample
from .risk_query_config import RiskQueryConfig


@dataclass
class RiskDataset(BaseSchema):
    """All risk-assessment samples produced for one (prompt dataset, model)."""

    prompt_dataset_id: str
    model: str
    config: RiskQueryConfig
    samples: list[RiskAssessmentSample] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        """Bare model name, dropping any org prefix (mirrors PreferenceDataset)."""
        return self.model.split("/")[-1]

    def save_as_json(self, path: Path | str) -> None:
        """Write the dataset to JSON, creating the parent directory."""
        path = Path(path)
        ensure_dir(path.parent)
        save_json(self.to_dict(), path)
