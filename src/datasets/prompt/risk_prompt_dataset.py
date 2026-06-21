"""A generated risk-assessment dataset: its config plus rendered samples.

Serialization leans entirely on BaseSchema (`to_dict`/`from_dict` already
reconstruct the nested config, the enum task types, and the label tuples), so
save/load are thin file_io wrappers — no bespoke parsing to drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common import BaseSchema
from src.common.file_io import ensure_dir, get_timestamp, save_json
from .risk_prompt_config import RiskPromptConfig
from .risk_prompt_sample import RiskPromptSample


@dataclass
class RiskPromptDataset(BaseSchema):
    """All risk prompts generated for one config run."""

    dataset_id: str
    config: RiskPromptConfig
    samples: list[RiskPromptSample] = field(default_factory=list)
    # Set at save time for provenance; not part of identity.
    _timestamp: str = ""

    def save_as_json(self, path: Path | str) -> None:
        """Write the dataset to JSON, stamping the save time for provenance."""
        self._timestamp = get_timestamp()
        path = Path(path)
        ensure_dir(path.parent)
        data = self.to_dict()
        data["timestamp"] = self._timestamp
        save_json(data, path)
