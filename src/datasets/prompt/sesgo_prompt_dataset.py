"""A generated SESGO prompt dataset: its config, scaffolds used, and samples.

Serialization leans entirely on BaseSchema (`to_dict`/`from_dict` already
reconstruct the nested config, the enum labels, and the marker tuples), so
save/load are thin file_io wrappers — no bespoke parsing to drift out of sync.
`scaffold_ids` records which scaffolds the run crossed over, so the dataset's
scaffold provenance survives even though the Scaffold texts live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common import BaseSchema
from src.common.file_io import ensure_dir, get_timestamp, save_json
from .sesgo_prompt_config import SesgoPromptConfig
from .sesgo_prompt_sample import SesgoPromptSample


@dataclass
class SesgoPromptDataset(BaseSchema):
    """All SESGO prompts generated for one config run."""

    dataset_id: str
    config: SesgoPromptConfig
    scaffold_ids: list[str] = field(default_factory=list)
    samples: list[SesgoPromptSample] = field(default_factory=list)
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
