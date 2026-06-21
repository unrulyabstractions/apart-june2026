"""A model's SESGO judgements over one prompt dataset, plus save/load.

Serialization leans entirely on BaseSchema: to_dict/from_dict already
reconstruct the nested config and per-sample distributions (the heavy raw
completions are dropped via their `_` prefix), so save/load are thin file_io
wrappers with no bespoke parsing to drift out of sync — mirroring RiskDataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common import BaseSchema
from src.common.file_io import ensure_dir, save_json
from .sesgo_query_config import SesgoQueryConfig
from .sesgo_sample import SesgoSample


@dataclass
class SesgoDataset(BaseSchema):
    """All SESGO samples produced for one (prompt dataset, model)."""

    prompt_dataset_id: str
    model: str
    config: SesgoQueryConfig
    samples: list[SesgoSample] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        """Bare model name, dropping any org prefix."""
        return self.model.split("/")[-1]

    def save_as_json(self, path: Path | str) -> None:
        """Write the dataset to JSON, creating the parent directory."""
        path = Path(path)
        ensure_dir(path.parent)
        save_json(self.to_dict(), path)
