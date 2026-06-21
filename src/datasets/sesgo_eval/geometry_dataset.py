"""A model's SESGO judgements + residual geometry over one prompt dataset.

The geometry-half analogue of SesgoDataset: same (prompt_dataset_id, model,
config) header, but its samples are GeometrySamples that additionally point at
the saved per-position residual tensors. Serialization leans entirely on
BaseSchema (the activations carry only relative paths, never tensors), so
save/load stay thin file_io wrappers with nothing bespoke to drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common import BaseSchema
from src.common.file_io import ensure_dir, save_json
from .geometry_sample import GeometrySample
from .sesgo_query_config import SesgoQueryConfig


@dataclass
class GeometryDataset(BaseSchema):
    """All GeometrySamples produced for one (prompt dataset, model)."""

    prompt_dataset_id: str
    model: str
    config: SesgoQueryConfig
    samples: list[GeometrySample] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        """Bare model name, dropping any org prefix."""
        return self.model.split("/")[-1]

    def save_as_json(self, path: Path | str) -> None:
        """Write the dataset to JSON, creating the parent directory."""
        path = Path(path)
        ensure_dir(path.parent)
        save_json(self.to_dict(), path)
