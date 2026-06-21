"""Load a single ``[d_model]`` residual snapshot for (sample, position, layer).

Mirrors the canonical risk-geometry loader
(``risk_geometry_analysis._load_tensor``: ``torch.load(root / a.path,
map_location='cpu').numpy()``): activation ``path`` is RELATIVE to the directory
holding response_samples.json, the saved tensor is a single ``[d_model]`` float32
CPU tensor, and the activations/ entries are symlinked but resolve transparently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.datasets.sesgo_eval import GeometrySample


def load_residual(
    root: Path, sample: GeometrySample, position_type: str, layer: int
) -> np.ndarray | None:
    """Return the ``[d_model]`` residual at (position_type, layer), or None.

    ``root`` is the directory containing response_samples.json; activation paths
    are resolved against it. None is returned when the sample has no activation at
    that (position_type, layer) cell.
    """
    act = next(
        (
            a
            for a in sample.activations
            if a.position_type == position_type and a.layer == layer
        ),
        None,
    )
    if act is None:
        return None
    tensor = torch.load(root / act.path, map_location="cpu")
    return tensor.numpy().astype(np.float32)
