"""Render the redesigned SELECTION bias-alignment figure (one point per SCAFFOLD).

Anchors on the high-power full-data run (``out/sesgo/full_data/Qwen3-0.6B``) and
splits it by scaffold (no-scaffold baseline + the three debiasing scaffolds). Each
scaffold becomes one (bias, accuracy) point per panel, coloured by scaffold from
the plain-language labels. Same soft-modern layout + realizable triangle + ideal
point. Existing data only; renders to a distinct ``_b`` path.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402
from sesgo.baseline.bias_alignment_points_b import BiasPoint, points_for_group  # noqa: E402
from sesgo.baseline.bias_alignment_redesign_b import plot_bias_redesign  # noqa: E402
from sesgo.baseline.bias_group_labels_b import (  # noqa: E402
    scaffold_color,
    scaffold_display_name,
    scaffold_sort_key,
)

_FULL_DATA = Path("out/sesgo/full_data/Qwen3-0.6B")
# Stable string key for each scaffold group ("None" == the no-scaffold baseline).
_BASELINE_KEY = "None"


def _scaffold_key(scaffold_id: str | None) -> str:
    """Group key for a sample's scaffold (string so it serialises cleanly)."""
    return scaffold_id if scaffold_id is not None else _BASELINE_KEY


def _collect() -> list[BiasPoint]:
    """Split the full-data run by scaffold -> per-scaffold bias-alignment points."""
    dataset = SesgoDataset.from_json(_FULL_DATA / "response_samples.json")
    by_scaffold: dict[str, list] = defaultdict(list)
    for sample in dataset.samples:
        by_scaffold[_scaffold_key(sample.scaffold_id)].append(sample)
    out: list[BiasPoint] = []
    for key, samples in by_scaffold.items():
        out += points_for_group(key, samples)
    return out


def main() -> None:
    """Collect per-scaffold points and render the redesigned two-panel figure."""
    points = _collect()
    keys = sorted({p.group_key for p in points}, key=scaffold_sort_key)
    colors = {k: scaffold_color(None if k == _BASELINE_KEY else k) for k in keys}
    names = {
        k: "No scaffold" if k == _BASELINE_KEY
        else scaffold_display_name(k)
        for k in keys
    }
    plots_dir = _FULL_DATA / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "bias_alignment_accuracy.png"
    plot_bias_redesign(points, colors, names, keys, out_path, ncol=len(keys))
    print(f"wrote {out_path}  ({len(keys)} scaffolds, {len(points)} points)")


if __name__ == "__main__":
    main()
