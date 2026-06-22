"""Render the redesigned BASELINE bias-alignment figure (one point per MODEL).

Scans the SESGO baseline sweep (``out/sesgo/baseline/<model>/response_samples.json``),
reduces every model to its two (bias, accuracy) points, and draws the soft-modern
two-panel figure with each model coloured by family hue shaded by size. Existing
data only; renders to a distinct ``_b`` path so variants never collide.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402
from sesgo.baseline.bias_alignment_points_b import BiasPoint, points_for_group  # noqa: E402
from sesgo.baseline.bias_alignment_redesign_b import plot_bias_redesign  # noqa: E402
from sesgo.baseline.bias_group_labels_b import (  # noqa: E402
    model_display_name,
    model_sort_key,
)
from sesgo.baseline.family_shade_palette_b import model_colors  # noqa: E402

_BASE_DIR = Path("out/sesgo/baseline")


def _collect() -> list[BiasPoint]:
    """One model per dir -> its bias-alignment points, keyed by bare model name."""
    out: list[BiasPoint] = []
    for path in sorted(_BASE_DIR.glob("*/response_samples.json")):
        dataset = SesgoDataset.from_json(path)
        out += points_for_group(dataset.model_name, dataset.samples)
    return out


def main() -> None:
    """Collect baseline points and render the redesigned two-panel figure."""
    points = _collect()
    models = sorted({p.group_key for p in points}, key=model_sort_key)
    colors = model_colors(models)
    names = {m: model_display_name(m) for m in models}
    plots_dir = _BASE_DIR / "cross_model" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "bias_alignment_accuracy.png"
    plot_bias_redesign(points, colors, names, models, out_path, ncol=4)
    print(f"wrote {out_path}  ({len(models)} models, {len(points)} points)")


if __name__ == "__main__":
    main()
