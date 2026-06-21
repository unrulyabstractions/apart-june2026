"""Build the baseline bias-alignment-vs-accuracy figure (one segment per MODEL).

Scans the SESGO baseline sweep (``out/sesgo/baseline/<model>/response_samples.json``),
reduces every model to its two panel segments (ambiguous abstention / disambiguated
correctness with the signed target-vs-other lean), and draws the two-panel reference
figure with each model coloured by family and ordered by size. Existing data only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402
from sesgo.baseline.bias_alignment_accuracy_figure import plot_bias_alignment  # noqa: E402
from sesgo.baseline.bias_alignment_group_labels import (  # noqa: E402
    model_display_name,
    model_family,
    model_sort_key,
)
from sesgo.baseline.bias_alignment_segments import (  # noqa: E402
    BiasSegment,
    segments_for_group,
)
from sesgo.baseline.cross_model_plot_styles import FAMILY_COLORS  # noqa: E402

_BASE_DIR = Path("out/sesgo/baseline")
_GREY = "#555555"


def _collect() -> list[BiasSegment]:
    """One model per dir -> its bias-alignment segments, keyed by bare model name."""
    out: list[BiasSegment] = []
    for path in sorted(_BASE_DIR.glob("*/response_samples.json")):
        dataset = SesgoDataset.from_json(path)
        out += segments_for_group(dataset.model_name, dataset.samples)
    return out


def main() -> None:
    """Collect baseline segments and render the two-panel figure."""
    segments = _collect()
    models = sorted({s.group_key for s in segments}, key=model_sort_key)
    colors = {m: FAMILY_COLORS.get(model_family(m), _GREY) for m in models}
    names = {m: model_display_name(m) for m in models}
    plots_dir = _BASE_DIR / "cross_model" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "bias_alignment_accuracy.png"
    suptitle = (
        f"Where does each model lean, and how accurate is it?   ({len(models)} models)\n"
        "How to read: each model is a horizontal bar at its accuracy; the bar spans its "
        "bias on neutral-vs-negative wording. Left (-1) = leans to the OTHER group, "
        "right (+1) = leans to the stereotyped group, centre = unbiased. Wilson 95% CIs."
    )
    plot_bias_alignment(segments, colors, names, models, suptitle, out_path)
    print(f"wrote {out_path}  ({len(models)} models, {len(segments)} segments)")


if __name__ == "__main__":
    main()
