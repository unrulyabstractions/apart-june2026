"""Build the selection bias-alignment-vs-accuracy figure (one segment per SCAFFOLD).

Anchors on the high-power full-data run (``out/sesgo/full_data/Qwen3-0.6B``, 16164
items) and splits it by scaffold (no-scaffold baseline + the three debiasing
scaffolds). Each scaffold becomes one segment per panel: ambiguous abstention /
disambiguated correctness, with the signed target-vs-other lean as the span. Same
two-panel reference layout as the baseline figure. Existing data only.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402
from sesgo.baseline.bias_alignment_accuracy_figure import plot_bias_alignment  # noqa: E402
from sesgo.baseline.bias_alignment_group_labels import (  # noqa: E402
    scaffold_color,
    scaffold_display_name,
    scaffold_sort_key,
)
from sesgo.baseline.bias_alignment_segments import (  # noqa: E402
    BiasSegment,
    segments_for_group,
)

_FULL_DATA = Path("out/sesgo/full_data/Qwen3-0.6B")
# Stable string key for each scaffold group ("None" == the no-scaffold baseline).
_BASELINE_KEY = "None"


def _scaffold_key(scaffold_id: str | None) -> str:
    """Group key for a sample's scaffold (string so it serialises cleanly)."""
    return scaffold_id if scaffold_id is not None else _BASELINE_KEY


def _collect() -> list[BiasSegment]:
    """Split the full-data run by scaffold -> per-scaffold bias-alignment segments."""
    dataset = SesgoDataset.from_json(_FULL_DATA / "response_samples.json")
    by_scaffold: dict[str, list] = defaultdict(list)
    for sample in dataset.samples:
        by_scaffold[_scaffold_key(sample.scaffold_id)].append(sample)
    out: list[BiasSegment] = []
    for key, samples in by_scaffold.items():
        out += segments_for_group(key, samples)
    return out


def main() -> None:
    """Collect per-scaffold segments and render the two-panel figure."""
    segments = _collect()
    keys = sorted({s.group_key for s in segments}, key=scaffold_sort_key)
    colors = {k: scaffold_color(None if k == _BASELINE_KEY else k) for k in keys}
    names = {k: scaffold_display_name(None if k == _BASELINE_KEY else k) for k in keys}
    n = sum(s.total for s in segments if s.panel == "disambig") + sum(
        s.total for s in segments if s.panel == "ambig"
    )
    plots_dir = _FULL_DATA / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "bias_alignment_accuracy.png"
    suptitle = (
        f"Do debiasing scaffolds move where Qwen3-0.6B leans?   ({len(keys)} scaffolds, "
        f"{n} readouts)\n"
        "How to read: each scaffold is a horizontal bar at its accuracy; the bar spans its "
        "bias on neutral-vs-negative wording. Left (-1) = leans to the OTHER group, "
        "right (+1) = leans to the stereotyped group, centre = unbiased. Wilson 95% CIs."
    )
    plot_bias_alignment(segments, colors, names, keys, suptitle, out_path)
    print(f"wrote {out_path}  ({len(keys)} scaffolds, {len(segments)} segments)")


if __name__ == "__main__":
    main()
