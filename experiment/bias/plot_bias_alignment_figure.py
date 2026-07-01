"""Build the SESGO bias-alignment-vs-accuracy figure (arXiv:2509.03329, count-based) for
every model in out/stability/, reusing the recovered two-panel plotter.

Each model is one horizontal segment per panel (Ambiguous | Disambiguated): y = accuracy,
x = F(Target) - F(Other) from answer COUNTS, span = the two wording endpoints. Colour by
family (darker = larger within a family); thinking variants are tagged in the label.

Usage:
  uv run python -m experiment.bias.plot_bias_alignment_figure \
    --stability-dir out/stability --dataset data/full_prompt_dataset.json --out-dir paper/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment.bias.bias_alignment_accuracy_figure import plot_bias_alignment
from experiment.bias.bias_model_style import ModelStyle
from experiment.bias.bias_segments import segments_for_group
from experiment.bias.stability_readout_join import enrich, load_metadata
from experiment.common.sweep_models import FAMILY_COLOR, FAMILY_ORDER, parse_model


def _size_label(size_b: float) -> str:
    """Short size tag: '0.8B', '2B', '27B' (drop a trailing '.0')."""
    return (f"{size_b:g}B")


def build(stab_root: Path, dataset_path: Path):
    meta = load_metadata(dataset_path)
    models = [m for d in sorted(stab_root.iterdir()) if d.is_dir()
              for m in (parse_model(d.name),) if m
              and (stab_root / d.name / "response_samples.json").exists()]
    segments: list = []
    styles: dict[str, ModelStyle] = {}
    for m in models:
        samples = json.load((stab_root / m.dir_name / "response_samples.json").open())["samples"]
        segments += segments_for_group(m.dir_name, enrich(samples, m.dir_name, meta))
        styles[m.dir_name] = ModelStyle(
            group_key=m.dir_name, family=m.family, size_b=m.size_b,
            is_thinking=(m.mode == "thinking"), color=FAMILY_COLOR[m.family],
            size_label=_size_label(m.size_b))
    order = [m.family for m in sorted(models, key=lambda m: FAMILY_ORDER.index(m.family))]
    family_order = list(dict.fromkeys(order))  # unique, in family order
    return segments, styles, family_order


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stability-dir", default="out/stability")
    ap.add_argument("--dataset", default="data/full_prompt_dataset.json")
    ap.add_argument("--out-dir", default="paper/figures")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    segments, styles, family_order = build(Path(args.stability_dir), Path(args.dataset))
    if not segments:
        print("[bias] no model slices found under", args.stability_dir); return
    out_path = out_dir / "fig3_bias_alignment_accuracy.png"
    plot_bias_alignment(
        segments, styles, family_order, FAMILY_COLOR,
        "Bias vs. accuracy across model scale (SESGO)", out_path)
    print(f"[bias] wrote {out_path}  ({len(styles)} models, {len(segments)} segments)")


if __name__ == "__main__":
    main()
