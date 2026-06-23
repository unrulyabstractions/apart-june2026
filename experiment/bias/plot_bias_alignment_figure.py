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
import math
from pathlib import Path

import matplotlib.colors as mcolors

from experiment.bias.bias_alignment_accuracy_figure import plot_bias_alignment
from experiment.bias.bias_segments import segments_for_group
from experiment.bias.stability_readout_join import enrich, load_metadata
from experiment.common.sweep_models import FAMILY_COLOR, FAMILY_ORDER, parse_model


def _shade(base_hex: str, frac: float) -> str:
    """Blend white->base by frac (small model light, large model dark)."""
    base = mcolors.to_rgb(base_hex)
    w = 0.35 + 0.65 * max(0.0, min(1.0, frac))
    return mcolors.to_hex(tuple(1 - w + w * b for b in base))


def build(stab_root: Path, dataset_path: Path):
    meta = load_metadata(dataset_path)
    models = [m for d in sorted(stab_root.iterdir()) if d.is_dir()
              for m in (parse_model(d.name),) if m
              and (stab_root / d.name / "response_samples.json").exists()]
    # per-family size spread (log) -> shade fraction
    fam_sizes: dict[str, list[float]] = {}
    for m in models:
        fam_sizes.setdefault(m.family, []).append(m.size_b)
    segments, colors, names = [], {}, {}
    for m in models:
        samples = json.load((stab_root / m.dir_name / "response_samples.json").open())["samples"]
        segments += segments_for_group(m.dir_name, enrich(samples, m.dir_name, meta))
        sizes = fam_sizes[m.family]
        lo, hi = math.log(min(sizes)), math.log(max(sizes))
        frac = 1.0 if hi == lo else (math.log(m.size_b) - lo) / (hi - lo)
        colors[m.dir_name] = _shade(FAMILY_COLOR[m.family], frac)
        names[m.dir_name] = m.name
    order = [m.dir_name for m in sorted(
        models, key=lambda m: (FAMILY_ORDER.index(m.family), m.size_b, m.mode))]
    return segments, colors, names, order


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stability-dir", default="out/stability")
    ap.add_argument("--dataset", default="data/full_prompt_dataset.json")
    ap.add_argument("--out-dir", default="paper/figures")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    segments, colors, names, order = build(Path(args.stability_dir), Path(args.dataset))
    if not segments:
        print("[bias] no model slices found under", args.stability_dir); return
    out_path = out_dir / "fig3_bias_alignment_accuracy.png"
    plot_bias_alignment(
        segments, colors, names, order,
        "Accuracy versus which group the model leans toward", out_path)
    print(f"[bias] wrote {out_path}  ({len(order)} models, {len(segments)} segments)")


if __name__ == "__main__":
    main()
