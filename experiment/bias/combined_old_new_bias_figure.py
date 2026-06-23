"""ONE bias-alignment-vs-accuracy figure carrying BOTH the OLD baseline models AND the
NEW model-sweep models on the same two panels.

The NEW segments come from the existing `plot_bias_alignment_figure.build` (out/stability +
data/full_prompt_dataset.json). The OLD segments are computed directly from the inline
metadata in `data/hf_old/sesgo/baseline/<model>/response_samples.json` using the PARSER-FREE
choice (role at argmax of the non_thinking probability vector, via the reparse helpers) — no
dataset join is needed because every OLD sample already carries its own polarity / context /
gold / target+other identities. Each OLD sample becomes an `EnrichedResponse`, then
`segments_for_group` reduces a model to its two `BiasSegment`s exactly like the NEW path.

Both segment lists (+ their colour/name tables + draw order) are merged and rendered once via
`plot_bias_alignment`. Family colours are shared old/new (the accuracy height + bias position
separate the points); within a family the shade tracks size across the COMBINED old+new spread.

Usage:
  uv run python -m experiment.bias.combined_old_new_bias_figure \
    --stability-dir out/stability --dataset data/full_prompt_dataset.json \
    --old-root data/hf_old/sesgo/baseline \
    --out paper/figures/baseline_bias_alignment_old_new.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from experiment.baseline.old_data_parser_free_figures import choice_of, model_meta
from experiment.baseline.reparse_old_thinking_labels import (
    option_labels_from_style,
    position_labels_from_prompt,
    prob_readout,
)
from experiment.bias.bias_alignment_accuracy_figure import plot_bias_alignment
from experiment.bias.bias_segments import segments_for_group
from experiment.bias.plot_bias_alignment_figure import _shade, build
from experiment.bias.stability_readout_join import EnrichedResponse
from experiment.common.sweep_models import FAMILY_COLOR, FAMILY_ORDER, parse_model
from src.common.file_io import load_json


def _old_enriched(name: str, samples: list[dict]) -> list[EnrichedResponse]:
    """Build EnrichedResponse rows for one OLD model from its inline metadata.

    `role` is the PARSER-FREE committed choice (argmax of non_thinking prob, mapped to the
    role at that option position). All scoring fields come straight off the OLD sample, so no
    join to the prompt dataset is required.
    """
    rows: list[EnrichedResponse] = []
    for s in samples:
        labels = option_labels_from_style(s["label_style"])
        roles = position_labels_from_prompt(s, labels)
        prob = s["non_thinking"]["prob"]
        argmax, label_prob, vocab_diversity = prob_readout(prob)
        rows.append(EnrichedResponse(
            group_key=name, prompt_id=f"{s['question_id']}_{s['sample_idx']}",
            question_id=s["question_id"], role=roles[argmax], gold=s["gold_label"],
            polarity=s["question_polarity"], context=s["context_condition"],
            bias_category=s["bias_category"], label_prob=label_prob,
            vocab_diversity=vocab_diversity,
        ))
    return rows


def build_old(old_root: Path):
    """OLD segments + (per-model size/family) for every model dir under `old_root`."""
    names = sorted(d.name for d in old_root.iterdir()
                   if (d / "response_samples.json").is_file())
    segments, metas = [], {}
    for name in names:
        samples = load_json(old_root / name / "response_samples.json")["samples"]
        segments += segments_for_group(name, _old_enriched(name, samples))
        metas[name] = model_meta(name)  # (family, size_b)
    return segments, metas


def _combined_shades(new_metas, old_metas) -> dict[str, str]:
    """Family-shared colours: shade each model by size across the COMBINED old+new spread."""
    fam_sizes: dict[str, list[float]] = {}
    for fam, size in list(new_metas.values()) + list(old_metas.values()):
        fam_sizes.setdefault(fam, []).append(size)
    colors: dict[str, str] = {}
    for key, (fam, size) in {**new_metas, **old_metas}.items():
        sizes = fam_sizes[fam]
        lo, hi = math.log(min(sizes)), math.log(max(sizes))
        frac = 1.0 if hi == lo else (math.log(size) - lo) / (hi - lo)
        colors[key] = _shade(FAMILY_COLOR[fam], frac)
    return colors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stability-dir", default="out/stability")
    ap.add_argument("--dataset", default="data/full_prompt_dataset.json")
    ap.add_argument("--old-root", default="data/hf_old/sesgo/baseline")
    ap.add_argument("--out", default="paper/figures/baseline_bias_alignment_old_new.png")
    args = ap.parse_args()

    # NEW: reuse the existing builder verbatim (out/stability + prompt dataset).
    new_segs, _new_colors, new_names, _new_order = build(
        Path(args.stability_dir), Path(args.dataset))
    # (family, size_b) per NEW model, taken from the same dir-name parser the builder used.
    new_metas = {k: (m.family, m.size_b)
                 for k in new_names
                 for m in (parse_model(k),) if m}

    # OLD: parser-free segments straight from the inline-metadata baseline samples.
    old_segs, old_metas = build_old(Path(args.old_root))
    old_names = {k: k for k in old_metas}  # OLD checkpoint names are already display-ready

    # Merge segments + tables; re-shade old+new together so family colours stay coherent.
    segments = new_segs + old_segs
    names = {**new_names, **old_names}
    colors = _combined_shades(new_metas, old_metas)

    # Draw order: family, then size, OLD before NEW within a tie so labels interleave readably.
    def sort_key(key: str):
        fam, size = {**new_metas, **old_metas}[key]
        return (FAMILY_ORDER.index(fam), size, 0 if key in old_metas else 1)
    order = sorted(names, key=sort_key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_bias_alignment(
        segments, colors, names, order,
        "SESGO bias alignment vs accuracy - OLD baselines + NEW model sweep "
        "(count-based F(Target)/F(Other))", out_path)
    print(f"[combined] wrote {out_path}")
    print(f"[combined] OLD models={len(old_metas)}  OLD segments={len(old_segs)}")
    print(f"[combined] NEW models={len(new_names)}  NEW segments={len(new_segs)}")
    print(f"[combined] total segments={len(segments)}  total models={len(order)}")


if __name__ == "__main__":
    main()
