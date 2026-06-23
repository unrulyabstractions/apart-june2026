"""Forced-fork size sweep: rows per model FAMILY, 3 panels each, X = model SIZE (log).

For every `out/forked/<bare>-<mode>/response_samples.json` slice we group the 2 order-flip
forks of each base item (consecutive `sample_idx` pairs, idx//2 = item key) and report, per
model:
  (a) avg output diversity  = mean over items of exp(H) of the item's CHOICE distribution
                              (`q_diversity(..., q=1)` = effective # of distinct fork outcomes),
  (b) avg vocab entropy     = mean per-prompt `vocab_diversity`,
  (c) avg label prob        = mean per-prompt `label_prob`.
One line per family across the three panels; thinking / non-thinking are distinct series.
Models are discovered from disk (no hardcoded lists) and parsed via the shared sweep parser.

Usage:
  uv run python -m experiment.forking.forced_fork_size_sweep_figure \
    --forked-dir out/forked --out-dir paper/figures
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from experiment.common.sweep_models import FAMILY_COLOR, FAMILY_ORDER, parse_model
from experiment.forking.forking_plot_styles import save_fig
from src.common import BaseSchema
from src.common.math import q_diversity

_PANELS = (
    ("avg output diversity\n(effective # fork outcomes)", "output_diversity"),
    ("avg vocab entropy", "vocab_entropy"),
    ("avg label prob", "label_prob"),
)


@dataclass
class ModelSweepPoint(BaseSchema):
    """One model slice aggregated to the three sweep metrics."""

    family: str
    size_b: float
    mode: str
    name: str
    n_items: int
    output_diversity: float
    vocab_entropy: float
    label_prob: float


def _item_outcome_diversity(choices: list[str]) -> float:
    """exp(H) of an item's fork-CHOICE distribution = effective # distinct outcomes."""
    counts = Counter(choices)
    total = sum(counts.values())
    logprobs = [math.log(c / total) for c in counts.values()]
    return q_diversity(logprobs, q=1.0)


def _aggregate(model, samples: list[dict]) -> ModelSweepPoint:
    """Group the 2 forks per base item (idx//2) and average the three metrics."""
    item_choices: dict[int, list[str]] = {}
    for s in samples:
        item_choices.setdefault(s["sample_idx"] // 2, []).append(s["choice"])
    diversities = [_item_outcome_diversity(c) for c in item_choices.values()]
    return ModelSweepPoint(
        family=model.family, size_b=model.size_b, mode=model.mode, name=model.name,
        n_items=len(item_choices),
        output_diversity=sum(diversities) / len(diversities),
        vocab_entropy=sum(s["vocab_diversity"] for s in samples) / len(samples),
        label_prob=sum(s["label_prob"] for s in samples) / len(samples),
    )


def load_points(forked_root: Path) -> list[ModelSweepPoint]:
    """Discover every parseable model slice on disk and aggregate it (no hardcoded lists)."""
    points = []
    for d in sorted(forked_root.iterdir()):
        model = parse_model(d.name) if d.is_dir() else None
        samples_path = d / "response_samples.json"
        if model is None or not samples_path.exists():
            continue
        samples = json.load(samples_path.open())["samples"]
        if samples:
            points.append(_aggregate(model, samples))
    return points


def _series(points: list[ModelSweepPoint]) -> dict[tuple[str, str], list[ModelSweepPoint]]:
    """Group points into one line per (family, mode), sorted by size for clean lines."""
    series: dict[tuple[str, str], list[ModelSweepPoint]] = {}
    for p in points:
        series.setdefault((p.family, p.mode), []).append(p)
    for pts in series.values():
        pts.sort(key=lambda p: p.size_b)
    return series


def plot(points: list[ModelSweepPoint], out_path: Path) -> Path:
    """Rows = families present; columns = the 3 metrics; X = size (log), one line per mode."""
    families = [f for f in FAMILY_ORDER if any(p.family == f for p in points)]
    series = _series(points)
    fig, axes = plt.subplots(len(families), 3, figsize=(13, 3.1 * len(families)), squeeze=False)
    for row, fam in enumerate(families):
        for col, (ylabel, attr) in enumerate(_PANELS):
            ax = axes[row][col]
            for (sfam, mode), pts in series.items():
                if sfam != fam:
                    continue
                style = "--" if mode == "thinking" else "-"
                ax.plot([p.size_b for p in pts], [getattr(p, attr) for p in pts],
                        style, marker="o", color=FAMILY_COLOR[fam],
                        label="thinking" if mode == "thinking" else "non-thinking")
            ax.set_xscale("log")
            ax.set_xlabel("model size (B params)")
            if col == 0:
                ax.set_ylabel(f"{fam}\n{ylabel}", fontweight="bold")
            else:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(ylabel.split("\n")[0], fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)
            handles, _ = ax.get_legend_handles_labels()
            if len(set(h.get_linestyle() for h in handles)) > 1:
                ax.legend(fontsize=7)
    fig.suptitle("Forced-fork size sweep (per model family)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return save_fig(fig, out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forked-dir", default="out/forked")
    ap.add_argument("--out-dir", default="paper/figures")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = load_points(Path(args.forked_dir))
    if not points:
        print("[forked] no model slices found under", args.forked_dir)
        return
    out_path = plot(points, out_dir / "forced_fork_size_sweep.png")
    for p in sorted(points, key=lambda p: (FAMILY_ORDER.index(p.family), p.size_b, p.mode)):
        print(f"[forked] {p.name:18s} ({p.mode:12s}) n={p.n_items:3d}  "
              f"div={p.output_diversity:.4f}  vocab={p.vocab_entropy:.4f}  "
              f"label_prob={p.label_prob:.4f}")
    print(f"[forked] wrote {out_path}  ({len(points)} model slices)")


if __name__ == "__main__":
    main()
