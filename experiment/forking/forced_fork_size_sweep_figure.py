"""Forced-fork size sweep: ONE ROW of 3 metric panels, ALL model slices overlaid, X = SIZE (log).

Per `out/forked/<bare>-<mode>/response_samples.json` slice we group the 2 order-flip forks of
each base item (idx//2 = item key) and report, per model: (a) avg output diversity = mean exp(H)
of an item's CHOICE distribution (`q_diversity(q=1)` = effective # fork outcomes), (b) avg vocab
entropy = mean `vocab_diversity`, (c) avg label prob = mean `label_prob`. Each panel overlays
every model: colour = FAMILY, marker = non-thinking (open circle) vs thinking (filled triangle);
a thin line connects same-family+mode points (a lone point is just a marker, not broken). Every
point is labelled with its short name, de-collided in y via `spread_labels` with a thin leader.

  uv run python -m experiment.forking.forced_fork_size_sweep_figure --forked-dir out/forked --out-dir paper/figures
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiment.bias.segment_label_layout import spread_labels
from experiment.common.sweep_models import FAMILY_COLOR, FAMILY_ORDER, parse_model
from experiment.forking.forking_plot_styles import REF, save_fig
from src.common import BaseSchema
from src.common.math import q_diversity

_PANELS = (
    ("Answer variety\n(effective number of distinct answers)", "output_diversity"),
    ("Word-choice variety", "vocab_entropy"),
    ("Answer confidence", "label_prob"),
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


def _marker(mode: str) -> str:
    return "^" if mode == "thinking" else "o"


def _draw_panel(ax, points, series, attr, ylabel) -> None:
    """Overlay every model on one metric panel: family colour, mode marker, trend line + labels."""
    for (fam, mode), pts in series.items():
        ys = [getattr(p, attr) for p in pts]
        color, mk = FAMILY_COLOR[fam], _marker(mode)
        ax.plot([p.size_b for p in pts], ys, "-", color=color, lw=0.9, alpha=0.55, zorder=1)
        ax.scatter([p.size_b for p in pts], ys, marker=mk, s=42, zorder=3,
                   facecolors=color if mode == "thinking" else "none", edgecolors=color, linewidths=1.4)
    lo, hi = ax.get_ylim()
    pad = (hi - lo) * 0.08 or 0.02
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xscale("log")
    sizes = [p.size_b for p in points]
    ax.set_xlim(min(sizes) * 0.7, max(sizes) * 1.9)  # headroom so offset labels aren't clipped
    _label_points(ax, points, attr)
    ax.set_xlabel("Model size (billion parameters)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _label_points(ax, points, attr) -> None:
    """Place every model's short name next to its point, spread in y so none collide."""
    ys = [getattr(p, attr) for p in points]
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.052
    for slot in spread_labels(ys, min_gap=gap, hi=hi):
        p = points[slot.index]
        ax.annotate(p.name, xy=(p.size_b, slot.y_natural),
                    xytext=(p.size_b * 1.1, slot.y_label), fontsize=6.2, va="center",
                    color=REF, arrowprops=dict(arrowstyle="-", lw=0.4, color=REF, alpha=0.7))


def _legend(fig, points) -> None:
    """Family-colour + thinking/non-thinking-marker key."""
    fams = [f for f in FAMILY_ORDER if any(p.family == f for p in points)]
    handles = [Line2D([], [], marker="s", ls="", mfc=FAMILY_COLOR[f], mec=FAMILY_COLOR[f], label=f)
               for f in fams]
    handles += [
        Line2D([], [], marker="o", ls="", mfc="none", mec=REF, label="Standard"),
        Line2D([], [], marker="^", ls="", mfc=REF, mec=REF, label="Reasoning"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))


def plot(points: list[ModelSweepPoint], out_path: Path) -> Path:
    """One row of 3 metric panels, all models overlaid, labelled; X = size (log)."""
    series = _series(points)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (ylabel, attr) in zip(axes, _PANELS):
        _draw_panel(ax, points, series, attr, ylabel)
        ax.set_title(ylabel.split("\n")[0], fontsize=11, fontweight="bold")
    _legend(fig, points)
    fig.suptitle("Answer variety and confidence by model size", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
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
