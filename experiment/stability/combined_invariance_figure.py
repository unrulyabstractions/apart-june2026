"""ONE format-invariance figure spanning BOTH data generations: the OLD stability study
(`data/hf_old/sesgo/stability/<model>`) and the NEW greedy-readout sweep (`out/stability/<dir>`),
as role-invariance rate vs model size.

Role-invariance = fraction of base ITEMS whose committed role is identical across the SAME 3
orthogonal variants (baseline / position-flip / label-swap), so the generations are apples-to-
apples. OLD stores 36 variants/qid; we take the 3 orthogonal combos and pair them by replicate
index within each (qid, ordering, style) so each item carries one baseline/flip/swap choice. OLD
choices are parser-free (argmax of `non_thinking.prob`); NEW are the committed greedy-readout
`choice`. OLD = filled SQUARE, NEW = open circle / filled triangle (thinking), family-coloured,
Wilson 95% CI; EVERY point carries an explicit model label (name + n), de-collided by
`spread_labels` with a thin leader line.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from experiment.baseline.old_data_parser_free_figures import model_meta
from experiment.baseline.reparse_old_thinking_labels import (
    old_nonthinking_role,
    option_labels_from_style,
    position_labels_from_prompt,
)
from experiment.bias.segment_label_layout import spread_labels
from experiment.bias.stability_readout_join import load_metadata
from experiment.common.sweep_models import FAMILY_COLOR, parse_model
from experiment.stability.new_sweep_baseline_stability_figures import _VARIANT, _invariance
from src.common.math import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[2]


def _old_choice(s: dict) -> tuple[str, tuple]:
    """Parser-free committed role + the (role-at-each-position) tuple for an OLD sample.

    The CHOICE is the OLD non_thinking committed role (argmax of the ROLE-ORDERED prob, via
    `old_nonthinking_role`); mapping through option positions would wrongly swap target<->other.
    The `tuple(roles)` from `position_labels_from_prompt` is kept ONLY to type the format VARIANT
    (baseline / position-flip / label-swap), never for the choice."""
    labels = option_labels_from_style(s["label_style"])
    roles = position_labels_from_prompt(s, labels)
    return old_nonthinking_role(s), tuple(roles)


def old_invariance(model_dir: Path) -> dict:
    """Same 3-orthogonal-variant role-invariance as the NEW sweep, on OLD data. Item =
    (question_id, replicate-index): the replicate index disambiguates the two draws per
    (qid, ordering, style) combo so every item gets exactly one baseline/flip/swap choice."""
    samples = json.load((model_dir / "response_samples.json").open())["samples"]
    seen: dict = defaultdict(int)
    items: dict = defaultdict(dict)
    for s in sorted(samples, key=lambda x: x["sample_idx"]):
        choice, roles = _old_choice(s)
        v = _VARIANT.get((roles, s["label_style"]))
        if v is None:
            continue
        combo = (s["question_id"], roles, s["label_style"])
        rep = seen[combo]
        seen[combo] += 1
        items[(s["question_id"], rep)][v] = choice
    full = [d for d in items.values() if {"baseline", "flip", "swap"} <= d.keys()]
    return {"n": len(full), "inv": sum(len(set(d.values())) == 1 for d in full)}


def _new_invariance(slice_dir: Path, meta: dict) -> dict:
    samples = json.load((slice_dir / "response_samples.json").open())["samples"]
    return _invariance(samples, meta)


def _legend_marker(**kw):
    return plt.Line2D([], [], ls="", **kw)  # shared zero-length marker for the manual legend


def _plot_point(ax, size, succ, n, color, marker, ms):
    p, lo, hi = wilson_interval(succ, n)
    # Clamp tiny floating-point negatives: at p==1.0 (perfect invariance) the Wilson upper
    # bound can land at 0.999...9, making `hi - p` a ~-1e-16 that matplotlib rejects.
    mfc = "white" if marker == "o" else color
    ax.errorbar(size, p, yerr=[[max(p - lo, 0.0)], [max(hi - p, 0.0)]], fmt=marker, color=color, mfc=mfc,
                mec=color, ms=ms, capsize=3, mew=1.6, zorder=3)
    return p


def _draw_side(ax, group: list[tuple], x_text: float, ha: str) -> None:
    """De-collide one side's labels in y (via `spread_labels`) and draw them with leaders."""
    slots = spread_labels([g[1] for g in group], 0.082, 1.0)
    for (x, y, text, color), slot in zip(group, slots):
        ax.annotate(text, (x, y), xytext=(x_text, slot.y_label),
                    textcoords="data", fontsize=7, color=color, va="center", ha=ha, zorder=4,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.6))


def _label_points(ax, pts: list[tuple]) -> None:
    """Draw an explicit model label (name + n) beside every point, de-colliding in y.

    Points are greedily clustered along x (a >0.18-decade gap starts a new cluster) so only
    near-overlapping markers compete for space; each cluster's labels are split left/right
    (the leftmost cluster spills fully into the empty left margin) and `spread_labels` (reused
    from the bias figure) pushes each side apart in y with a thin family-coloured leader line.
    """
    clusters: list[list[tuple]] = []
    for p in sorted(pts, key=lambda g: g[0]):
        if clusters and math.log10(p[0]) - math.log10(clusters[-1][-1][0]) <= 0.18:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    for ci, group in enumerate(clusters):
        group.sort(key=lambda g: g[1])
        xmin, xmax = min(g[0] for g in group), max(g[0] for g in group)
        if ci == 0:  # leftmost cluster: send ALL labels into the empty left margin (no neighbour)
            _draw_side(ax, group, xmin / 1.12, "right")
        elif len(group) >= 2:  # split: lower-y labels hug the left edge, upper-y the right edge
            mid = len(group) // 2
            _draw_side(ax, group[:mid], xmin / 1.12, "right")
            _draw_side(ax, group[mid:], xmax * 1.12, "left")
        else:
            _draw_side(ax, group, xmax * 1.2, "left")


def build(new_dir: Path, dataset: Path, old_root: Path, out: Path) -> None:
    meta = load_metadata(dataset)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    rows: list[tuple] = []
    labels: list[tuple] = []  # (x, y, "name (n=..)", color) per point, for de-collided labelling

    # OLD models: filled squares.
    old_models = sorted(d.name for d in old_root.iterdir()
                        if (d / "response_samples.json").is_file())
    for name in old_models:
        fam, size = model_meta(name)
        color = FAMILY_COLOR.get(fam, "gray")
        iv = old_invariance(old_root / name)
        p = _plot_point(ax, size, iv["inv"], iv["n"], color, "s", 10)
        # Square marker conveys generation; `-old` tag keeps the name unambiguous next to NEW.
        labels.append((size, p, f"{fam} {size:g}B-old (n={iv['n']})", color))
        rows.append(("OLD", name, fam, size, iv["n"], iv["inv"], p))

    # NEW sweep: open circle (non-thinking) / filled triangle (thinking).
    new_dirs = sorted(p.name for p in new_dir.iterdir()
                      if (p / "response_samples.json").exists())
    for name in new_dirs:
        sm = parse_model(name)
        if sm is None:
            print(f"  SKIP unparsed: {name}")
            continue
        iv = _new_invariance(new_dir / name, meta)
        if not iv["n"]:
            continue
        marker = "^" if sm.mode == "thinking" else "o"
        color = FAMILY_COLOR[sm.family]
        p = _plot_point(ax, sm.size_b, iv["inv"], iv["n"], color, marker, 8)
        labels.append((sm.size_b, p, f"{sm.name} (n={iv['n']})", color))
        rows.append(("NEW", sm.name, sm.family, sm.size_b, iv["n"], iv["inv"], p))

    _label_points(ax, labels)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.grid(True, which="both", alpha=0.25)
    xs = [l[0] for l in labels]
    ax.set_xlim(min(xs) * 0.28, max(xs) * 3.0)  # headroom so left/right offset labels aren't clipped
    ax.set_xlabel("Model size (B params, log scale)")
    ax.set_ylabel("Role-invariance rate (Wilson 95% CI)")
    ax.set_title("Format-invariance rises with scale across BOTH data generations\n"
                 "role identical across the 3 orthogonal variants (baseline / position-flip / label-swap)")

    fams = sorted({r[2] for r in rows}, key=lambda f: list(FAMILY_COLOR).index(f) if f in FAMILY_COLOR else 9)
    handles = [_legend_marker(marker="o", color=FAMILY_COLOR[f], label=f) for f in fams]
    handles += [_legend_marker(marker="s", mfc="gray", mec="gray", color="gray",
                               label="OLD (3 orthogonal variants;\nparser-free prob method)"),
                _legend_marker(marker="o", mfc="white", mec="k", label="NEW non-thinking"),
                _legend_marker(marker="^", color="k", label="NEW thinking")]
    ax.legend(handles=handles, fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)

    for gen, name, fam, size, n, inv, p in sorted(rows, key=lambda r: (r[0], r[3])):
        print(f"{gen:4s} {name:22s} {fam:8s} {size:6g}  n={n:5d}  inv={inv:5d}  rate={p:6.3f}")
    print(f"\nWrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-dir", type=Path, default=REPO_ROOT / "out" / "stability")
    ap.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "full_prompt_dataset.json")
    ap.add_argument("--old-root", type=Path, default=REPO_ROOT / "data" / "hf_old" / "sesgo" / "stability")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "paper" / "figures" / "combined_invariance_old_new.png")
    a = ap.parse_args()
    build(a.new_dir, a.dataset, a.old_root, a.out)


if __name__ == "__main__":
    main()
