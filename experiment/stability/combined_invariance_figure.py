"""TWO-panel format-stability figure spanning BOTH data generations: the OLD stability study
(`data/hf_old/sesgo/stability/<model>`) and the NEW greedy-readout sweep (`out/stability/<dir>`).

We split the single "role identical across all 3 orthogonal variants" metric into its two
orthogonal axes, each its own panel:
  - ORDER stability  = fraction of base items where choice(baseline) == choice(flip), i.e. the
    role survives swapping the target/other ANSWER POSITIONS.   [needs baseline & flip present]
  - LABEL stability  = fraction of base items where choice(baseline) == choice(swap), i.e. the
    role survives swapping the label style a)b)c) -> x)y)z).      [needs baseline & swap present]

Both panels: y = agreement rate (Wilson 95% CI), x = model size (log), family-coloured, OLD =
filled SQUARE / NEW = open circle (non-thinking) / filled triangle (thinking). OLD choices are
parser-free (argmax of `non_thinking.prob`, via `old_nonthinking_role` — NEVER mapped through
option positions, which would wrongly swap target<->other); NEW choices are the committed greedy
`choice`. EVERY point carries an explicit model label (name + n), de-collided by `spread_labels`.
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
from experiment.stability.new_sweep_baseline_stability_figures import _VARIANT
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
    """Same 3-orthogonal-variant role choices as the NEW sweep, on OLD data. Item =
    (question_id, replicate-index): the replicate index disambiguates the two draws per
    (qid, ordering, style) combo so every item gets exactly one baseline/flip/swap choice.

    Returns ORDER stability (baseline==flip over items having both) and LABEL stability
    (baseline==swap over items having both), each with its own n — matching the NEW
    `_invariance` order/label semantics but on a per-axis denominator."""
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
    return _split_stability(items.values())


def new_invariance(slice_dir: Path, meta: dict) -> dict:
    """ORDER + LABEL stability for one NEW sweep slice, with their per-axis denominators.

    The shared `_invariance` requires all 3 variants per item; here ORDER only needs
    baseline+flip and LABEL only needs baseline+swap, so we re-group from scratch to use
    the larger, axis-specific denominators (more items, tighter Wilson CI)."""
    samples = json.load((slice_dir / "response_samples.json").open())["samples"]
    choice = {s["prompt_id"]: s["choice"] for s in samples}
    items: dict = defaultdict(dict)
    for pid, m in meta.items():
        v = _VARIANT.get((tuple(m["position_labels"]), m["label_style"]))
        if v is None or pid not in choice:
            continue
        items[_item_key(m)][v] = choice[pid]
    return _split_stability(items.values())


def _item_key(m: dict) -> tuple:
    """Group a NEW record's variants back to their shared base item (mirrors new_sweep figure)."""
    return (m["question_id"], m["context_condition"], m["question_polarity"],
            m["bias_category"], m["target_identity"], m["other_identity"], m["language"])


def _split_stability(items) -> dict:
    """From per-item {variant: choice} dicts, the ORDER (baseline vs flip) and LABEL
    (baseline vs swap) agreement counts, each over only the items carrying both variants."""
    order_items = [d for d in items if {"baseline", "flip"} <= d.keys()]
    label_items = [d for d in items if {"baseline", "swap"} <= d.keys()]
    return {
        "order_n": len(order_items),
        "order": sum(d["baseline"] == d["flip"] for d in order_items),
        "label_n": len(label_items),
        "label": sum(d["baseline"] == d["swap"] for d in label_items),
    }


def _legend_marker(**kw):
    return plt.Line2D([], [], ls="", **kw)  # shared zero-length marker for the manual legend


def _plot_point(ax, size, succ, n, color, marker, ms) -> float:
    p, lo, hi = wilson_interval(succ, n)
    # Clamp tiny floating-point negatives: at p==1.0 the Wilson bound can land at 0.999...9.
    mfc = "white" if marker == "o" else color
    ax.errorbar(size, p, yerr=[[max(p - lo, 0.0)], [max(hi - p, 0.0)]], fmt=marker, color=color,
                mfc=mfc, mec=color, ms=ms, capsize=3, mew=1.6, zorder=3)
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
    (the leftmost cluster spills fully into the empty left margin) and `spread_labels` pushes
    each side apart in y with a thin family-coloured leader line.
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


def _collect(new_dir: Path, old_root: Path, meta: dict) -> list[dict]:
    """One row per (generation, model): family, size, marker, colour, and both stabilities."""
    rows: list[dict] = []
    for name in sorted(d.name for d in old_root.iterdir()
                       if (d / "response_samples.json").is_file()):
        fam, size = model_meta(name)
        iv = old_invariance(old_root / name)
        rows.append({"gen": "OLD", "name": f"{fam} {size:g}B-old", "family": fam, "size": size,
                     "marker": "s", "ms": 10, "color": FAMILY_COLOR.get(fam, "gray"), **iv})
    for name in sorted(p.name for p in new_dir.iterdir()
                       if (p / "response_samples.json").exists()):
        sm = parse_model(name)
        if sm is None:
            print(f"  SKIP unparsed: {name}")
            continue
        iv = new_invariance(new_dir / name, meta)
        marker = "^" if sm.mode == "thinking" else "o"
        rows.append({"gen": "NEW", "name": sm.name, "family": sm.family, "size": sm.size_b,
                     "marker": marker, "ms": 8, "color": FAMILY_COLOR[sm.family], **iv})
    return rows


def _draw_panel(ax, rows: list[dict], axis: str, title: str) -> None:
    """Render one stability axis ('order' or 'label') as a labelled scatter with Wilson CIs."""
    n_key, succ_key = f"{axis}_n", axis
    labels: list[tuple] = []
    for r in rows:
        n = r[n_key]
        if not n:
            continue
        p = _plot_point(ax, r["size"], r[succ_key], n, r["color"], r["marker"], r["ms"])
        labels.append((r["size"], p, f"{r['name']} (n={n})", r["color"]))
    _label_points(ax, labels)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.grid(True, which="both", alpha=0.25)
    xs = [l[0] for l in labels]
    ax.set_xlim(min(xs) * 0.28, max(xs) * 3.0)  # headroom so offset labels aren't clipped
    ax.set_xlabel("Model size (B params, log scale)")
    ax.set_title(title)


def build(new_dir: Path, dataset: Path, old_root: Path, out: Path) -> None:
    meta = load_metadata(dataset)
    rows = _collect(new_dir, old_root, meta)

    fig, (ax_order, ax_label) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    _draw_panel(ax_order, rows, "order", "Order stability (target/other position-flip)")
    _draw_panel(ax_label, rows, "label", "Label stability (label-style swap)")
    ax_order.set_ylabel("Agreement rate (Wilson 95% CI)")

    fams = sorted({r["family"] for r in rows},
                  key=lambda f: list(FAMILY_COLOR).index(f) if f in FAMILY_COLOR else 9)
    handles = [_legend_marker(marker="o", color=FAMILY_COLOR[f], label=f) for f in fams]
    handles += [
        _legend_marker(marker="s", mfc="gray", mec="gray", color="gray",
                       label="Older models (Qwen3 / Gemma-2 / Mistral / Llama-3)"),
        _legend_marker(marker="o", mfc="white", mec="k", label="New sweep — non-thinking"),
        _legend_marker(marker="^", color="k", label="New sweep — thinking"),
    ]
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=len(fams) + 3,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Order vs label stability across scale (both data generations)", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")

    print(f"{'gen':4s} {'model':24s} {'fam':8s} {'size':>6s}  {'order':>22s}  {'label':>22s}")
    for r in sorted(rows, key=lambda r: (r["gen"], r["size"])):
        o = f"{r['order']}/{r['order_n']}={r['order']/r['order_n']:.3f}" if r["order_n"] else "-"
        l = f"{r['label']}/{r['label_n']}={r['label']/r['label_n']:.3f}" if r["label_n"] else "-"
        print(f"{r['gen']:4s} {r['name']:24s} {r['family']:8s} {r['size']:6g}  {o:>22s}  {l:>22s}")
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
