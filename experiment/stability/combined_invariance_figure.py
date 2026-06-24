"""TWO-panel format-stability figure for the greedy-readout sweep (`out/stability/<dir>`).

We split the single "role identical across all 3 orthogonal variants" metric into its two
orthogonal axes, each its own panel:
  - ORDER stability  = fraction of base items where choice(baseline) == choice(flip), i.e. the
    role survives swapping the target/other ANSWER POSITIONS.   [needs baseline & flip present]
  - LABEL stability  = fraction of base items where choice(baseline) == choice(swap), i.e. the
    role survives swapping the label style a)b)c) -> x)y)z).      [needs baseline & swap present]

Both panels: y = agreement rate (Wilson 95% CI), x = model size (log), family-coloured, the
committed greedy `choice`; open circle = non-thinking, filled triangle = thinking. A thin
family-coloured line joins each (family, marker) series in size order, and EVERY point carries
its proper versioned name (e.g. "Qwen3.5 4B"), de-collided into a left/right margin column so
the ~17 labels stay legible.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from experiment.bias.segment_label_layout import spread_labels
from experiment.bias.stability_readout_join import load_metadata
from experiment.common.sweep_models import FAMILY_COLOR, parse_model
from experiment.stability.new_sweep_baseline_stability_figures import _VARIANT
from src.common.math import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[2]


def new_invariance(slice_dir: Path, meta: dict) -> dict:
    """ORDER + LABEL stability for one sweep slice, with their per-axis denominators.

    ORDER needs baseline+flip and LABEL needs baseline+swap, so we group per base item and
    score each axis over only the items carrying both of its variants (the larger,
    axis-specific denominator gives a tighter Wilson CI)."""
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
    """Group a record's variants back to their shared base item (mirrors new_sweep figure)."""
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


def _collect(new_dir: Path, meta: dict) -> list[dict]:
    """One row per model slice: family, size, marker, colour, proper name, both stabilities."""
    rows: list[dict] = []
    for name in sorted(p.name for p in new_dir.iterdir()
                       if (p / "response_samples.json").exists()):
        sm = parse_model(name)
        if sm is None:
            print(f"  SKIP unparsed: {name}")
            continue
        iv = new_invariance(new_dir / name, meta)
        marker = "^" if sm.mode == "thinking" else "o"
        rows.append({"name": sm.name, "family": sm.family, "size": sm.size_b,
                     "marker": marker, "color": FAMILY_COLOR[sm.family], **iv})
    return rows


def _label_side(ax, group: list[tuple], x_text: float, ha: str) -> None:
    """Draw one margin column of labels: sort by y, spread vertically, thin leader to each point.
    Sorting by y makes the leaders monotonic, so a column of many labels never crosses itself."""
    group = sorted(group, key=lambda g: g[1])
    for (x, y, text, color), slot in zip(group, spread_labels([g[1] for g in group], 0.07, 1.0)):
        ax.annotate(text, (x, y), xytext=(x_text, slot.y_label), textcoords="data",
                    fontsize=7, color=color, va="center", ha=ha, zorder=4,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.6))


def _label_points(ax, labels: list[tuple], xmin: float, xmax: float) -> None:
    """Name every point, splitting the labels into a left and a right margin column by SIZE rank
    (smaller half left, larger half right) so the two columns carry an equal share and neither
    crowds — most models are small, so an x-midpoint split would overload the left column."""
    by_size = sorted(labels, key=lambda l: l[0])
    half = (len(by_size) + 1) // 2
    _label_side(ax, by_size[:half], xmin / 1.9, "right")
    _label_side(ax, by_size[half:], xmax * 1.9, "left")


def _draw_panel(ax, rows: list[dict], axis: str, title: str) -> None:
    """Render one stability axis ('order' or 'label') as named family trend-lines with Wilson CIs.

    Each marker is one model (family colour, mode marker, Wilson 95% CI); a thin family-coloured
    line joins every (family, marker) series in size order, and every point is labelled with its
    proper versioned name in a de-collided left/right margin column."""
    n_key, succ_key = f"{axis}_n", axis
    pts = [r for r in rows if r[n_key]]
    series: dict[tuple, list[tuple]] = defaultdict(list)
    labels: list[tuple] = []
    for r in pts:
        p = _plot_point(ax, r["size"], r[succ_key], r[n_key], r["color"], r["marker"], ms=8)
        series[(r["family"], r["marker"])].append((r["size"], p, r["color"]))
        labels.append((r["size"], p, r["name"], r["color"]))
    for group in series.values():
        group.sort(key=lambda g: g[0])
        if len(group) >= 2:  # a lone model is just its marker, not a broken one-point line
            ax.plot([g[0] for g in group], [g[1] for g in group], "-",
                    color=group[0][2], lw=0.9, alpha=0.5, zorder=1)
    sizes = [r["size"] for r in pts]
    xmin, xmax = min(sizes), max(sizes)
    _label_points(ax, labels, xmin, xmax)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(xmin / 3.2, xmax * 3.2)  # margin room so the offset labels are not clipped
    ax.set_xlabel("Model size (billion parameters)")
    ax.set_title(title)


def build(new_dir: Path, dataset: Path, out: Path) -> None:
    meta = load_metadata(dataset)
    rows = _collect(new_dir, meta)

    fig, (ax_order, ax_label) = plt.subplots(1, 2, figsize=(17, 8), sharey=True)
    _draw_panel(ax_order, rows, "order", "Stability when answer positions are swapped")
    _draw_panel(ax_label, rows, "label", "Stability when answer labels are swapped")
    ax_order.set_ylabel("Agreement rate (95% CI)")

    fams = sorted({r["family"] for r in rows},
                  key=lambda f: list(FAMILY_COLOR).index(f) if f in FAMILY_COLOR else 9)
    handles = [_legend_marker(marker="o", color=FAMILY_COLOR[f], label=f) for f in fams]
    handles += [
        _legend_marker(marker="o", mfc="white", mec="k", label="Standard"),
        _legend_marker(marker="^", color="k", label="Reasoning"),
    ]
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=len(fams) + 2,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Answer stability by model size", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")

    print(f"{'model':18s} {'fam':8s} {'size':>6s}  {'order':>22s}  {'label':>22s}")
    for r in sorted(rows, key=lambda r: r["size"]):
        o = f"{r['order']}/{r['order_n']}={r['order']/r['order_n']:.3f}" if r["order_n"] else "-"
        l = f"{r['label']}/{r['label_n']}={r['label']/r['label_n']:.3f}" if r["label_n"] else "-"
        print(f"{r['name']:18s} {r['family']:8s} {r['size']:6g}  {o:>22s}  {l:>22s}")
    print(f"\nWrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-dir", type=Path, default=REPO_ROOT / "out" / "stability")
    ap.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "full_prompt_dataset.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "paper" / "figures" / "combined_invariance_old_new.png")
    a = ap.parse_args()
    build(a.new_dir, a.dataset, a.out)


if __name__ == "__main__":
    main()
