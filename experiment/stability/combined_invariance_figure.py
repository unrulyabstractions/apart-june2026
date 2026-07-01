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


_AXES = (("order", "Position swap: option order flipped"),
         ("label", "Label swap: a)b)c) $\\rightarrow$ x)y)z)"))


def _draw_cell(ax, fam_rows: list[dict], axis: str, xlim: tuple[float, float]) -> None:
    """One family x one swap axis: agreement vs size, direct (solid) / thinking (dashed)."""
    n_key, succ_key = f"{axis}_n", axis
    pts = [r for r in fam_rows if r[n_key]]
    series: dict[str, list[tuple]] = defaultdict(list)
    for r in pts:
        p = _plot_point(ax, r["size"], r[succ_key], r[n_key], r["color"], r["marker"], ms=8)
        series[r["marker"]].append((r["size"], p, r["color"]))
    for marker, group in series.items():
        group.sort(key=lambda g: g[0])
        if len(group) >= 2:
            ax.plot([g[0] for g in group], [g[1] for g in group],
                    ls="--" if marker == "^" else "-", color=group[0][2], lw=1.4,
                    alpha=0.55, zorder=1)
    ax.set_xscale("log"); ax.set_ylim(0.40, 1.02); ax.set_xlim(*xlim)
    ax.axhline(0.95, color="#CC3311", ls="--", lw=1.1, zorder=0)  # 95% reference (red)
    ax.axhline(0.90, color="#999999", ls="--", lw=1.0, zorder=0)  # 90% reference (grey)
    ax.grid(True, which="major", axis="y", ls=":", alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build(new_dir: Path, dataset: Path, out: Path) -> None:
    meta = load_metadata(dataset)
    rows = _collect(new_dir, meta)
    fams = [f for f in FAMILY_COLOR if any(r["family"] == f for r in rows)]
    sizes = [r["size"] for r in rows]
    xlim = (min(sizes) / 1.6, max(sizes) * 1.6)

    fig, axes = plt.subplots(len(fams), 2, figsize=(11, 2.15 * len(fams) + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.11, right=0.985, wspace=0.08, hspace=0.28, top=0.90, bottom=0.09)
    for r_i, fam in enumerate(fams):
        fam_rows = [r for r in rows if r["family"] == fam]
        for c_i, (axis, title) in enumerate(_AXES):
            ax = axes[r_i][c_i]
            _draw_cell(ax, fam_rows, axis, xlim)
            if r_i == 0:
                ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
            if r_i == len(fams) - 1:
                ax.set_xlabel("Model size (billion parameters)", fontsize=10.5)
        axes[r_i][0].set_ylabel(fam, fontsize=12, fontweight="bold", color=FAMILY_COLOR[fam])

    handles = [_legend_marker(marker="o", mfc="white", mec="k", ms=9, label="direct (solid line)"),
               _legend_marker(marker="^", color="k", ms=9, label="thinking / reasoning (dashed)"),
               plt.Line2D([], [], color="#CC3311", ls="--", lw=1.3, label="95%"),
               plt.Line2D([], [], color="#999999", ls="--", lw=1.3, label="90%")]
    fig.legend(handles=handles, fontsize=9.5, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.965), columnspacing=1.6)
    fig.suptitle("Answer stability rises with model size (by family)", fontsize=13.5,
                 fontweight="bold", y=0.995)
    fig.text(0.028, 0.5, "Agreement rate  (Wilson 95% CI)", va="center", ha="center",
             rotation=90, fontsize=10.5)
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
