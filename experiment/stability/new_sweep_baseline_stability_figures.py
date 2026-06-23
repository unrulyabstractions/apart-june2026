"""Two Stage-1 figures from the NEW greedy-readout stability sweep (`out/stability/`):
  1. new_baseline_accuracy_vs_size.png  — baseline accuracy vs model size (ambig | disambig)
     using the count-based convention of `bias_segments` (ambig acc = fraction 'unknown',
     disambig acc = fraction == gold), one point/model-slice, Wilson 95% CIs.
  2. new_stability_format_invariance.png — fraction of base items whose committed ROLE is
     identical across all 3 orthogonal variants (baseline / position-flip / label-swap),
     requiring all 3 present per item; thin order- and label-sensitivity series.

Reuses: stability_readout_join.enrich, bias_segments._accuracy_count, sweep_models.parse_model,
confidence_intervals.wilson_interval. A base ITEM groups its 3 variants by the dataset's
question_id + context + polarity + bias_category + identities + language (verified 2310x3=6930).
Variant TYPE is read from (position_labels, label_style): baseline=(t,o,u)+a)b)c),
flip=(o,t,u)+a)b)c), swap=(t,o,u)+x)y)z).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment.bias.bias_segments import _accuracy_count
from experiment.bias.stability_readout_join import enrich, load_metadata
from experiment.common.sweep_models import FAMILY_COLOR, parse_model
from src.common.math import wilson_interval

# Variant type from (position_labels tuple, label_style); see build_stability_datasets._SELECTED_VARIANTS.
_VARIANT = {
    (("target", "other", "unknown"), "a)b)c)"): "baseline",
    (("other", "target", "unknown"), "a)b)c)"): "flip",
    (("target", "other", "unknown"), "x)y)z)"): "swap",
}


def _item_key(m: dict) -> tuple:
    """Group a record's 3 orthogonal variants back to their shared base item."""
    return (m["question_id"], m["context_condition"], m["question_polarity"],
            m["bias_category"], m["target_identity"], m["other_identity"], m["language"])


def _load_slice(d: Path) -> list[dict]:
    return json.load((d / "response_samples.json").open())["samples"]


def _baseline_rows(samples: list[dict], group_key: str, meta: dict):
    """Enriched responses for BASELINE-variant prompts only (the reference accuracy panel)."""
    base_ids = {pid for pid, m in meta.items()
                if _VARIANT.get((tuple(m["position_labels"]), m["label_style"])) == "baseline"}
    return enrich([s for s in samples if s["prompt_id"] in base_ids], group_key, meta)


def _invariance(samples: list[dict], meta: dict) -> dict:
    """Per base item present in full (all 3 variants), the role chosen under each variant.
    Returns counts for full-invariance, order-agreement, label-agreement, and n_items."""
    choice = {s["prompt_id"]: s["choice"] for s in samples}
    items: dict[tuple, dict] = {}
    for pid, m in meta.items():
        v = _VARIANT.get((tuple(m["position_labels"]), m["label_style"]))
        if v is None or pid not in choice:
            continue
        items.setdefault(_item_key(m), {})[v] = choice[pid]
    full = [d for d in items.values() if {"baseline", "flip", "swap"} <= d.keys()]
    inv = sum(len(set(d.values())) == 1 for d in full)
    order = sum(d["baseline"] == d["flip"] for d in full)
    label = sum(d["baseline"] == d["swap"] for d in full)
    return {"n": len(full), "inv": inv, "order": order, "label": label}


def _err(succ: int, n: int) -> tuple[float, list[list[float]]]:
    p, lo, hi = wilson_interval(succ, n)
    return p, [[p - lo], [hi - p]]


def _scatter(ax, x, succ, n, color, marker, label=None, alpha=1.0, ms=8):
    p, yerr = _err(succ, n)
    ax.errorbar(x, p, yerr=yerr, fmt=marker, color=color, mfc=(color if marker != "o" else "white"),
                mec=color, ms=ms, capsize=3, alpha=alpha, label=label, mew=1.5, zorder=3)
    ax.annotate(f"{n}", (x, p), textcoords="offset points", xytext=(6, 6), fontsize=6, alpha=0.8)


def build(stability_dir: Path, dataset: Path, out_dir: Path) -> None:
    meta = load_metadata(dataset)
    slices = sorted(p.name for p in stability_dir.iterdir() if (p / "response_samples.json").exists())

    fig1, ax1 = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    acc_tbl, inv_tbl = [], []
    for name in slices:
        sm = parse_model(name)
        if sm is None:
            print(f"  SKIP unparsed: {name}")
            continue
        color = FAMILY_COLOR[sm.family]
        marker = "^" if sm.mode == "thinking" else "o"  # thinking = filled triangle, else open circle
        samples = _load_slice(stability_dir / name)
        rows = _baseline_rows(samples, name, meta)
        for j, panel in enumerate(("ambig", "disambig")):
            succ, n = _accuracy_count([r for r in rows if r.context == panel], panel)
            if n:
                _scatter(ax1[j], sm.size_b, succ, n, color, marker)
                acc_tbl.append((sm.name, panel, succ, n, succ / n))
        iv = _invariance(samples, meta)
        if iv["n"]:
            _scatter(ax2, sm.size_b, iv["inv"], iv["n"], color, marker, ms=9)
            _scatter(ax2, sm.size_b, iv["order"], iv["n"], color, marker, alpha=0.3, ms=5)
            _scatter(ax2, sm.size_b, iv["label"], iv["n"], color, marker, alpha=0.3, ms=5)
            inv_tbl.append((sm.name, iv["n"], iv["inv"] / iv["n"], iv["order"] / iv["n"], iv["label"] / iv["n"]))

    for j, panel in enumerate(("Ambiguous (acc = abstain rate)", "Disambiguated (acc = gold)")):
        ax1[j].set_xscale("log"); ax1[j].set_title(panel); ax1[j].set_xlabel("Model size (B params, log)")
        ax1[j].grid(True, which="both", alpha=0.25); ax1[j].set_ylim(-0.02, 1.02)
    ax1[0].set_ylabel("Accuracy")
    fams = {parse_model(n).family for n in slices if parse_model(n)}
    handles = [plt.Line2D([], [], marker="s", ls="", color=FAMILY_COLOR[f], label=f) for f in sorted(fams)]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="non-thinking"),
                plt.Line2D([], [], marker="^", ls="", color="k", label="thinking")]
    ax1[1].legend(handles=handles, fontsize=8, loc="best")
    fig1.suptitle("NEW sweep: baseline accuracy vs size (greedy READOUT choice, not the old probability method)", fontsize=12)
    fig1.tight_layout()

    ax2.set_xscale("log"); ax2.set_ylim(-0.02, 1.02); ax2.grid(True, which="both", alpha=0.25)
    ax2.set_xlabel("Model size (B params, log)"); ax2.set_ylabel("Agreement rate (Wilson 95% CI)")
    ax2.set_title("NEW sweep: format-invariance — role identical across all 3 orthogonal variants\n"
                  "(bold = full invariance; faint = order-only & label-only agreement)")
    h2 = handles + [plt.Line2D([], [], marker="o", ls="", color="grey", alpha=0.3, label="order / label only")]
    ax2.legend(handles=h2, fontsize=8, loc="best")
    fig2.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig1.savefig(out_dir / "new_baseline_accuracy_vs_size.png", dpi=150)
    fig2.savefig(out_dir / "new_stability_format_invariance.png", dpi=150)
    print("\nACCURACY (model, panel, succ, n, acc):")
    for r in acc_tbl:
        print(f"  {r[0]:<16} {r[1]:<9} {r[2]:>5}/{r[3]:<5} {r[4]:.3f}")
    print("\nINVARIANCE (model, n_items, full_inv, order_agree, label_agree):")
    for r in inv_tbl:
        print(f"  {r[0]:<16} n={r[1]:<5} inv={r[2]:.3f} order={r[3]:.3f} label={r[4]:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stability-dir", default="out/stability", type=Path)
    ap.add_argument("--dataset", default="data/full_prompt_dataset.json", type=Path)
    ap.add_argument("--out-dir", default="paper/figures", type=Path)
    a = ap.parse_args()
    build(a.stability_dir, a.dataset, a.out_dir)


if __name__ == "__main__":
    main()
