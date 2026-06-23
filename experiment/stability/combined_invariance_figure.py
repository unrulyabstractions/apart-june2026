"""ONE format-invariance figure spanning BOTH data generations: the OLD stability study
(`data/hf_old/sesgo/stability/<model>`, 4 models) and the NEW greedy-readout sweep
(`out/stability/<dir>`, 17 models), plotted together as role-invariance rate vs model size.

Role-invariance = fraction of base ITEMS whose committed role is identical across the SAME
3 orthogonal variants (baseline / position-flip / label-swap), so the two generations are
apples-to-apples. The OLD data stores 36 variants/qid (6 orderings x 3 label styles x 2
replicates); we extract exactly the 3 orthogonal combos and pair them by replicate index
within each (qid, ordering, style) combo so each item carries one baseline/flip/swap choice
(verified: e.g. Qwen3-32B 24 items, Llama-3.1-70B 30 items). OLD choices are parser-free
(argmax of `non_thinking.prob` mapped through reconstructed position_labels); NEW choices are
the committed greedy-readout `choice`. OLD = filled SQUARE, NEW = open circle / filled triangle
(thinking), family-coloured, Wilson 95% CI, n annotated.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from experiment.baseline.old_data_parser_free_figures import model_meta
from experiment.baseline.reparse_old_thinking_labels import (
    option_labels_from_style,
    position_labels_from_prompt,
)
from experiment.bias.stability_readout_join import load_metadata
from experiment.common.sweep_models import FAMILY_COLOR, parse_model
from experiment.stability.new_sweep_baseline_stability_figures import _VARIANT, _invariance
from src.common.math import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[2]


def _old_choice(s: dict) -> tuple[str, tuple]:
    """Parser-free committed role + the (role-at-each-position) tuple for an OLD sample."""
    labels = option_labels_from_style(s["label_style"])
    roles = position_labels_from_prompt(s, labels)
    prob = s["non_thinking"]["prob"]
    return roles[max(range(len(prob)), key=lambda i: prob[i])], tuple(roles)


def old_invariance(model_dir: Path) -> dict:
    """Same 3-orthogonal-variant role-invariance as the NEW sweep, on OLD data.

    Item = (question_id, replicate-index): the replicate index disambiguates the two draws
    per (qid, ordering, style) combo so every item gets exactly one baseline/flip/swap choice.
    """
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


def _plot_point(ax, size, succ, n, color, marker, ms):
    p, lo, hi = wilson_interval(succ, n)
    mfc = "white" if marker == "o" else color
    ax.errorbar(size, p, yerr=[[p - lo], [hi - p]], fmt=marker, color=color, mfc=mfc,
                mec=color, ms=ms, capsize=3, mew=1.6, zorder=3)
    ax.annotate(f"{n}", (size, p), textcoords="offset points", xytext=(6, 5), fontsize=6, alpha=0.8)
    return p


def build(new_dir: Path, dataset: Path, old_root: Path, out: Path) -> None:
    meta = load_metadata(dataset)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    rows: list[tuple] = []

    # OLD models: filled squares.
    old_models = sorted(d.name for d in old_root.iterdir()
                        if (d / "response_samples.json").is_file())
    for name in old_models:
        fam, size = model_meta(name)
        iv = old_invariance(old_root / name)
        p = _plot_point(ax, size, iv["inv"], iv["n"], FAMILY_COLOR.get(fam, "gray"), "s", 10)
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
        p = _plot_point(ax, sm.size_b, iv["inv"], iv["n"], FAMILY_COLOR[sm.family], marker, 8)
        rows.append(("NEW", sm.name, sm.family, sm.size_b, iv["n"], iv["inv"], p))

    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("Model size (B params, log scale)")
    ax.set_ylabel("Role-invariance rate (Wilson 95% CI)")
    ax.set_title("Format-invariance rises with scale across BOTH data generations\n"
                 "role identical across the 3 orthogonal variants (baseline / position-flip / label-swap)")

    fams = sorted({r[2] for r in rows}, key=lambda f: list(FAMILY_COLOR).index(f) if f in FAMILY_COLOR else 9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], label=f) for f in fams]
    handles += [
        plt.Line2D([], [], marker="s", ls="", mfc="gray", mec="gray", color="gray",
                   label="OLD (3 orthogonal variants;\nparser-free prob method)"),
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="NEW non-thinking"),
        plt.Line2D([], [], marker="^", ls="", color="k", label="NEW thinking"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)

    print(f"\n{'gen':4s} {'model':18s} {'fam':8s} {'size':>6s} {'n':>5s} {'inv':>5s} {'rate':>6s}")
    for gen, name, fam, size, n, inv, p in sorted(rows, key=lambda r: (r[0], r[3])):
        print(f"{gen:4s} {name:18s} {fam:8s} {size:6g} {n:5d} {inv:5d} {p:6.3f}")
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
