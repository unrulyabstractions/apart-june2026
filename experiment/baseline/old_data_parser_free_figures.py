"""OLD-data figures using ONLY the parser-free non_thinking probability method.

The buggy text parser (greedy_thinking.label) is NEVER used here: every choice comes from
`argmax(non_thinking['prob'])` mapped through the reconstructed `position_labels` (target /
other / unknown at each option position), reusing the exact reconstruction helper in
`reparse_old_thinking_labels.py`.

Fig1: baseline accuracy/abstention vs model size (ambig | disambig panels), Wilson 95% CI.
Fig2: stability format-invariance rate per model (choice identical across all format variants
of a question_id). Family/size/colour come from `sweep_models.parse_model`, with a small local
size map for OLD checkpoints whose names predate the current parser.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from experiment.baseline.reparse_old_thinking_labels import old_nonthinking_role
from experiment.common.sweep_models import FAMILY_COLOR, parse_model
from src.common.file_io import load_json
from src.common.logging import log, log_header
from src.common.math import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD = REPO_ROOT / "data" / "hf_old" / "sesgo"
FIGS = REPO_ROOT / "paper" / "figures"

# Sizes (billions) + family for OLD checkpoints whose names predate sweep_models.parse_model.
OLD_SIZE = {
    "Qwen3-0.6B": ("Qwen", 0.6), "Qwen3-1.7B": ("Qwen", 1.7), "Qwen3-4B": ("Qwen", 4.0),
    "Qwen3-14B": ("Qwen", 14.0), "Qwen3-32B": ("Qwen", 32.0),
    "gemma-2-2b-it": ("Gemma", 2.0), "gemma-2-9b-it": ("Gemma", 9.0),
    "Mistral-7B-Instruct-v0.3": ("Mistral", 7.0),
    "Mistral-Small-24B-Instruct-2501": ("Mistral", 24.0),
}


def model_meta(name: str) -> tuple[str, float]:
    """(family, size_b) — parse_model for Llama, local map for older checkpoints."""
    if name in OLD_SIZE:
        return OLD_SIZE[name]
    m = parse_model(name)
    if m is None:
        raise ValueError(f"unknown model size: {name}")
    return m.family, m.size_b


def choice_of(sample: dict) -> str:
    """Parser-free choice: the OLD non_thinking committed role. The OLD `non_thinking.prob`
    is stored in ROLE order [target, other, unknown], so argmax indexes the role directly
    (see `old_nonthinking_role`). Mapping through option positions would wrongly swap
    target<->other and collapsed old disambiguated accuracy to ~0."""
    return old_nonthinking_role(sample)


def baseline_counts(name: str) -> dict[str, tuple[int, int]]:
    """Per-model (successes,total) for ambig accuracy, disambig accuracy, overall abstention."""
    samples = load_json(OLD / "baseline" / name / "response_samples.json")["samples"]
    acc = {"ambig": [0, 0], "disambig": [0, 0]}
    abst = [0, 0]
    for s in samples:
        c = choice_of(s)
        cond = s["context_condition"]
        ok = c == "unknown" if cond == "ambig" else c == s["gold_label"]
        acc[cond][0] += ok
        acc[cond][1] += 1
        abst[0] += c == "unknown"
        abst[1] += 1
    return {"ambig": tuple(acc["ambig"]), "disambig": tuple(acc["disambig"]), "abstain": tuple(abst)}


def invariance_rate(name: str) -> tuple[int, int, int]:
    """(role-invariant qids, position-invariant qids, total qids) across all format variants.

    role-invariant: the role chosen (target/other/unknown) is identical across all 36 variants.
    position-invariant: the chosen OPTION POSITION (argmax index) is identical across all variants
    — exposing whether the model answers by content (role) or by slot (position).
    """
    samples = load_json(OLD / "stability" / name / "response_samples.json")["samples"]
    roles: dict = defaultdict(set)
    positions: dict = defaultdict(set)
    for s in samples:
        prob = s["non_thinking"]["prob"]
        roles[s["question_id"]].add(choice_of(s))
        positions[s["question_id"]].add(max(range(len(prob)), key=lambda i: prob[i]))
    role_inv = sum(len(v) == 1 for v in roles.values())
    pos_inv = sum(len(v) == 1 for v in positions.values())
    return role_inv, pos_inv, len(roles)


def fig1(models: list[str], out: Path) -> None:
    metas = {m: model_meta(m) for m in models}
    counts = {m: baseline_counts(m) for m in models}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, panel, title in zip(axes, ("ambig", "disambig"), ("Ambiguous (gold=unknown)", "Disambiguated")):
        for m in models:
            fam, size = metas[m]
            succ, tot = counts[m][panel]
            p, lo, hi = wilson_interval(succ, tot)
            ax.errorbar(size, p, yerr=[[p - lo], [hi - p]], fmt="o", ms=7,
                        color=FAMILY_COLOR.get(fam, "gray"), capsize=3,
                        label=fam if fam not in ax.get_legend_handles_labels()[1] else None)
        ax.set_xscale("log"); ax.set_xlabel("Model size (B params, log)")
        ax.set_title(title); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy (parser-free non_thinking)")
    axes[1].legend(title="Family", fontsize=8)
    fig.suptitle("OLD baseline accuracy vs model size — parser-free probability method")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return counts, metas


def fig2(models: list[str], out: Path) -> dict:
    rates = {m: invariance_rate(m) for m in models}
    order = sorted(models, key=lambda m: model_meta(m)[1])
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.38
    for i, m in enumerate(order):
        fam, _ = model_meta(m)
        role_inv, pos_inv, tot = rates[m]
        ax.bar(i - w / 2, role_inv / tot, w, color=FAMILY_COLOR.get(fam, "gray"),
               label="role-invariant" if i == 0 else None)
        ax.bar(i + w / 2, pos_inv / tot, w, color=FAMILY_COLOR.get(fam, "gray"), alpha=0.45, hatch="//",
               label="position-invariant" if i == 0 else None)
        ax.text(i, 1.0, f"n={tot}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{m}\n({model_meta(m)[1]:g}B)" for m in order], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Format-invariance rate (parser-free)"); ax.set_ylim(0, 1.08); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("OLD stability: invariance across 36 format variants per question\n"
                 "(role-invariance ~0: choices follow the OPTION SLOT, not the content)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=FIGS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = sorted(d.name for d in (OLD / "baseline").iterdir()
                  if (d / "response_samples.json").is_file())
    stab = sorted(d.name for d in (OLD / "stability").iterdir()
                  if (d / "response_samples.json").is_file())

    p1 = args.out_dir / "old_baseline_accuracy_vs_size.png"
    counts, metas = fig1(base, p1)
    log_header("FIG1: baseline accuracy/abstention (parser-free non_thinking)")
    log(f"{'model':32s} {'fam':8s} {'size':>5s} {'ambig_acc':>10s} {'disamb_acc':>11s} {'abstain':>9s}")
    for m in sorted(base, key=lambda x: metas[x][1]):
        a, d, ab = counts[m]["ambig"], counts[m]["disambig"], counts[m]["abstain"]
        log(f"{m:32s} {metas[m][0]:8s} {metas[m][1]:5g} "
            f"{a[0]/a[1]:9.3f} {d[0]/d[1]:10.3f} {ab[0]/ab[1]:8.3f}")

    p2 = args.out_dir / "old_stability_format_invariance.png"
    rates = fig2(stab, p2)
    log_header("FIG2: stability variant-count + format-invariance")
    for m in stab:
        samples = load_json(OLD / "stability" / m / "response_samples.json")["samples"]
        dist = Counter(Counter(s["question_id"] for s in samples).values())
        role_inv, pos_inv, tot = rates[m]
        log(f"{m:28s} variants_per_qid={dict(dist)} "
            f"role_inv={role_inv}/{tot}={role_inv/tot:.3f} pos_inv={pos_inv}/{tot}={pos_inv/tot:.3f}")
    log(f"\nWrote {p1}\nWrote {p2}")


if __name__ == "__main__":
    main()
