"""Re-parse OLD hf_old/sesgo response samples with the CURRENT (fixed) answer parser.

The OLD `data/hf_old/sesgo/{baseline,stability}/<model>/response_samples.json` were
produced with a BUGGY answer parser: it text-parsed `greedy_thinking.label` from the
generated reasoning even when the reasoning was truncated mid-thought (no `</think>`,
no answer cue) — fabricating target/other/unknown labels that the model never actually
committed to. The fixed parser (`src.inference.answer_parser.parse_answer`) only commits
a label when the answer segment after the reasoning block carries the localized answer
cue (or is a terse direct answer), and returns `invalid` otherwise.

This script re-parses every `greedy_thinking` sample with the fixed parser, quantifies how
many choices changed (the bug's impact), and writes corrected records in a schema readable
by the new readout plotters (`GreedyReadout`).

Only the PROBABILITY method (`non_thinking.prob`, argmax over the 3 options) is unaffected
by the parser bug; for non-thinking-only samples (the stability study) it supplies the
choice, mapped through the reconstructed position labels.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from src.common.file_io import load_json, save_json
from src.common.logging import log, log_header
from src.common.math import q_diversity
from src.datasets.prompt.sesgo_prompt_localization import (
    sesgo_answer_cue,
    sesgo_markers,
)
from src.inference.answer_parser import parse_answer

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = REPO_ROOT / "data" / "hf_old" / "sesgo"
OUT_ROOT = REPO_ROOT / "out" / "old_reparsed"
STUDIES = ("baseline", "stability")


def option_labels_from_style(label_style: str) -> list[str]:
    """`'a)b)c)'` -> `['a)','b)','c)']` (three two-char option markers)."""
    return [label_style[i : i + 2] for i in range(0, 6, 2)]


def position_labels_from_prompt(sample: dict, option_labels: list[str]) -> list[str]:
    """The role (target/other/unknown) at each of the 3 option positions, IN ORDER.

    Each option line under the `##opciones` marker is matched against the sample's
    `target_identity` / `other_identity`; anything else (the 'No hay suficiente
    informacion' line) is `unknown`.
    """
    _, _, options_marker = sesgo_markers(sample["language"])
    options_segment = sample["prompt_text"].split(options_marker, 1)[1]
    lines = [ln.strip() for ln in options_segment.splitlines() if ln.strip()]
    target, other = sample["target_identity"], sample["other_identity"]
    roles: list[str] = []
    for marker in option_labels:
        line = next((ln for ln in lines if ln.startswith(marker)), "")
        text = line[len(marker) :].strip()
        if target and target in text:
            roles.append("target")
        elif other and other in text:
            roles.append("other")
        else:
            roles.append("unknown")
    return roles


def prob_readout(prob: list[float]) -> tuple[int, float, float]:
    """(argmax index, max prob, effective #choices = exp(Shannon entropy)) of a 3-vector."""
    argmax = max(range(len(prob)), key=lambda i: prob[i])
    logprobs = [math.log(max(p, 1e-12)) for p in prob]
    return argmax, max(prob), float(q_diversity(logprobs, 1.0))


def reparse_sample(sample: dict, answer_cue: str, change_counter: Counter) -> dict:
    """Build one corrected record; tally the old->new thinking choice transition."""
    option_labels = option_labels_from_style(sample["label_style"])
    position_labels = position_labels_from_prompt(sample, option_labels)
    prob = sample["non_thinking"]["prob"]
    argmax, label_prob, vocab_diversity = prob_readout(prob)

    greedy_thinking = sample.get("greedy_thinking")
    if greedy_thinking:
        old_choice = greedy_thinking.get("label") or "invalid"
        response_text = "".join(greedy_thinking.get("text") or [])
        new_label, new_choice, _ = parse_answer(
            response_text, option_labels, position_labels, answer_cue
        )
        change_counter[(old_choice, new_choice)] += 1
        choice, label = new_choice, new_label
    else:
        # Non-thinking-only sample (stability study): probability method supplies the
        # choice — unaffected by the parser bug. No reasoning text was generated.
        response_text = ""
        choice = position_labels[argmax]
        label = option_labels[argmax]

    return {
        "prompt_id": f"{sample['question_id']}_{sample['sample_idx']}",
        "sample_idx": sample["sample_idx"],
        "question_id": sample["question_id"],
        "question_polarity": sample["question_polarity"],
        "bias_category": sample["bias_category"],
        "context_condition": sample["context_condition"],
        "gold_label": sample["gold_label"],
        "label_style": sample["label_style"],
        "target_identity": sample["target_identity"],
        "other_identity": sample["other_identity"],
        "response_text": response_text,
        "choice": choice,
        "label": label,
        "label_prob": label_prob,
        "vocab_diversity": vocab_diversity,
        "degenerate": False,
    }


def reparse_model(study: str, model_dir: Path) -> tuple[dict, Counter]:
    """Re-parse one model's response_samples.json; return (dataset, change_counter)."""
    model = model_dir.name
    samples = load_json(model_dir / "response_samples.json")["samples"]
    change_counter: Counter = Counter()
    corrected = [
        reparse_sample(s, sesgo_answer_cue(s["language"]), change_counter)
        for s in samples
    ]
    return {"study": study, "model": model, "samples": corrected}, change_counter


def print_impact_table(per_model_changes: dict[str, Counter]) -> Counter:
    """Print per-model and overall change tables; return the overall change counter."""
    overall: Counter = Counter()
    log_header("BUG IMPACT: greedy_thinking choice (OLD buggy -> NEW fixed)")
    for key, counter in per_model_changes.items():
        total = sum(counter.values())
        changed = sum(c for (o, n), c in counter.items() if o != n)
        if total == 0:
            log(f"{key:42s} no thinking samples (non-thinking-only)")
            continue
        log(f"{key:42s} thinking_samples={total:5d}  changed={changed:5d} ({changed / total:6.1%})")
        for (old, new), c in sorted(counter.items(), key=lambda kv: -kv[1]):
            if old != new:
                log(f"    {old:>8s} -> {new:<8s} : {c:5d}")
        overall += counter

    log_header("OVERALL (all thinking samples across all models/studies)")
    total = sum(overall.values())
    changed = sum(c for (o, n), c in overall.items() if o != n)
    log(f"total thinking samples : {total}")
    log(f"choice changed         : {changed} ({changed / total:.1%})" if total else "no thinking samples")
    log("transition breakdown (old -> new):")
    for (old, new), c in sorted(overall.items(), key=lambda kv: -kv[1]):
        tag = "" if old == new else "   <-- CHANGED"
        log(f"    {old:>8s} -> {new:<8s} : {c:6d}{tag}")
    return overall


def main() -> None:
    per_model_changes: dict[str, Counter] = {}
    written: list[tuple[str, int]] = []

    for study in STUDIES:
        study_root = OLD_ROOT / study
        for model_dir in sorted(study_root.iterdir()):
            if not (model_dir / "response_samples.json").is_file():
                continue  # skip non-model dirs (e.g. cross_model) and loose files
            dataset, change_counter = reparse_model(study, model_dir)
            per_model_changes[f"{study}/{model_dir.name}"] = change_counter

            out_path = OUT_ROOT / study / model_dir.name / "response_samples.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(dataset, out_path)
            written.append((str(out_path), len(dataset["samples"])))

    print_impact_table(per_model_changes)

    log_header("WRITTEN FILES")
    for path, n in written:
        log(f"  {n:5d} samples -> {path}")
    log(f"\n{len(written)} files written under {OUT_ROOT}")


if __name__ == "__main__":
    main()
