"""Build the two Stage-1 prompt datasets from the SESGO corpus (writes to data/, NOT out/).

  data/full_prompt_dataset.json  ALL stability prompts: per item, every role->position
      permutation (6) x label style (3) = 18 three-option prompts (target/other/UNKNOWN).
  data/forced_fork.json          Ambiguous + NON-NEGATIVE items only, as a forced
      2-option fork (target vs other, NO UNKNOWN) with order (2) x label style (3) variation.

Each record is self-contained and keyed by a stable `prompt_id` (content hash); the
runner maps its outputs back through `prompt_id`. Cleaned schema vs the old prompt
sample: drops scaffold_id and all 2-option side fields, adds `prompt_id` and a
`response_formatting_instruction`, and stores `text` as a list of lines. There is no
binary-choice metadata: the runner always greedy-decodes and parses the answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

from src.datasets.prompt.sesgo_label_style import get_sesgo_label_styles
from src.datasets.prompt.sesgo_prompt_config import SesgoPromptConfig
from src.datasets.prompt.sesgo_prompt_generator import SesgoPromptDatasetGenerator, _render
from src.datasets.prompt.sesgo_prompt_localization import sesgo_answer_cue, sesgo_choice_prefix
from src.datasets.sesgo import SesgoLabel, load_items

# Forced-fork presents only these two roles, in each order (defeats position bias).
_FORK_ORDERS = ((SesgoLabel.OTHER, SesgoLabel.TARGET), (SesgoLabel.TARGET, SesgoLabel.OTHER))


def _instruction(language: str, option_labels) -> str:
    """The format instruction, built from the SINGLE-SOURCE answer cue (sesgo_answer_cue)
    and the prompt's ACTUAL option letters (a/b/c, 1/2/3, x/y/z) — so the hint never lies
    about which letters are valid, and the parser reads the same cue from `answer_cue`."""
    cue = sesgo_answer_cue(language)
    letters = [m.rstrip(")") for m in option_labels]
    if language == "es":
        opts = ", ".join(letters[:-1]) + " o " + letters[-1]
        return (f"En la última línea escribe EXACTAMENTE «{cue} X)», sustituyendo X por la "
                f"letra de la opción elegida ({opts}).")
    opts = ", ".join(letters[:-1]) + ", or " + letters[-1]
    return (f"On the last line write EXACTLY '{cue} X)', replacing X with the chosen "
            f"option letter ({opts}).")


def _prompt_id(text: str) -> str:
    """Stable, unique id for a rendered prompt (content hash)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _with_instruction(text: str, prefix: str, instruction: str) -> str:
    """End the prompt with the response-format instruction. The trailing answer cue
    (`prefix`, e.g. 'Respuesta: ') is DROPPED — it's redundant now that the model is told
    to write the full «<cue> X)» line itself."""
    body = text[: -len(prefix)] if text.endswith(prefix) else text + "\n"
    return f"{body}{instruction}"


def _record(text, sample_idx, *, question_id, bias_category, question_polarity,
            context_condition, language, label_style, option_labels, position_labels,
            choice_prefix, gold_label, bbq, target_identity, other_identity, instruction):
    """One cleaned, self-contained prompt record (flat, JSON-able)."""
    return {
        "sample_idx": sample_idx,
        "prompt_id": _prompt_id(text),
        "question_id": question_id,
        "bias_category": bias_category,
        "question_polarity": question_polarity,
        "context_condition": context_condition,
        "language": language,
        "label_style": label_style,
        "text": text.split("\n"),
        "option_labels": list(option_labels),
        "position_labels": [r.value for r in position_labels],
        "choice_prefix": choice_prefix,
        "response_formatting_instruction": instruction,
        "answer_cue": sesgo_answer_cue(language),
        "gold_label": gold_label.value if hasattr(gold_label, "value") else gold_label,
        "bbq": bbq,
        "target_identity": target_identity,
        "other_identity": other_identity,
    }


def build_full(items) -> list[dict]:
    """All 18-per-item three-option stability prompts (format variation, no scaffold)."""
    cfg = SesgoPromptConfig(
        name="stability", all_permutations=True,
        label_styles=get_sesgo_label_styles(), include_no_scaffold=True,
    )
    dataset = SesgoPromptDatasetGenerator(cfg).generate(items, [])
    out = []
    for i, s in enumerate(dataset.samples):
        instr = _instruction(s.language, s.option_labels)
        text = _with_instruction(s.text, s.choice_prefix, instr)
        out.append(_record(
            text, i, question_id=s.question_id, bias_category=s.bias_category,
            question_polarity=s.question_polarity, context_condition=s.context_condition,
            language=s.language, label_style=s.label_style, option_labels=s.option_labels,
            position_labels=s.position_labels, choice_prefix=s.choice_prefix,
            gold_label=s.gold_label, bbq=s.bbq, target_identity=s.target_identity,
            other_identity=s.other_identity, instruction=instr,
        ))
    return out


def build_forced_fork(items) -> list[dict]:
    """Ambiguous + non-negative items as forced 2-option (target vs other) forks."""
    styles_2opt = [(a, b) for (a, b, _c) in get_sesgo_label_styles()]
    out, idx = [], 0
    for item in items:
        if item.context_condition != "ambig" or item.polarity == "neg":
            continue  # ambiguous AND non-negative only
        prefix = sesgo_choice_prefix(item.language)
        for roles, markers in product(_FORK_ORDERS, styles_2opt):
            instr = _instruction(item.language, markers)  # letters depend on this fork's markers
            text = _with_instruction(_render(None, item, markers, roles, prefix), prefix, instr)
            out.append(_record(
                text, idx, question_id=item.question_id, bias_category=item.category.value,
                question_polarity=item.polarity, context_condition=item.context_condition,
                language=item.language, label_style="".join(markers), option_labels=markers,
                position_labels=roles, choice_prefix=prefix, gold_label=SesgoLabel.UNKNOWN,
                bbq=item.bbq, target_identity=item.target_text, other_identity=item.other_text,
                instruction=instr,
            ))
            idx += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sesgo-dir", default="datasets/SESGO")
    ap.add_argument("--language", default="es")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--limit", type=int, default=None, help="cap distinct items (debug)")
    args = ap.parse_args()

    items = load_items(args.sesgo_dir, languages=(args.language,), limit=args.limit)
    full, fork = build_full(items), build_forced_fork(items)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("full_prompt_dataset.json", full), ("forced_fork.json", fork)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  wrote {out_dir/name}: {len(data)} prompts")
    print(f"items={len(items)}  full={len(full)} ({len(full)//max(len(items),1)}/item)  fork={len(fork)}")


if __name__ == "__main__":
    main()
