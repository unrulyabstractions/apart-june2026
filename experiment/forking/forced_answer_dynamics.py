"""Per-token-position FORCED-ANSWER dynamics for one model's saved forking base path.

At every base-path token position t we force-close the reasoning ("</think>" + the answer
cue) and read what the model would answer IMMEDIATELY, as the normalized next-token
distribution over the three option labels (target / other / unknown) — the SESGO
non-thinking probability readout, reconstructed (the old TernaryChoiceRunner.choose3 was
removed) from the answer-position logits via TernaryChoice + from_ternary. We also record
the full next-token VOCAB entropy at that position.

Combined downstream with the sampled outcome distribution O_t (forking_trajectory.json),
this lets us compare, per position: KL(O_t || forced), the forced answer itself, and vocab
entropy. Output: <model_dir>/forced_answer_dynamics.json (one row per position).

  uv run python -m experiment.forking.forced_answer_dynamics --model Qwen/Qwen3.5-0.8B \
      --forking-dir out/forking [--thinking] [--validate]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from src.common import BaseSchema
from src.common.math import shannon_entropy
from src.datasets.prompt import SesgoPromptSample
from src.datasets.sesgo_eval.sesgo_non_thinking import SesgoNonThinking, _ROLE_ORDER
from src.datasets.sesgo import SesgoLabel
from src.common.choice.ternary_choice import TernaryChoice
from src.inference import ModelRunner
from src.inference.backends import ModelBackend
from src.inference.answer_parser import answer_segment, parse_answer
from src.datasets.prompt.sesgo_prompt_localization import sesgo_answer_cue

_LABEL = {"target": SesgoLabel.TARGET, "other": SesgoLabel.OTHER, "unknown": SesgoLabel.UNKNOWN}


@dataclass
class ForcedAnswerRow(BaseSchema):
    """Per-position forced-answer readout (role order = target, other, unknown)."""

    position: int
    forced_prob: list[float]   # 3-way [target, other, unknown] normalized next-token dist
    forced_choice: str         # argmax role of forced_prob (or 'unknown' on tie)
    label_prob: float          # prob of the model's own committed label token here
    vocab_entropy: float       # Shannon entropy (nats) of the full next-token distribution


def _option_token_ids(runner: ModelRunner, option_labels: list[str]) -> list[int]:
    """First-token id of each option label (the token that distinguishes a) / b) / c))."""
    return [runner.encode_ids(lbl, add_special_tokens=False)[0] for lbl in option_labels]


def forced_answer_dynamics(runner: ModelRunner, sample: SesgoPromptSample,
                           templated_prompt: str, base_path_text: str) -> list[ForcedAnswerRow]:
    """At every base-path position: force-close, find the answer token, and read the 3-way
    option-label distribution there (from_ternary -> role order) + the vocab entropy."""
    close = "\n" + (runner.reasoning_close_marker or "</think>") + "\n\n"
    close_ids = runner.encode_ids(close, add_special_tokens=False)
    prefill_ids = runner.encode_ids(templated_prompt, add_special_tokens=True)
    base_ids = runner.encode_ids(base_path_text, add_special_tokens=False)
    cue = sesgo_answer_cue(sample.language)
    roles = [r.value if hasattr(r, "value") else r for r in sample.position_labels]
    pos_labels = tuple(_LABEL[r] for r in roles)
    opt_ids = _option_token_ids(runner, sample.option_labels)

    prompts = [runner.decode_ids(list(prefill_ids) + list(base_ids[:t]) + list(close_ids))
               for t in range(len(base_ids))]
    answers = runner.continue_from_text_batch(prompts, max_new_tokens=48, temperature=0.0)

    rows: list[ForcedAnswerRow] = []
    for t, ans in enumerate(answers):
        label, choice, off = parse_answer(ans, sample.option_labels, roles, cue)
        pids = runner.encode_ids(list_to_text := prompts[t], add_special_tokens=True)
        ans_ids = runner.encode_ids(ans, add_special_tokens=False)
        full_ids = list(pids) + list(ans_ids)
        prefix = ans[:off] if off >= 0 else ans[: len(ans) - len(answer_segment(ans))]
        pos = len(pids) + len(runner.encode_ids(prefix, add_special_tokens=False))
        pos = min(max(pos, len(pids)), len(full_ids) - 1)
        ct = runner.compute_trajectory(full_ids[: pos + 1])
        logrow = torch.log_softmax(ct.full_logits[pos].float(), dim=-1)
        lp = [float(logrow[i]) for i in opt_ids]
        lg = [float(ct.full_logits[pos][i]) for i in opt_ids]
        tern = TernaryChoice(labels=tuple(sample.option_labels), logprobs=tuple(lp), logits=tuple(lg))
        ntc = SesgoNonThinking.from_ternary(tern, pos_labels)
        rows.append(ForcedAnswerRow(
            position=t, forced_prob=list(ntc.prob), forced_choice=ntc.predicted.value,
            label_prob=math.exp(float(ct.logprobs[pos])),
            vocab_entropy=float(shannon_entropy(logrow.tolist())),
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--forking-dir", type=Path, default=Path("out/forking"))
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--validate", action="store_true", help="print readout sanity vs the greedy choice")
    a = ap.parse_args()
    bare = a.model.rstrip("/").split("/")[-1]
    mdir = a.forking_dir / bare
    traj = json.load((mdir / "forking_trajectory.json").open())
    sample = SesgoPromptSample.from_dict(json.load((mdir / "selected_item.json").open())["sample"])
    runner = ModelRunner(model_name=a.model, backend=ModelBackend.HUGGINGFACE)
    runner.force_thinking = a.thinking
    rows = forced_answer_dynamics(runner, sample, traj["prompt_text"], traj["base_path_text"])
    out = mdir / "forced_answer_dynamics.json"
    out.write_text(json.dumps([r.to_dict() for r in rows], indent=2))
    print(f"[forced] wrote {out}  ({len(rows)} positions)")
    if a.validate:
        last = rows[-1]
        print(f"[validate] final forced_prob(t,o,u)={[round(x,3) for x in last.forced_prob]} "
              f"choice={last.forced_choice} label_prob={last.label_prob:.3f} "
              f"vocab_entropy={last.vocab_entropy:.3f}")
        print(f"[validate] committed greedy answer from trajectory final hist: "
              f"{[round(x,2) for x in traj['final_histogram']]} labels={traj['outcome_set']['labels']}")


if __name__ == "__main__":
    main()
