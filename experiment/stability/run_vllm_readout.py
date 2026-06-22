"""vLLM greedy readout — for FP8 / newer checkpoints the HF backend can't run (e.g.
Ministral-3's finegrained-fp8 weights). vLLM loads FP8 natively.

vLLM is generation-only, so the measurement uses its per-token top-k logprobs:
  - label_logprob is EXACT (the chosen answer token's own logprob),
  - vocab_entropy is an APPROXIMATION over the top-k logprobs at the answer position
    (full-vocab entropy isn't exposed). The mode is suffixed '-vllm' so this is never
    silently conflated with the HF-exact entropy.

Same minimal schema + same answer parser as the HF runner; mapped by prompt_id.

Usage (CUDA box only — vLLM is CUDA-only):
  python -m experiment.stability.run_vllm_readout --model mistralai/Ministral-3-3B-Instruct-2512 \
    --mode nonthinking --dataset data/full_prompt_dataset.json --study stability --out-dir out [--limit N]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiment.stability.greedy_readout_schema import GreedyReadout, GreedyReadoutDataset
from src.inference.answer_parser import answer_segment, parse_answer

_UNBOUNDED = 8192
_TOPK = 20


def _entropy_topk(lp_dict) -> float:
    """Shannon entropy (nats) over the available top-k logprobs at one position."""
    lps = [v.logprob for v in lp_dict.values()]
    return float(-sum(math.exp(lp) * lp for lp in lps)) if lps else float("nan")


def _answer_token_index(tok, text: str, char_off: int) -> int:
    """Generated-token index of the answer (parsed char offset, else first post-think)."""
    if char_off < 0:
        seg = answer_segment(text)
        char_off = len(text) - len(seg)
    return len(tok(text[:char_off], add_special_tokens=False).input_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="data/full_prompt_dataset.json")
    ap.add_argument("--study", default="stability")
    ap.add_argument("--mode", default="nonthinking")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = json.load(open(args.dataset))
    if args.limit is not None:
        records = records[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="auto", trust_remote_code=True, gpu_memory_utilization=0.9)
    sp = SamplingParams(temperature=0.0, max_tokens=_UNBOUNDED, logprobs=_TOPK)

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": "\n".join(r["text"])}],
            tokenize=False, add_generation_prompt=True,
        )
        for r in records
    ]
    outs = llm.generate(prompts, sp)

    ds = GreedyReadoutDataset(study=args.study, model=args.model, mode=f"{args.mode}-vllm")
    for rec, prompt, out in zip(records, prompts, outs):
        o = out.outputs[0]
        text = o.text
        label, choice, off = parse_answer(
            text, rec["option_labels"], rec["position_labels"], rec.get("answer_cue", ""))
        logprobs = o.logprobs or []
        ti = min(max(_answer_token_index(tok, text, off), 0), max(len(logprobs) - 1, 0))
        lp_dict = logprobs[ti] if logprobs else {}
        chosen = o.token_ids[ti] if ti < len(o.token_ids) else None
        label_lp = lp_dict[chosen].logprob if (chosen is not None and chosen in lp_dict) else float("nan")
        ds.samples.append(GreedyReadout(
            sample_idx=rec["sample_idx"], prompt_id=rec["prompt_id"], prompt_text=prompt,
            response_text=text, choice=choice, label=label,
            label_logprob=float(label_lp), vocab_entropy=_entropy_topk(lp_dict),
        ))

    out_dir = Path(args.out_dir) / args.study / f"{args.model.rstrip('/').split('/')[-1]}-{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(ds.to_dict(), (out_dir / "response_samples.json").open("w"), ensure_ascii=False, indent=2)
    print(f"[vllm] DONE {args.model} {args.mode}: {len(ds.samples)} samples -> {out_dir}")


if __name__ == "__main__":
    main()
