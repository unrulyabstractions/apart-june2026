"""Greedy-readout runner for the cleaned stability / forced-fork datasets.

For each prompt in data/<dataset>.json (mapped by prompt_id) it: greedy-decodes a
FULL, unbounded trajectory (let the CoT finish at natural EOS), parses the answer
into (label, choice), then teacher-forces once to read the model's commitment at
the answer position — `label_logprob` (logprob of the chosen label token) and
`vocab_entropy` (Shannon entropy of the next-token distribution there).

A thinking model and a non-thinking model are SEPARATE runs (--mode), each writing
its own response_samples.json. Resumable: skips prompt_ids already in the output.

Builds on the existing inference stack: ModelRunner (generation + teacher-forced
trajectory), parse_sesgo_answer (label/choice), and src.common.math (entropy).

Usage:
  uv run python experiment/stability/run_greedy_readout.py \
    --model meta-llama/Llama-3.2-1B-Instruct --dataset data/full_prompt_dataset.json \
    --study stability --mode nonthinking --out-dir out [--limit N] [--shard-index i --shard-count K]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common.math import shannon_entropy
from src.inference.answer_parser import answer_segment, parse_answer
from src.inference.backends import ModelBackend
from src.inference.model_runner import ModelRunner

from experiment.stability.greedy_readout_schema import GreedyReadout, GreedyReadoutDataset

# "Unbounded" generation: a high cap that EOS almost always hits first, so the CoT
# finishes naturally (per the spec — do not bound greedy thinking by max tokens).
_UNBOUNDED = 8192
_CHECKPOINT_EVERY = 25


def _bare(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def _is_degenerate(text: str, min_len: int = 40) -> bool:
    """True if the response is a short repetition loop / garbage (a backend silently
    mis-generating, e.g. mlx_lm on an unsupported arch emitting 'ipiipi...'). Such a
    response is NOT data — we flag it instead of letting a 0-entropy or invalid read
    pass silently."""
    t = text.strip()
    if len(t) < min_len:
        return False
    if len(set(t)) <= 3:  # almost no unique characters
        return True
    for period in range(1, 11):  # a short cycle that explains >=85% of the text
        unit = t[:period]
        if unit and t.count(unit) * period >= 0.85 * len(t):
            return True
    # A long-block repetition loop (a CoT stuck re-deriving the same paragraph until the
    # token cap): many non-empty lines but few DISTINCT ones.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 12 and len(set(lines)) < len(lines) * 0.5:
        return True
    return False


def _out_dir(out_root: Path, study: str, model: str, mode: str, idx: int, count: int) -> Path:
    """out/<study>/<model>-<mode>[/shard_i_of_K]/  (post-flatten layout, mode-separated)."""
    d = out_root / study / f"{_bare(model)}-{mode}"
    if count > 1:
        d = d / f"shard_{idx}_of_{count}"
    return d


def _templated_prompt(runner: ModelRunner, prompt: str, thinking: bool) -> str:
    """Chat-templated prompt with thinking explicitly on/off.

    Non-reasoning models have a single mode. For reasoning models, thinking=False
    suppresses the scratch-pad (skip prefix, or enable_thinking=False in the
    template); thinking=True lets it reason — including Qwen3.5, whose template
    forces thinking off by default, so we re-template it with enable_thinking=True.
    """
    if not runner.is_reasoning_model:
        return runner.apply_chat_template(prompt)
    if thinking and runner._disables_thinking_via_template:
        return runner._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
    templated = runner.apply_chat_template(prompt)
    if not thinking:
        templated += runner.skip_thinking_prefix  # no-op for template-disabled families
    return templated


def _readout(runner: ModelRunner, rec: dict, thinking: bool, temperature: float) -> GreedyReadout:
    """One prompt -> one GreedyReadout. Non-thinking decodes greedily (temp 0); thinking
    runs SAMPLE at a low, near-greedy temperature, because greedy decoding traps small
    reasoning models in repetition loops that never terminate."""
    prompt = "\n".join(rec["text"])
    templated = _templated_prompt(runner, prompt, thinking)
    prompt_ids = runner.encode_ids(templated, add_special_tokens=True)

    traj = runner.generate_trajectory(prompt_ids, max_new_tokens=_UNBOUNDED, temperature=temperature)
    full_ids = traj.token_ids
    generated = runner.decode_ids(full_ids[len(prompt_ids):])

    label, choice, off = parse_answer(
        generated, rec["option_labels"], rec["position_labels"], rec.get("answer_cue", ""))
    # Token index of the answer position: the parsed label, else the first
    # post-thinking token (so invalid answers still get a defined readout).
    if off >= 0:
        prefix = generated[:off]
    else:
        seg = answer_segment(generated)
        prefix = generated[: len(generated) - len(seg)]
    pos = len(prompt_ids) + len(runner.encode_ids(prefix, add_special_tokens=False))
    pos = min(max(pos, len(prompt_ids)), len(full_ids) - 1)

    # One teacher-forced pass over prompt+answer; read the distribution that PRODUCED
    # the answer token (full_logits[pos-1]) and that token's own logprob (logprobs[pos]).
    ct = runner.compute_trajectory(full_ids[: pos + 1])
    dist = torch.log_softmax(ct.full_logits[pos - 1], dim=-1)
    # Cut-off CoT: a thinking run that never closed </think> AND committed no answer
    # (it ran past the token budget mid-thought). A short, direct answer with no think
    # block is fine — only flag the case with no parseable commitment, never guess one.
    unclosed_think = thinking and "</think>" not in generated and choice == "invalid"
    degenerate = _is_degenerate(generated) or unclosed_think
    return GreedyReadout(
        sample_idx=rec["sample_idx"], prompt_id=rec["prompt_id"], prompt_text=templated,
        response_text=generated, choice="invalid" if degenerate else choice, label=label,
        label_logprob=float(ct.logprobs[pos]), vocab_entropy=float(shannon_entropy(dist)),
        degenerate=degenerate,
    )


def _load_records(path: Path, limit: int | None, idx: int, count: int) -> list[dict]:
    records = json.load(path.open())
    if limit is not None:
        records = records[:limit]
    if count > 1:  # contiguous shard
        n = len(records)
        lo, hi = n * idx // count, n * (idx + 1) // count
        records = records[lo:hi]
    return records


def _resume(out_file: Path) -> tuple[list[GreedyReadout], set[str]]:
    if not out_file.exists():
        return [], set()
    prior = GreedyReadoutDataset.from_dict(json.load(out_file.open()))
    return list(prior.samples), {s.prompt_id for s in prior.samples}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="data/full_prompt_dataset.json")
    ap.add_argument("--study", default="stability")
    ap.add_argument("--mode", choices=["thinking", "nonthinking"], default="nonthinking")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--backend", choices=["auto", "huggingface", "mlx", "vllm"], default="auto",
                    help="force a backend (e.g. huggingface for archs MLX can't load locally)")
    ap.add_argument("--thinking-temp", type=float, default=0.6,
                    help="sampling temperature for THINKING mode (low/near-greedy; breaks "
                         "repetition loops). Non-thinking always decodes greedily (0.0).")
    args = ap.parse_args()

    records = _load_records(Path(args.dataset), args.limit, args.shard_index, args.shard_count)
    out_dir = _out_dir(Path(args.out_dir), args.study, args.model, args.mode,
                       args.shard_index, args.shard_count)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "response_samples.json"

    samples, done = _resume(out_file)
    print(f"[greedy] {args.model} mode={args.mode} study={args.study} "
          f"prompts={len(records)} resumed={len(done)} -> {out_file}")

    thinking = args.mode == "thinking"
    backend = None if args.backend == "auto" else ModelBackend[args.backend.upper()]
    runner = ModelRunner(args.model, backend=backend)
    ds = GreedyReadoutDataset(study=args.study, model=args.model, mode=args.mode, samples=samples)

    temp = args.thinking_temp if thinking else 0.0
    todo = [r for r in records if r["prompt_id"] not in done]
    for i, rec in enumerate(todo, 1):
        ds.samples.append(_readout(runner, rec, thinking, temp))
        if i % _CHECKPOINT_EVERY == 0 or i == len(todo):
            json.dump(ds.to_dict(), out_file.open("w"), ensure_ascii=False, indent=2)
            print(f"[greedy] {i}/{len(todo)} (+{len(done)} resumed) checkpointed")
    n_degen = sum(1 for s in ds.samples if s.degenerate)
    print(f"[greedy] DONE {args.model} {args.mode}: {len(ds.samples)} samples -> {out_file}")
    if n_degen:
        print(f"[greedy] !! WARNING: {n_degen}/{len(ds.samples)} responses are DEGENERATE "
              f"(repetition/garbage) — this backend mis-generates {args.model}; the data is NOT "
              f"usable. Re-run on a backend that supports this arch (HF/vLLM on CUDA).")


if __name__ == "__main__":
    main()
