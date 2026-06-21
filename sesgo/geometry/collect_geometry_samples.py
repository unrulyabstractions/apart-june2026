"""Capture residual-stream geometry while answering SESGO prompts.

Run-by-path driver for the SESGO GEOMETRY half. For every prompt it (1) runs the
normal SesgoQuerier readout (non-thinking 3-way + thinking draws) and (2) follows
the model's GREEDY NON-THINKING answer path (greedy decode past an empty
<think></think> block) and snapshots the FULL per-layer residual stream
([n_layers, d_model]) at four structural token positions:

    turn         - the last <|im_start|> (the assistant turn boundary)
    think_open   - the <think> token of the skip-thinking prefix
    think_close  - the </think> token
    answer       - the first greedily-generated answer token

Each snapshot is torch.save'd under out/sesgo/geometry/<MODEL>/activations/ and
referenced (by relative path only) from a GeometrySample, so the samples.json
stays small. The headline downstream question is geometric: how far does each
scaffold move these representations versus the no-scaffold baseline.

Usage:
  uv run python sesgo/geometry/collect_geometry_samples.py
  uv run python sesgo/geometry/collect_geometry_samples.py PROMPTS.json \
      --model Qwen/Qwen3-0.6B --n-thinking 4 --subsample 0.5 --layers 0,6,12
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections import Counter
from pathlib import Path

import torch

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/geometry/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import load_json  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.prompt import (  # noqa: E402
    SesgoPromptConfig,
    SesgoPromptDataset,
    SesgoPromptSample,
)
from src.datasets.sesgo_eval import (  # noqa: E402
    GeometryActivation,
    GeometryDataset,
    GeometrySample,
    SesgoQuerier,
    SesgoQueryConfig,
)
from src.ternary_choice import TernaryChoiceRunner  # noqa: E402
from src.inference.backends import ModelBackend  # noqa: E402

# Structural positions we look for, in capture order. think_* exist only for
# reasoning models (skip-thinking prefix); missing ones are logged and skipped.
_POSITION_TYPES = ("turn", "think_open", "think_close", "answer")

# Tokens to greedily decode for the non-thinking answer path. We only need the
# answer token (the first generated token); a few extra give a stable sequence.
_GREEDY_TOKENS = 24


def load_prompt_dataset(path: Path, subsample: float) -> SesgoPromptDataset:
    """Load prompts, striding the RAW json before deserializing (fast path).

    Mirrors the stability collect: when subsample < 1 we json-load once and take
    an evenly-spaced stride over the raw sample dicts (so the slice still spans
    every scaffold/permutation block) and build only the kept samples.
    """
    if subsample >= 1.0:
        return SesgoPromptDataset.from_json(path)
    data = load_json(Path(path))
    raw = data["samples"]
    n = max(1, math.ceil(len(raw) * subsample))
    stride = max(1, len(raw) // n)
    kept = [SesgoPromptSample.from_dict(d) for d in raw[::stride][:n]]
    return SesgoPromptDataset(
        dataset_id=data["dataset_id"],
        config=SesgoPromptConfig.from_dict(data["config"]),
        scaffold_ids=data.get("scaffold_ids", []),
        samples=kept,
    )


def _single_token_id(runner: TernaryChoiceRunner, text: str) -> int | None:
    """Token id for ``text`` iff it encodes to exactly one (special) token.

    Special markers like <|im_start|> / <think> are single tokens in the Qwen
    vocab; we encode WITHOUT special tokens so we read the literal id and can
    search for it in the forced sequence. Returns None when it isn't single.
    """
    ids = runner.encode_ids(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def _resid_filter(layers: list[int] | None):
    """names_filter selecting per-layer residual-stream (resid_post) hooks.

    resid_post is the post-block residual stream — the canonical "the model's
    state after layer L" vector. Optionally restrict to a subset of layers.
    """

    def keep(name: str) -> bool:
        if "resid_post" not in name:
            return False
        if layers is None:
            return True
        return any(name == f"blocks.{layer}.hook_resid_post" for layer in layers)

    return keep


def _stack_resid(cache: dict, layers: list[int], pos: int) -> torch.Tensor:
    """Stack the residual vector at token ``pos`` across layers -> [n_layers, d_model]."""
    vecs = []
    for layer in layers:
        # cache tensors are [batch, seq, d_model]; batch is always 1 here.
        act = cache[f"blocks.{layer}.hook_resid_post"]
        vecs.append(act[0, pos].detach().float().cpu())
    return torch.stack(vecs, dim=0)


def _last_index(ids: list[int], target: int | None) -> int | None:
    """Index of the LAST occurrence of ``target`` in ``ids`` (None if absent)."""
    if target is None:
        return None
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == target:
            return i
    return None


def find_positions(
    runner: TernaryChoiceRunner, ids: list[int], answer_start: int
) -> dict[str, int]:
    """Locate the four structural token positions in the forced id sequence.

    Resolves each special marker to its (single) token id and searches ``ids``:
    the assistant turn is the LAST <|im_start|>; think_open/close are the
    <think>/</think> tokens. The answer marker is appended LAST, so its first
    token sits at ``answer_start`` (the length of the prompt+prefix prefix); we
    use that index directly rather than re-tokenizing the marker, which would
    miss leading-space BPE merges (" a" != "a"). Missing positions are omitted
    so the caller can log + skip them.
    """
    found: dict[str, int] = {}

    turn = _last_index(ids, _single_token_id(runner, "<|im_start|>"))
    if turn is not None:
        found["turn"] = turn

    open_i = _last_index(ids, _single_token_id(runner, "<think>"))
    if open_i is not None:
        found["think_open"] = open_i
    close_i = _last_index(ids, _single_token_id(runner, "</think>"))
    if close_i is not None:
        found["think_close"] = close_i

    # answer_start indexes the first token of the appended marker; guard the
    # rare case where re-tokenizing prefix+marker shifted a boundary token.
    if 0 <= answer_start < len(ids):
        found["answer"] = answer_start
    return found


def capture_activations(
    runner: TernaryChoiceRunner,
    prompt: SesgoPromptSample,
    layers: list[int],
    sample_dir: Path,
    rel_root: Path,
) -> tuple[list[GeometryActivation], list[str]]:
    """Snapshot residuals along the model's GREEDY NON-THINKING answer path.

    Rather than teacher-forcing a chosen marker, we let the model produce its own
    answer: greedily decode (temperature 0) past the empty <think></think> block
    (skip-thinking prefill), then run one forward pass over that realized sequence
    (chat-templated prompt + skip-thinking prefix + choice prefix + greedy answer)
    and snapshot the per-layer residual stream at the structural positions. The
    "answer" position is the first generated token. Returns the captured
    GeometryActivations plus the position types NOT found.
    """
    prefix = prompt.choice_prefix or "Answer: "
    templated = runner.apply_chat_template(prompt.text)
    head = templated + runner.skip_thinking_prefix + prefix
    # The model's own greedy non-thinking continuation (new tokens after the
    # skip-thinking + choice prefix). Deterministic, so it matches the querier's
    # non-thinking greedy decode.
    greedy = runner.generate(
        prompt.text,
        max_new_tokens=_GREEDY_TOKENS,
        temperature=0.0,
        prefilling=runner.skip_thinking_prefix + prefix,
    )
    forced = head + greedy
    ids = runner.encode_ids(forced, add_special_tokens=True)
    # The answer token is the FIRST generated token = the first index where the
    # full sequence diverges from the prefix-only one (divergence trick choose3
    # uses; robust to leading-space BPE merges, " a" != "a").
    head_ids = runner.encode_ids(head, add_special_tokens=True)
    answer_start = next(
        (i for i in range(min(len(head_ids), len(ids))) if head_ids[i] != ids[i]),
        len(head_ids),
    )

    # One forward pass capturing only the per-layer residual stream.
    _, cache = runner._backend.run_with_cache(
        torch.tensor([ids], device=runner.device), names_filter=_resid_filter(layers)
    )

    positions = find_positions(runner, ids, answer_start)
    captured: list[GeometryActivation] = []
    for ptype in _POSITION_TYPES:
        pos = positions.get(ptype)
        if pos is None:
            continue
        resid = _stack_resid(cache, layers, pos)  # [n_layers, d_model]
        fname = f"sample_{prompt.sample_idx}_{ptype}.pt"
        torch.save(resid, sample_dir / fname)
        token_text = runner.decode_ids([ids[pos]])
        captured.append(
            GeometryActivation(
                position_type=ptype,
                token_position=pos,
                token_text=token_text,
                path=str((sample_dir / fname).relative_to(rel_root)),
            )
        )
    missing = [p for p in _POSITION_TYPES if p not in positions]
    return captured, missing


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for geometry collection."""
    parser = argparse.ArgumentParser(
        description="Capture residual geometry while answering SESGO prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset",
        nargs="?",
        type=Path,
        default=Path("out/sesgo/geometry/prompt_dataset.json"),
        help="Path to a geometry prompt_dataset.json (default: out/sesgo/geometry/...)",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name")
    parser.add_argument(
        "--n-thinking", type=int, default=4, help="Thinking draws per prompt"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max new tokens per draw"
    )
    parser.add_argument(
        "--subsample", type=float, default=1.0, help="Fraction of prompts (0-1)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"), help="Base output directory"
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Optional comma list of layer indices to subset (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    """Collect both SESGO readouts and the residual geometry for every prompt."""
    args = parse_args()
    log_header(f"COLLECT GEOMETRY SAMPLES ({args.model})")

    prompt_dataset = load_prompt_dataset(args.prompt_dataset, args.subsample)
    log(f"[geom] loaded {len(prompt_dataset.samples)} prompts")

    # n_thinking=0 disables the thinking level (skips its sampling cost).
    config = SesgoQueryConfig(
        n_thinking_samples=args.n_thinking,
        max_new_tokens=args.max_new_tokens,
        do_thinking=args.n_thinking > 0,
        subsample=1.0,
    )
    querier = SesgoQuerier(config)
    # Geometry capture needs run_with_cache (residual-stream snapshots). The MLX
    # backend (the Apple-Silicon default) does NOT support it, so force the
    # HuggingFace backend, which provides run_with_cache on CPU/MPS/CUDA and loads
    # any HF model. query_sample/capture_activations below take this runner
    # explicitly, bypassing the querier's MLX auto-load.
    runner = TernaryChoiceRunner(model_name=args.model, backend=ModelBackend.HUGGINGFACE)
    querier._runner = runner

    # All resid_post layers by default; --layers subsets them.
    all_layers = list(range(runner.n_layers))
    layers = (
        [int(x) for x in args.layers.split(",")] if args.layers else all_layers
    )

    out_root = args.out_dir / "sesgo" / "geometry" / runner.model_name.split("/")[-1]
    act_dir = out_root / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    samples: list[GeometrySample] = []
    found_counter: Counter[str] = Counter()
    n_act_files = 0
    for i, prompt in enumerate(prompt_dataset.samples):
        sesgo = querier.query_sample(prompt, runner)
        activations, missing = capture_activations(
            runner, prompt, layers, act_dir, out_root
        )
        for a in activations:
            found_counter[a.position_type] += 1
        n_act_files += len(activations)
        if missing:
            log(f"[geom] sample {prompt.sample_idx}: missing {missing}")
        samples.append(
            GeometrySample(
                sample_idx=prompt.sample_idx,
                question_id=prompt.question_id,
                scaffold_id=prompt.scaffold_id,
                bias_category=prompt.bias_category,
                question_polarity=prompt.question_polarity,
                language=prompt.language,
                gold_label=prompt.gold_label,
                prompt_text=prompt.text,
                non_thinking=sesgo.non_thinking,
                thinking=sesgo.thinking,
                activations=activations,
            )
        )
        log(f"[geom] {i + 1}/{len(prompt_dataset.samples)} done")

    dataset = GeometryDataset(
        prompt_dataset_id=prompt_dataset.dataset_id,
        model=args.model,
        config=config,
        samples=samples,
    )
    out_path = out_root / "samples.json"
    dataset.save_as_json(out_path)

    log_section("geometry collection summary")
    log(f"  samples written : {len(samples)} -> {out_path}")
    log(f"  activation files: {n_act_files} -> {act_dir}")
    log("  positions located (count over samples):")
    for ptype in _POSITION_TYPES:
        log(f"    {ptype:<12} {found_counter.get(ptype, 0)}")


if __name__ == "__main__":
    main()
