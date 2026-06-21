"""Capture residual-stream geometry while answering SESGO prompts.

Run-by-path driver for the SESGO GEOMETRY half. For every prompt it (1) runs the
normal SesgoQuerier readout (non-thinking 3-way + thinking draws) and (2) follows
the model's GREEDY NON-THINKING answer path (greedy decode past an empty
<think></think> block) and snapshots the FULL per-layer residual stream
([n_layers, d_model]) at up to four MODEL-AWARE structural token positions:

    turn         - the last assistant-turn marker (family-specific: Qwen
                   <|im_start|>, Llama <|start_header_id|>, Gemma
                   <start_of_turn>, Mistral [/INST])
    think_open   - the <think> token of the skip-thinking prefix (reasoning
                   models only; absent for Llama/Gemma/Mistral)
    think_close  - the </think> token (reasoning models only)
    answer       - the first greedily-generated answer token

For non-reasoning families the think_* positions simply don't exist, so each
sample captures turn + answer; reasoning models additionally capture think_*.

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

from src.common.device_utils import clear_gpu_memory  # noqa: E402
from src.datasets.sesgo_eval.sesgo_batched_query import query_chunk  # noqa: E402

from sesgo.geometry.geometry_capture_helpers import (  # noqa: E402
    _POSITION_TYPES,
    capture_activations,
)
from sesgo.geometry.batched_geometry_capture import (  # noqa: E402
    capture_activations_batch,
)

# Free accelerator memory every this many CHUNKS so long runs don't accumulate.
_GPU_CLEAR_EVERY = 25


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


def _make_sample(prompt: SesgoPromptSample, sesgo, activations) -> GeometrySample:
    """Assemble one GeometrySample from its readout + captured activations."""
    return GeometrySample(
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


def collect_geometry(
    prompts: list[SesgoPromptSample],
    querier: SesgoQuerier,
    runner: TernaryChoiceRunner,
    layers: list[int],
    act_dir: Path,
    out_root: Path,
    bs: int,
) -> tuple[list[GeometrySample], Counter, int]:
    """Run readout + residual capture over the prompts in chunks of ``bs``.

    ``bs == 1`` keeps the exact single-sample path; ``bs > 1`` batches both the
    SESGO readout (``query_chunk``) and the residual capture
    (``capture_activations_batch``) into shared forward passes per chunk.
    """
    samples: list[GeometrySample] = []
    found: Counter = Counter()
    n_act_files = 0
    total = len(prompts)
    for start in range(0, total, bs):
        chunk = prompts[start : start + bs]
        if bs == 1:
            readouts = [querier.query_sample(chunk[0], runner)]
            caps = [capture_activations(runner, chunk[0], layers, act_dir, out_root)]
        else:
            readouts = query_chunk(chunk, runner, querier.config)
            captured, missing = capture_activations_batch(
                runner, chunk, layers, act_dir, out_root
            )
            caps = list(zip(captured, missing))
        for prompt, sesgo, (activations, missing) in zip(chunk, readouts, caps):
            for a in activations:
                found[a.position_type] += 1
            n_act_files += len(activations)
            if missing:
                log(f"[geom] sample {prompt.sample_idx}: missing {missing}")
            samples.append(_make_sample(prompt, sesgo, activations))
        log(f"[geom] {min(start + bs, total)}/{total} done")
        if (start // bs + 1) % _GPU_CLEAR_EVERY == 0:
            clear_gpu_memory()
    clear_gpu_memory()
    return samples, found, n_act_files


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
        "--batch-size",
        type=int,
        default=1,
        help="Prompts per batched forward pass (default: 1 == single-sample path)",
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
        batch_size=max(1, args.batch_size),
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

    bs = max(1, args.batch_size)
    samples, found_counter, n_act_files = collect_geometry(
        prompt_dataset.samples, querier, runner, layers, act_dir, out_root, bs
    )

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
