"""Capture residual-stream geometry while answering mental_risk prompts.

Run-by-path driver for the mental_risk GEOMETRY study (risk analogue of
sesgo/geometry/collect_geometry_samples.py). For every prompt it (1) runs the
normal RiskQuerier readout (non-thinking calibrated risk + thinking score cloud)
and (2) follows the model's GREEDY NON-THINKING answer path and snapshots the
FULL per-layer residual stream at four structural token positions (turn /
think_open / think_close / answer). Each snapshot is torch.save'd under
out/mental_risk/geometry/<MODEL>/activations/ and referenced by relative path only
from a RiskGeometrySample, so response_samples.json stays small.

The headline downstream question is geometric: how far does each FRAMING move
these representations versus the others (analyze_geometry_risk.py).

Geometry needs run_with_cache, which only the HuggingFace backend exposes (the
Apple-Silicon MLX default does NOT), so we force backend=HUGGINGFACE — exactly as
the SESGO geometry collector forces it on TernaryChoiceRunner.

Usage:
  uv run python mental_risk/geometry/collect_geometry_risk.py
  uv run python mental_risk/geometry/collect_geometry_risk.py PROMPTS.json \
      --model Qwen/Qwen3-0.6B --n-thinking 4 --subsample 0.5 --layers 0,6,12
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mental_risk.risk_sample_io import load_risk_prompt_dataset  # noqa: E402
from src.binary_choice.binary_choice_runner import BinaryChoiceRunner  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.inference.backends import ModelBackend  # noqa: E402
from src.datasets.risk import RiskQuerier, RiskQueryConfig  # noqa: E402
from src.datasets.risk_geometry import (  # noqa: E402
    POSITION_TYPES,
    RiskGeometryDataset,
    RiskGeometrySample,
    capture_activations,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for geometry collection."""
    parser = argparse.ArgumentParser(
        description="Capture residual geometry while answering mental_risk prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset", nargs="?", type=Path,
        default=Path("out/mental_risk/geometry/prompt_dataset.json"),
        help="Path to a geometry prompt_dataset.json",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name")
    parser.add_argument("--n-thinking", type=int, default=4, help="Thinking draws per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per draw")
    parser.add_argument("--subsample", type=float, default=1.0, help="Fraction of prompts (0-1)")
    parser.add_argument("--out-dir", type=Path, default=Path("out"), help="Base output directory")
    parser.add_argument(
        "--layers", default=None,
        help="Optional comma list of layer indices to subset (default: all)",
    )
    return parser.parse_args()


def _build_runner(model: str) -> BinaryChoiceRunner:
    """A BinaryChoiceRunner forced onto the HuggingFace backend (run_with_cache)."""
    return BinaryChoiceRunner(model_name=model, backend=ModelBackend.HUGGINGFACE)


def _to_geometry_sample(prompt, readout, activations) -> RiskGeometrySample:
    """Assemble one RiskGeometrySample from a prompt + its readout + activations."""
    return RiskGeometrySample(
        sample_idx=prompt.sample_idx,
        subject_id=prompt.subject_id,
        disorder=prompt.disorder,
        framing=prompt.framing,
        language=prompt.language,
        task_type=prompt.task_type,
        gold_risk=prompt.gold_risk,
        prompt_text=prompt.text,
        non_thinking=readout.non_thinking,
        thinking=readout.thinking,
        activations=activations,
    )


def main() -> None:
    """Collect both risk readouts and the residual geometry for every prompt."""
    args = parse_args()
    log_header(f"COLLECT GEOMETRY RISK ({args.model})")

    prompt_dataset = load_risk_prompt_dataset(args.prompt_dataset, args.subsample)
    log(f"[geom] loaded {len(prompt_dataset.samples)} prompts")

    config = RiskQueryConfig(
        n_thinking_samples=args.n_thinking,
        max_new_tokens=args.max_new_tokens,
        do_thinking=args.n_thinking > 0,
        subsample=1.0,
    )
    querier = RiskQuerier(config)
    runner = _build_runner(args.model)
    querier._runner = runner  # bypass the querier's default backend auto-load

    all_layers = list(range(runner.n_layers))
    layers = [int(x) for x in args.layers.split(",")] if args.layers else all_layers

    out_root = args.out_dir / "mental_risk" / "geometry" / runner.model_name.split("/")[-1]
    act_dir = out_root / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    samples: list[RiskGeometrySample] = []
    found_counter: Counter[str] = Counter()
    n_act_files = 0
    for i, prompt in enumerate(prompt_dataset.samples):
        readout = querier.query_sample(prompt, runner)
        activations, missing = capture_activations(runner, prompt, layers, act_dir, out_root)
        for a in activations:
            found_counter[a.position_type] += 1
        n_act_files += len(activations)
        if missing:
            log(f"[geom] sample {prompt.sample_idx}: missing {missing}")
        samples.append(_to_geometry_sample(prompt, readout, activations))
        log(f"[geom] {i + 1}/{len(prompt_dataset.samples)} done")

    dataset = RiskGeometryDataset(
        prompt_dataset_id=prompt_dataset.dataset_id,
        model=args.model,
        config=config,
        samples=samples,
    )
    out_path = out_root / "response_samples.json"
    dataset.save_as_json(out_path)

    log_section("geometry collection summary")
    log(f"  samples written : {len(samples)} -> {out_path}")
    log(f"  activation files: {n_act_files} -> {act_dir}")
    log("  positions located (count over samples):")
    for ptype in POSITION_TYPES:
        log(f"    {ptype:<12} {found_counter.get(ptype, 0)}")


if __name__ == "__main__":
    main()
