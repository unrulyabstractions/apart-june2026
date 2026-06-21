"""Collect the cheap NON-THINKING baseline risk readout into a RiskDataset.

Run-by-path driver for the mental_risk BASELINE study (the risk analogue of
sesgo/baseline/collect_baseline_samples.py). Loads the baseline prompt dataset
(one framing, one CATEGORIZE format, no variation) and records the teacher-forced
calibrated P(at risk) for every prompt — a single forward pass, no sampled
reasoning. The downstream question (visualize_baseline_risk.py) is how well that
cheap readout tracks the continuous gold risk.

Thinking is enabled by default too (so the bare grid also yields a sampled
reasoning readout to contrast against), but --no-thinking drops it for the
cheapest possible run.

Output lands at out/mental_risk/baseline/<MODEL>/samples.json (MODEL == bare name).

Usage:
  uv run python mental_risk/baseline/collect_baseline_risk.py
  uv run python mental_risk/baseline/collect_baseline_risk.py \
      out/mental_risk/baseline/prompt_dataset.json --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import numpy as np

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/mental_risk/baseline/x.py, parents[2] is root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mental_risk.risk_sample_io import load_risk_prompt_dataset  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.risk import RiskDataset, RiskQuerier, RiskQueryConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline-sample collection."""
    parser = argparse.ArgumentParser(
        description="Collect the cheap non-thinking baseline risk into a RiskDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset",
        type=Path,
        nargs="?",
        default=Path("out/mental_risk/baseline/prompt_dataset.json"),
        help="Path to the baseline prompt_dataset.json (default: out/mental_risk/baseline/...)",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name")
    parser.add_argument(
        "--n-thinking", type=int, default=8, help="Sampled thinking draws per prompt"
    )
    parser.add_argument(
        "--no-thinking", action="store_true", help="Skip the sampled thinking level"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Max new tokens per draw"
    )
    parser.add_argument(
        "--subsample", type=float, default=1.0, help="Fraction of prompts (0-1)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"), help="Base output directory"
    )
    return parser.parse_args()


def _pearson(preds: list[float], golds: list[float]) -> float | None:
    """Pearson r between paired predictions and golds, or None if degenerate."""
    if len(preds) < 2 or np.std(preds) == 0 or np.std(golds) == 0:
        return None
    return float(np.corrcoef(preds, golds)[0, 1])


def log_summary(dataset: RiskDataset) -> None:
    """Report mean non-thinking risk + its correlation with gold (sanity check)."""
    pairs = [
        (s.predicted_risk_non_thinking, s.gold_risk)
        for s in dataset.samples
        if s.predicted_risk_non_thinking is not None and s.gold_risk is not None
    ]
    log_section("summary")
    log(f"  samples: {len(dataset.samples)}")
    if pairs:
        preds, golds = map(list, zip(*pairs))
        log(f"  mean non-thinking risk: {np.mean(preds):.3f}")
        log(f"  corr(non-thinking, gold): {_pearson(preds, golds)}")
    else:
        log("  no non-thinking + gold pairs to correlate")


def main() -> None:
    """Load baseline prompts, read the cheap risk, and persist the dataset."""
    args = parse_args()
    log_header(f"COLLECT BASELINE RISK ({args.model})")

    prompt_dataset = load_risk_prompt_dataset(args.prompt_dataset, args.subsample)
    log(f"[baseline] loaded {len(prompt_dataset.samples)} prompts")

    config = RiskQueryConfig(
        n_thinking_samples=args.n_thinking,
        max_new_tokens=args.max_new_tokens,
        do_thinking=not args.no_thinking,
        subsample=1.0,  # already subsampled at load
    )
    with P("query_dataset"):
        dataset = RiskQuerier(config).query_dataset(prompt_dataset, args.model)

    log_summary(dataset)

    out_dir = args.out_dir / "mental_risk" / "baseline" / dataset.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.json"
    dataset.save_as_json(out_path)
    log(f"[baseline] wrote {out_path}")


if __name__ == "__main__":
    main()
