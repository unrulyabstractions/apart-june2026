"""Collect the DIVERGENCE risk readouts (many thinking draws) into a RiskDataset.

Run-by-path driver (risk analogue of sesgo/divergence/collect_divergence_samples.py).
Loads the divergence prompt dataset (one framing, canonical format, both task
types) and draws a LARGE number of thinking samples per prompt so the sampled
risk-score distribution is well characterized. SESGO measured the spread of the
3-way role distribution; here we already have ScoreSummary (mean / std / entropy /
diversity over the sampled risk scores), so divergence is just baseline collection
with a big n_thinking and the thinking level forced on.

Output lands at out/mental_risk/divergence/<MODEL>/samples.json (MODEL == bare name).

Usage:
  uv run python mental_risk/divergence/collect_divergence_risk.py
  uv run python mental_risk/divergence/collect_divergence_risk.py \
      out/mental_risk/divergence/prompt_dataset.json --model Qwen/Qwen3-0.6B --n-thinking 16
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import numpy as np

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mental_risk.risk_sample_io import load_risk_prompt_dataset  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.risk import RiskDataset, RiskQuerier, RiskQueryConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for divergence-sample collection."""
    parser = argparse.ArgumentParser(
        description="Collect many-draw DIVERGENCE risk into a RiskDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset", type=Path, nargs="?",
        default=Path("out/mental_risk/divergence/prompt_dataset.json"),
        help="Path to the divergence prompt_dataset.json",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name")
    parser.add_argument(
        "--n-thinking", type=int, default=16,
        help="Sampled thinking draws per prompt (large, to characterize the cloud)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
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


def log_summary(dataset: RiskDataset) -> None:
    """Report the mean Shannon entropy of the thinking score clouds (sanity check)."""
    ents = [s.thinking.entropy for s in dataset.samples
            if s.thinking is not None and s.thinking.n > 0]
    log_section("summary (entropy of the sampled risk-score cloud)")
    log(f"  samples: {len(dataset.samples)}")
    log(f"  mean thinking entropy: {np.mean(ents):.4f} (n={len(ents)})"
        if ents else "  mean thinking entropy: n/a")


def main() -> None:
    """Load divergence prompts, draw the score clouds, and persist the dataset."""
    args = parse_args()
    log_header(f"COLLECT DIVERGENCE RISK ({args.model})")

    prompt_dataset = load_risk_prompt_dataset(args.prompt_dataset, args.subsample)
    log(f"[divergence] loaded {len(prompt_dataset.samples)} prompts")

    config = RiskQueryConfig(
        n_thinking_samples=args.n_thinking,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        do_thinking=True,
        subsample=1.0,
    )
    with P("query_dataset"):
        dataset = RiskQuerier(config).query_dataset(prompt_dataset, args.model)

    log_summary(dataset)

    out_dir = args.out_dir / "mental_risk" / "divergence" / dataset.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.json"
    dataset.save_as_json(out_path)
    log(f"[divergence] wrote {out_path}")


if __name__ == "__main__":
    main()
