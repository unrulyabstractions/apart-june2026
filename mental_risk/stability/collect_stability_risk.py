"""Collect the STABILITY risk readout for every prompt into a RiskDataset.

Run-by-path driver (risk analogue of sesgo/stability/collect_stability_samples.py).
Loads the stability prompt dataset (all superficial FORMAT variation — label style
x option order x scale direction x task type, ONE framing) and records the model's
risk readout per prompt. SCORE prompts get a sampled thinking score; CATEGORIZE
prompts get the cheap non-thinking calibrated risk (plus thinking if enabled).

Because the stability grid holds the (subject, framing) fixed and varies only
surface form, the downstream question (visualize_stability_risk.py) is how
CONSISTENT the risk answer is across those format-only rewrites. Here we just log
a one-line spread summary as a sanity check.

Output lands at out/mental_risk/stability/<MODEL>/response_samples.json (MODEL == bare name).

Usage:
  uv run python mental_risk/stability/collect_stability_risk.py
  uv run python mental_risk/stability/collect_stability_risk.py \
      out/mental_risk/stability/prompt_dataset.json --model Qwen/Qwen3-0.6B --subsample 0.25
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import numpy as np

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/mental_risk/stability/x.py, parents[2] is root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mental_risk.risk_sample_io import load_risk_prompt_dataset  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.risk import RiskDataset, RiskQuerier, RiskQueryConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for stability-sample collection."""
    parser = argparse.ArgumentParser(
        description="Collect the STABILITY risk readouts into a RiskDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset", type=Path, nargs="?",
        default=Path("out/mental_risk/stability/prompt_dataset.json"),
        help="Path to the stability prompt_dataset.json",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name")
    parser.add_argument(
        "--n-thinking", type=int, default=8, help="Sampled thinking draws per prompt"
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Skip thinking (CATEGORIZE then uses only the cheap non-thinking risk)",
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
    """Report the mean per-(subject, task) std of predicted risk (the spread)."""
    by_group: dict[tuple, list[float]] = {}
    for s in dataset.samples:
        pred = (
            s.predicted_risk_non_thinking
            if s.predicted_risk_non_thinking is not None
            else s.predicted_risk_thinking
        )
        if pred is None:
            continue
        by_group.setdefault((s.subject_id, s.task_type.value), []).append(pred)
    spreads = [float(np.std(v)) for v in by_group.values() if len(v) > 1]
    log_section("summary (lower spread = more format-stable)")
    log(f"  samples: {len(dataset.samples)}")
    log(f"  groups (subject x task): {len(by_group)}")
    log(f"  mean within-group std of predicted risk: "
        f"{np.mean(spreads):.4f}" if spreads else "  mean within-group std: n/a")


def main() -> None:
    """Load stability prompts, read the risk, and persist the dataset."""
    args = parse_args()
    log_header(f"COLLECT STABILITY RISK ({args.model})")

    prompt_dataset = load_risk_prompt_dataset(args.prompt_dataset, args.subsample)
    log(f"[stability] loaded {len(prompt_dataset.samples)} prompts")

    config = RiskQueryConfig(
        n_thinking_samples=args.n_thinking,
        max_new_tokens=args.max_new_tokens,
        do_thinking=not args.no_thinking,
        subsample=1.0,
    )
    with P("query_dataset"):
        dataset = RiskQuerier(config).query_dataset(prompt_dataset, args.model)

    log_summary(dataset)

    out_dir = args.out_dir / "mental_risk" / "stability" / dataset.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "response_samples.json"
    dataset.save_as_json(out_path)
    log(f"[stability] wrote {out_path}")


if __name__ == "__main__":
    main()
