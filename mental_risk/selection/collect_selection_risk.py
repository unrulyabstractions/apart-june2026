"""Collect the SELECTION risk readouts (every framing) into a RiskDataset.

Run-by-path driver (risk analogue of sesgo/selection/collect_selection_samples.py).
Loads the selection prompt dataset (the same subject crossed against all framing
conditions — at_risk_of / suffering / safe / intervene — in one canonical
CATEGORIZE format) and records both the non-thinking and thinking risk for every
prompt. The downstream question (visualize_selection_risk.py) is which framing
best tracks the gold risk — the framing-selection analogue of SESGO scaffold
selection. Unlike SESGO there is NO no-op baseline framing, so framings are
ranked against gold rather than against a baseline condition.

Output lands at out/mental_risk/selection/<MODEL>/samples.json (MODEL == bare name).

Usage:
  uv run python mental_risk/selection/collect_selection_risk.py
  uv run python mental_risk/selection/collect_selection_risk.py \
      out/mental_risk/selection/prompt_dataset.json --model Qwen/Qwen3-0.6B --n-thinking 4
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
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.risk import RiskDataset, RiskQuerier, RiskQueryConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for selection-sample collection."""
    parser = argparse.ArgumentParser(
        description="Collect every-framing SELECTION risk into a RiskDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset", type=Path, nargs="?",
        default=Path("out/mental_risk/selection/prompt_dataset.json"),
        help="Path to the selection prompt_dataset.json",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name")
    parser.add_argument(
        "--n-thinking", type=int, default=4, help="Sampled thinking draws per prompt"
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
    """Report the per-framing prompt counts (the selection conditions)."""
    by_framing = Counter(s.framing for s in dataset.samples)
    log_section("summary")
    log(f"  samples:    {len(dataset.samples)}")
    log(f"  by framing: {dict(by_framing)}")


def main() -> None:
    """Load selection prompts, read both levels, and persist the dataset."""
    args = parse_args()
    log_header(f"COLLECT SELECTION RISK ({args.model})")

    prompt_dataset = load_risk_prompt_dataset(args.prompt_dataset, args.subsample)
    log(f"[selection] loaded {len(prompt_dataset.samples)} prompts")

    config = RiskQueryConfig(
        n_thinking_samples=args.n_thinking,
        max_new_tokens=args.max_new_tokens,
        do_thinking=args.n_thinking > 0,
        subsample=1.0,
    )
    with P("query_dataset"):
        dataset = RiskQuerier(config).query_dataset(prompt_dataset, args.model)

    log_summary(dataset)

    out_dir = args.out_dir / "mental_risk" / "selection" / dataset.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.json"
    dataset.save_as_json(out_path)
    log(f"[selection] wrote {out_path}")


if __name__ == "__main__":
    main()
