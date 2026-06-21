"""Run a RiskPromptDataset through a model and write a RiskDataset.

Run-by-path driver for baseline task 1.b. Loads a prompt dataset produced by
generate_prompt_dataset.py, queries every prompt at both the non-thinking
(teacher-forced) and thinking (sampled) levels, and persists the resulting
RiskDataset plus a short summary.

Output lands at out/mental_risk/<MODEL>/responses.json (MODEL == bare name).

Usage:
  uv run python mental_risk/baseline/collect_llm_responses.py \
      out/mental_risk/prompt_dataset.json
  uv run python mental_risk/baseline/collect_llm_responses.py \
      PROMPTS.json --model Qwen/Qwen3-0.6B --n-thinking 8 --subsample 0.5
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

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.prompt import RiskPromptDataset  # noqa: E402
from src.datasets.risk import RiskDataset, RiskQuerier, RiskQueryConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for response collection."""
    parser = argparse.ArgumentParser(
        description="Query a RiskPromptDataset through a model into a RiskDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset",
        type=Path,
        help="Path to prompt_dataset.json from generate_prompt_dataset.py",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HF model name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--n-thinking",
        type=int,
        default=8,
        help="Sampled thinking generations per prompt (default: 8)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for thinking draws (default: 0.7)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        # Qwen3 burns budget inside <think>; too small truncates before the
        # answer and the draw is dropped, so keep this generous.
        help="Max new tokens per thinking generation (default: 512)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of prompts to query, 0-1 (default: 1.0 == all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output directory; responses land at <out-dir>/mental_risk/<MODEL>/",
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float | None:
    """Mean over present values, or None when nothing is comparable."""
    return float(np.mean(values)) if values else None


def _pearson(preds: list[float], golds: list[float]) -> float | None:
    """Pearson r between paired predictions and golds, or None if degenerate."""
    if len(preds) < 2 or np.std(preds) == 0 or np.std(golds) == 0:
        return None
    return float(np.corrcoef(preds, golds)[0, 1])


def log_summary(dataset: RiskDataset) -> None:
    """Report mean risks and prediction-vs-gold correlation for a quick gut check."""
    non_thinking = [
        s.predicted_risk_non_thinking
        for s in dataset.samples
        if s.predicted_risk_non_thinking is not None
    ]
    thinking = [
        s.predicted_risk_thinking
        for s in dataset.samples
        if s.predicted_risk_thinking is not None
    ]
    # Pair predictions with gold only where both exist, per level.
    nt_pairs = [
        (s.predicted_risk_non_thinking, s.gold_risk)
        for s in dataset.samples
        if s.predicted_risk_non_thinking is not None and s.gold_risk is not None
    ]
    th_pairs = [
        (s.predicted_risk_thinking, s.gold_risk)
        for s in dataset.samples
        if s.predicted_risk_thinking is not None and s.gold_risk is not None
    ]

    log_section("summary")
    log(f"  samples:            {len(dataset.samples)}")
    log(f"  mean non-thinking:  {_mean(non_thinking)}")
    log(f"  mean thinking:      {_mean(thinking)}")
    log(
        f"  corr(non-thinking, gold): {_pearson(*map(list, zip(*nt_pairs)))}"
        if nt_pairs
        else "  corr(non-thinking, gold): n/a"
    )
    log(
        f"  corr(thinking, gold):     {_pearson(*map(list, zip(*th_pairs)))}"
        if th_pairs
        else "  corr(thinking, gold):     n/a"
    )


def main() -> None:
    """Load prompts, query the model, and persist the RiskDataset + summary."""
    args = parse_args()
    log_header(f"COLLECT LLM RESPONSES ({args.model})")

    prompt_dataset = RiskPromptDataset.from_json(args.prompt_dataset)
    log(f"[collect] loaded {len(prompt_dataset.samples)} prompts")

    # Subsampling is applied by RiskQuerier from the config (args.subsample).
    config = RiskQueryConfig(
        n_thinking_samples=args.n_thinking,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        subsample=args.subsample,
    )
    with P("query_dataset"):
        dataset = RiskQuerier(config).query_dataset(prompt_dataset, args.model)

    log_summary(dataset)

    # out/mental_risk/<MODEL>/responses.json, keyed by bare model name.
    out_dir = args.out_dir / "mental_risk" / dataset.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "responses.json"
    dataset.save_as_json(out_path)
    log(f"[collect] wrote {out_path}")


if __name__ == "__main__":
    main()
