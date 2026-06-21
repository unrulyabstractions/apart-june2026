"""Run a SesgoPromptDataset through a model and write a SesgoDataset.

Run-by-path driver for baseline task 1.b. Loads a prompt dataset produced by
generate_prompt_dataset.py, queries every prompt at both the non-thinking
(teacher-forced 3-way softmax remapped to roles) and thinking (sampled,
parsed) levels, and persists the resulting SesgoDataset plus a per-scaffold
accuracy summary.

On ambiguous SESGO items the gold answer is always UNKNOWN, so "accuracy" is
the fraction of predictions that abstain (predict UNKNOWN) rather than pick a
group. The summary breaks this down by scaffold condition — the no-scaffold
baseline (None) and each scaffold_id — which is the headline of the study: a
working scaffold should raise the % unknown versus the baseline.

Output lands at out/sesgo/<MODEL>/responses.json (MODEL == bare model name).

Usage:
  uv run python sesgo/baseline/collect_llm_responses.py out/sesgo/prompt_dataset.json
  uv run python sesgo/baseline/collect_llm_responses.py \
      PROMPTS.json --model Qwen/Qwen3-0.6B --n-thinking 8 --subsample 0.5
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/baseline/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.prompt import SesgoPromptDataset  # noqa: E402
from src.datasets.sesgo_eval import (  # noqa: E402
    SesgoDataset,
    SesgoQuerier,
    SesgoQueryConfig,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for response collection."""
    parser = argparse.ArgumentParser(
        description="Query a SesgoPromptDataset through a model into a SesgoDataset",
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
        help="Base output directory; responses land at <out-dir>/sesgo/<MODEL>/",
    )
    return parser.parse_args()


def _accuracy(flags: list[bool]) -> float | None:
    """Fraction of True flags (== predicted UNKNOWN), or None when empty."""
    return sum(flags) / len(flags) if flags else None


def _fmt(value: float | None) -> str:
    """Render an accuracy as a percentage, or n/a when undefined."""
    return f"{value:.1%}" if value is not None else "n/a"


def log_summary(dataset: SesgoDataset) -> None:
    """Report per-scaffold abstention accuracy so with/without is visible.

    For each scaffold condition we report the non-thinking and thinking
    accuracy = fraction of predictions that are UNKNOWN (the ambiguous gold).
    Predictions with no parsed thinking draw are excluded from the thinking
    rate (they have no decodable answer), so the two columns can rest on
    different denominators.
    """
    nt_flags: dict[str | None, list[bool]] = defaultdict(list)
    th_flags: dict[str | None, list[bool]] = defaultdict(list)
    for s in dataset.samples:
        if s.predicted_non_thinking is not None:
            nt_flags[s.scaffold_id].append(s.correct_non_thinking)
        if s.predicted_thinking is not None:
            th_flags[s.scaffold_id].append(s.correct_thinking)

    log_section("summary (accuracy = fraction predicted UNKNOWN)")
    log(f"  samples: {len(dataset.samples)}")
    # None (baseline) first, then scaffolds in encounter order, so the baseline
    # anchors the comparison at the top of the table.
    conditions = sorted(
        set(nt_flags) | set(th_flags),
        key=lambda sid: (sid is not None, sid or ""),
    )
    for sid in conditions:
        label = sid or "(none / baseline)"
        nt = _accuracy(nt_flags.get(sid, []))
        th = _accuracy(th_flags.get(sid, []))
        log(
            f"  {label:<34} non-thinking={_fmt(nt):>6} "
            f"(n={len(nt_flags.get(sid, []))})  "
            f"thinking={_fmt(th):>6} (n={len(th_flags.get(sid, []))})"
        )


def main() -> None:
    """Load prompts, query the model, and persist the SesgoDataset + summary."""
    args = parse_args()
    log_header(f"COLLECT LLM RESPONSES ({args.model})")

    prompt_dataset = SesgoPromptDataset.from_json(args.prompt_dataset)
    log(f"[collect] loaded {len(prompt_dataset.samples)} prompts")

    # Subsampling is applied by SesgoQuerier from the config (args.subsample).
    config = SesgoQueryConfig(
        n_thinking_samples=args.n_thinking,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        subsample=args.subsample,
    )
    with P("query_dataset"):
        dataset = SesgoQuerier(config).query_dataset(prompt_dataset, args.model)

    log_summary(dataset)

    # out/sesgo/<MODEL>/responses.json, keyed by bare model name.
    out_dir = args.out_dir / "sesgo" / dataset.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "responses.json"
    dataset.save_as_json(out_path)
    log(f"[collect] wrote {out_path}")


if __name__ == "__main__":
    main()
