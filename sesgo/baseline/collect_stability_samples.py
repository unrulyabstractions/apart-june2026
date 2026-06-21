"""Run the STABILITY prompt dataset through a model into a SesgoDataset.

Run-by-path driver for the STABILITY half. Loads the stability prompt dataset
(all superficial FORMAT variation, NO scaffolding), queries every prompt at both
the non-thinking (teacher-forced 3-way softmax remapped to roles) and thinking
(sampled, parsed) levels, and persists the resulting SesgoDataset.

On ambiguous SESGO items the gold answer is always UNKNOWN, so "accuracy" is the
fraction of predictions that abstain (predict UNKNOWN) rather than pick a group.
Because the stability grid holds the item fixed and varies only surface form, the
downstream question is how CONSISTENT that abstention is across the variations —
computed by visualize_stability_samples.py. Here we just log a one-line overall
abstention summary as a sanity check.

Output lands at out/sesgo/stability/<MODEL>/samples.json (MODEL == bare name).

Usage:
  uv run python sesgo/baseline/collect_stability_samples.py
  uv run python sesgo/baseline/collect_stability_samples.py \
      out/sesgo/stability/prompt_dataset.json --model Qwen/Qwen3-0.6B \
      --n-thinking 4 --subsample 0.2
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/baseline/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import load_json  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.prompt import (  # noqa: E402
    SesgoPromptConfig,
    SesgoPromptDataset,
    SesgoPromptSample,
)
from src.datasets.sesgo_eval import (  # noqa: E402
    SesgoDataset,
    SesgoQuerier,
    SesgoQueryConfig,
)


def load_prompt_dataset(path: Path, subsample: float) -> SesgoPromptDataset:
    """Load the prompt dataset, striding the RAW json before deserializing.

    The full dataset can be large; deserializing every prompt just to keep a
    fraction is the run's bottleneck. When subsample < 1 we json-load once, take
    an evenly-spaced stride over the raw sample dicts (so the slice still spans
    every permutation/label-style/category block, not just the first item), and
    build only the kept SesgoPromptSamples. The querier then runs with
    subsample=1.0.
    """
    if subsample >= 1.0:
        return SesgoPromptDataset.from_json(path)
    # load_json (not raw json) rejoins readable_text line-lists back to strings.
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for stability-sample collection."""
    parser = argparse.ArgumentParser(
        description="Query the STABILITY prompt dataset into a SesgoDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset",
        type=Path,
        nargs="?",
        default=Path("out/sesgo/stability/prompt_dataset.json"),
        help="Path to the stability prompt_dataset.json (default: out/sesgo/stability/...)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HF model name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--n-thinking",
        type=int,
        default=4,
        help="Sampled thinking generations per prompt (default: 4)",
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
        default=256,
        # Qwen3 burns budget inside <think>; too small truncates before the
        # answer and the draw is dropped, so keep this generous.
        help="Max new tokens per thinking generation (default: 256)",
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
        help="Base output dir; samples land at <out-dir>/sesgo/stability/<MODEL>/",
    )
    return parser.parse_args()


def _fmt(value: float | None) -> str:
    """Render an accuracy as a percentage, or n/a when undefined."""
    return f"{value:.1%}" if value is not None else "n/a"


def log_summary(dataset: SesgoDataset) -> None:
    """Report overall non-thinking/thinking abstention accuracy.

    Accuracy = fraction of predictions that are UNKNOWN (the ambiguous gold).
    Predictions with no parsed thinking draw are excluded from the thinking rate
    (they have no decodable answer), so the two columns rest on possibly
    different denominators.
    """
    nt = [s.correct_non_thinking for s in dataset.samples if s.predicted_non_thinking is not None]
    th = [s.correct_thinking for s in dataset.samples if s.predicted_thinking is not None]
    nt_acc = sum(nt) / len(nt) if nt else None
    th_acc = sum(th) / len(th) if th else None

    log_section("summary (accuracy = fraction predicted UNKNOWN)")
    log(f"  samples: {len(dataset.samples)}")
    log(f"  non-thinking abstention: {_fmt(nt_acc):>6} (n={len(nt)})")
    log(f"  thinking abstention:     {_fmt(th_acc):>6} (n={len(th)})")


def main() -> None:
    """Load stability prompts, query the model, and persist the SesgoDataset."""
    args = parse_args()
    log_header(f"COLLECT STABILITY SAMPLES ({args.model})")

    # Stride the raw json before deserializing when subsampling (fast path).
    prompt_dataset = load_prompt_dataset(args.prompt_dataset, args.subsample)
    log(f"[collect] loaded {len(prompt_dataset.samples)} prompts")

    # Already subsampled at load, so the querier runs over all loaded prompts.
    config = SesgoQueryConfig(
        n_thinking_samples=args.n_thinking,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        subsample=1.0,
    )
    with P("query_dataset"):
        dataset = SesgoQuerier(config).query_dataset(prompt_dataset, args.model)

    log_summary(dataset)

    # out/sesgo/stability/<MODEL>/samples.json, keyed by bare model name.
    out_dir = args.out_dir / "sesgo" / "stability" / dataset.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.json"
    dataset.save_as_json(out_path)
    log(f"[collect] wrote {out_path}")


if __name__ == "__main__":
    main()
