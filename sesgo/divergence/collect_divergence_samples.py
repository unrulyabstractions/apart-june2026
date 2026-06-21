"""Run the DIVERGENCE prompt dataset through a model into a SesgoDataset.

Run-by-path driver for the DIVERGENCE study. Loads the divergence prompt dataset
(the base ambiguous SESGO items, ONE rendering each, NO scaffolding) and queries
every prompt at both the non-thinking (teacher-forced 3-way softmax remapped to
roles) and thinking (sampled, parsed) levels, then persists the SesgoDataset.

Where STABILITY holds the item fixed and varies surface FORMAT to measure
consistency, DIVERGENCE holds BOTH the item and the format fixed and instead
draws the THINKING level MANY times (n_thinking_samples large by default) to
characterize the DISPERSION of the model's free-form reasoning distribution on a
single ambiguous prompt. Each item's thinking readout is therefore a Monte-Carlo
estimate of a [TARGET, OTHER, UNKNOWN] role distribution, and the question is how
spread out / how far from the correct abstention that distribution is — computed
downstream by visualize_divergence_samples.py.

On ambiguous SESGO items the gold answer is always UNKNOWN, so "accuracy" is the
fraction of predictions that abstain (predict UNKNOWN) rather than pick a group.
Here we just log a one-line abstention summary plus the MEAN per-item thinking
ENTROPY as a sanity check that the draws actually disperse.

Output lands at out/sesgo/divergence/<MODEL>/samples.json (MODEL == bare name).

Usage:
  uv run python sesgo/divergence/collect_divergence_samples.py
  uv run python sesgo/divergence/collect_divergence_samples.py \
      out/sesgo/divergence/prompt_dataset.json --model Qwen/Qwen3-0.6B \
      --n-thinking 16 --subsample 0.5
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/divergence/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import load_json  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.math import probs_to_logprobs, shannon_entropy  # noqa: E402
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
    every category/language/polarity block, not just the first items), and build
    only the kept SesgoPromptSamples. The querier then runs with subsample=1.0.
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
    """Parse command-line arguments for divergence-sample collection."""
    parser = argparse.ArgumentParser(
        description="Query the DIVERGENCE prompt dataset into a SesgoDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset",
        type=Path,
        nargs="?",
        default=Path("out/sesgo/divergence/prompt_dataset.json"),
        help="Path to the divergence prompt_dataset.json (default: out/sesgo/divergence/...)",
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
        # DIVERGENCE characterizes the reasoning DISTRIBUTION, so we want MANY
        # draws per prompt — the default is intentionally larger than stability's.
        help="Sampled thinking generations per prompt (default: 8 — more draws to characterize the distribution)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        # DIVERGENCE wants WIDE diversity across draws so the reasoning
        # distribution is well characterized, so it runs hotter than the
        # stability/selection default (0.7). Push to ~1.1 for even more spread.
        help="Sampling temperature for thinking draws (default: 1.0 — tuned hot for wider diversity)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        # Qwen3 burns budget inside <think>; too small truncates before the
        # answer and the draw is dropped (no parsed role), so keep this generous.
        help="Max new tokens per thinking generation (default: 512)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of prompts to query, 0-1 (default: 1.0 == all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Prompts per batched forward pass (default: 1 == single-sample path)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; samples land at <out-dir>/sesgo/divergence/<MODEL>/",
    )
    return parser.parse_args()


def _fmt(value: float | None) -> str:
    """Render an accuracy as a percentage, or n/a when undefined."""
    return f"{value:.1%}" if value is not None else "n/a"


def _mean_thinking_entropy(dataset: SesgoDataset) -> tuple[float | None, int]:
    """Mean Shannon entropy (nats) of each item's thinking role-distribution.

    Each item's `thinking.mean` is a [TARGET, OTHER, UNKNOWN] probability vector
    estimated over the parsed draws; its Shannon entropy measures how DISPERSED
    that item's reasoning distribution is. We average over items with at least
    one parsed draw (sample_size > 0). Returns (mean_entropy, n_items_used).
    """
    ents: list[float] = []
    for s in dataset.samples:
        th = s.thinking
        if th is None or th.sample_size == 0:
            continue
        ents.append(float(shannon_entropy(probs_to_logprobs(th.mean))))
    return (sum(ents) / len(ents) if ents else None), len(ents)


def log_summary(dataset: SesgoDataset) -> None:
    """Report thinking abstention accuracy + mean per-item thinking entropy.

    Accuracy = fraction of predictions that are UNKNOWN (the ambiguous gold).
    Predictions with no parsed thinking draw are excluded from the thinking rate
    (they have no decodable answer). The mean thinking entropy is the headline
    DIVERGENCE signal: higher entropy == the reasoning distribution is more
    spread across the three roles instead of collapsing onto one.
    """
    nt = [s.correct_non_thinking for s in dataset.samples if s.predicted_non_thinking is not None]
    th = [s.correct_thinking for s in dataset.samples if s.predicted_thinking is not None]
    nt_acc = sum(nt) / len(nt) if nt else None
    th_acc = sum(th) / len(th) if th else None
    mean_ent, n_ent = _mean_thinking_entropy(dataset)

    log_section("summary (accuracy = fraction predicted UNKNOWN)")
    log(f"  samples: {len(dataset.samples)}")
    log(f"  non-thinking abstention: {_fmt(nt_acc):>6} (n={len(nt)})")
    log(f"  thinking abstention:     {_fmt(th_acc):>6} (n={len(th)})")
    ent_str = f"{mean_ent:.3f}" if mean_ent is not None else "n/a"
    log(f"  mean thinking entropy:   {ent_str:>6} nats (n={n_ent})")


def main() -> None:
    """Load divergence prompts, query the model, and persist the SesgoDataset."""
    args = parse_args()
    log_header(f"COLLECT DIVERGENCE SAMPLES ({args.model})")

    # Stride the raw json before deserializing when subsampling (fast path).
    prompt_dataset = load_prompt_dataset(args.prompt_dataset, args.subsample)
    log(f"[collect] loaded {len(prompt_dataset.samples)} prompts")

    # Already subsampled at load, so the querier runs over all loaded prompts.
    # DIVERGENCE: thinking + non-thinking on, no extra greedy decode, MANY draws.
    config = SesgoQueryConfig(
        do_non_thinking=True,
        do_thinking=True,
        n_thinking_samples=args.n_thinking,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_greedy=False,
        subsample=1.0,
        batch_size=args.batch_size,
    )
    with P("query_dataset"):
        dataset = SesgoQuerier(config).query_dataset(prompt_dataset, args.model)

    log_summary(dataset)

    # out/sesgo/divergence/<MODEL>/samples.json, keyed by bare model name.
    out_dir = args.out_dir / "sesgo" / "divergence" / dataset.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.json"
    dataset.save_as_json(out_path)
    log(f"[collect] wrote {out_path}")


if __name__ == "__main__":
    main()
