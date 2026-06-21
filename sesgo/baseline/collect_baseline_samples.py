"""Run the NON-THINKING BASELINE prompt dataset through a model into a SesgoDataset.

Run-by-path driver for the NON_THINKING_BASELINE study. Loads the baseline prompt
dataset (one rendering per item: NO format variation, NO scaffolding) and queries
every prompt at the NON-THINKING readouts — the teacher-forced 3-way softmax over
the displayed positions remapped to roles, plus the greedy non-thinking decode —
AND the single greedy-THINKING decode (one deterministic temperature-0 generation
WITH reasoning enabled, parsed for the post-</think> answer). The SAMPLED thinking
level is deliberately switched OFF, so this stays a cheap, direct readout.

On ambiguous SESGO items the gold answer is always UNKNOWN, so "accuracy" is the
fraction of predictions that abstain (predict UNKNOWN) rather than pick a group.
Here we log one-line overall non-thinking AND greedy-thinking abstention summaries
as a sanity check; the full slice-by-axis breakdown lives in
visualize_baseline_samples.py.

Output lands at out/sesgo/baseline/<MODEL>/response_samples.json (MODEL == bare
name).

Usage:
  uv run python sesgo/baseline/collect_baseline_samples.py
  uv run python sesgo/baseline/collect_baseline_samples.py \
      out/sesgo/baseline/prompt_dataset.json --model Qwen/Qwen3-0.6B \
      --subsample 0.2
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/baseline/x.py, parents[2] is
# the root.
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
from sesgo.shard_output_paths import apply_shard, shard_out_dir  # noqa: E402


def load_prompt_dataset(path: Path, subsample: float) -> SesgoPromptDataset:
    """Load the prompt dataset, striding the RAW json before deserializing.

    The full dataset can be large; deserializing every prompt just to keep a
    fraction is the run's bottleneck. When subsample < 1 we json-load once, take
    an evenly-spaced stride over the raw sample dicts (so the slice still spans
    every polarity/category/language block, not just the first item), and build
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
    """Parse command-line arguments for non-thinking-baseline collection."""
    parser = argparse.ArgumentParser(
        description="Query the NON_THINKING_BASELINE prompt dataset into a SesgoDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset",
        type=Path,
        nargs="?",
        default=Path("out/sesgo/baseline/prompt_dataset.json"),
        help="Path to the baseline prompt_dataset.json (default: out/sesgo/baseline/...)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HF model name (default: Qwen/Qwen3-0.6B)",
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
        "--shard-index", type=int, default=0, help="This box's shard index (0-based)"
    )
    parser.add_argument(
        "--shard-count", type=int, default=1, help="Total shards (1 == full grid)"
    )
    parser.add_argument(
        "--study",
        default="baseline",
        help=(
            "Output study subdir under <out-dir>/sesgo/ (default: baseline). Set to "
            "baseline_full to route the full-data run to its own distinct tree so it "
            "never clobbers the es-original baseline."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; samples land at <out-dir>/sesgo/<study>/<MODEL>/",
    )
    return parser.parse_args()


def _fmt(value: float | None) -> str:
    """Render an accuracy as a percentage, or n/a when undefined."""
    return f"{value:.1%}" if value is not None else "n/a"


def log_summary(dataset: SesgoDataset) -> None:
    """Report overall non-thinking and greedy-thinking abstention accuracy.

    Accuracy = fraction of predictions that are UNKNOWN (the ambiguous gold).
    Samples with no parsed prediction for a given readout are excluded, so the
    greedy-thinking line also reports how many draws parsed (n) as a sanity check
    against the non-thinking readout (every prompt yields a non-thinking pred).
    """
    nt = [s.correct_non_thinking for s in dataset.samples if s.predicted_non_thinking is not None]
    nt_acc = sum(nt) / len(nt) if nt else None
    gt = [
        s.correct_greedy_thinking
        for s in dataset.samples
        if s.predicted_greedy_thinking is not None
    ]
    gt_acc = sum(gt) / len(gt) if gt else None

    log_section("summary (accuracy = fraction predicted UNKNOWN)")
    log(f"  samples: {len(dataset.samples)}")
    log(f"  non-thinking abstention:    {_fmt(nt_acc):>6} (n={len(nt)})")
    log(f"  greedy-thinking abstention: {_fmt(gt_acc):>6} (n={len(gt)})")


def main() -> None:
    """Load baseline prompts, query non-thinking only, and persist the SesgoDataset."""
    args = parse_args()
    log_header(f"COLLECT NON_THINKING_BASELINE SAMPLES ({args.model})")

    # Stride the raw json before deserializing when subsampling (fast path).
    prompt_dataset = load_prompt_dataset(args.prompt_dataset, args.subsample)
    prompt_dataset = apply_shard(prompt_dataset, args.shard_index, args.shard_count)
    log(f"[collect] loaded {len(prompt_dataset.samples)} prompts")

    # NON-THINKING readouts plus the single greedy-THINKING decode; sampled
    # thinking off. Already subsampled at load, so the querier runs over all
    # loaded prompts.
    config = SesgoQueryConfig(
        do_non_thinking=True,
        do_thinking=False,
        do_greedy=True,
        do_greedy_thinking=True,
        subsample=1.0,
        batch_size=args.batch_size,
    )

    # Resolve the FINAL output path up front so the periodic checkpoint writes the
    # SAME file: a crash leaves a valid partial response_samples.json that the next
    # run resumes from. shard_out_dir needs the bare model name (no org prefix).
    bare_model = args.model.split("/")[-1]
    out_dir = shard_out_dir(
        args.out_dir, args.study, bare_model, args.shard_index, args.shard_count
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "response_samples.json"

    with P("query_dataset"):
        dataset = SesgoQuerier(config).query_dataset(
            prompt_dataset, args.model, checkpoint_path=out_path
        )

    log_summary(dataset)

    # Idempotent final save (checkpoint already wrote this path during the run).
    dataset.save_as_json(out_path)
    log(f"[collect] wrote {out_path}")


if __name__ == "__main__":
    main()
