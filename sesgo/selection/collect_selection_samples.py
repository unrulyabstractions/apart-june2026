"""Run the SELECTION prompt dataset through a model into a SesgoDataset.

Run-by-path driver for the SELECTION study. The selection prompt dataset crosses
the SAME ambiguous SESGO item against FIVE scaffold conditions — the no-scaffold
baseline (scaffold_id == None) plus the four debiasing scaffolds — so we can ask
which preamble best pushes the model toward abstaining. It queries every prompt
at both the non-thinking (teacher-forced 3-way softmax remapped to roles) and
thinking (sampled, parsed) levels and persists the resulting SesgoDataset.

On ambiguous SESGO items the gold answer is always UNKNOWN, so "accuracy" is the
fraction of predictions that abstain (predict UNKNOWN) rather than pick a group.
The point of the selection grid is to compare that abstention rate ACROSS the
five scaffold conditions; the actual SELECT (pick the best scaffold) lives in
visualize_selection_samples.py, but we log a per-scaffold non-thinking + thinking
abstention table here as a sanity check.

Output lands at out/sesgo/selection/<MODEL>/response_samples.json (MODEL == bare name).

Usage:
  uv run python sesgo/selection/collect_selection_samples.py
  uv run python sesgo/selection/collect_selection_samples.py \
      out/sesgo/selection/prompt_dataset.json --model Qwen/Qwen3-0.6B \
      --n-thinking 4 --subsample 0.5
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections import defaultdict
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/selection/x.py, parents[2] is the root.
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

# The no-scaffold condition has scaffold_id == None; label it so it sorts first.
_BASELINE = "(baseline)"


def load_prompt_dataset(path: Path, subsample: float) -> SesgoPromptDataset:
    """Load the prompt dataset, striding the RAW json before deserializing.

    The full dataset can be large; deserializing every prompt just to keep a
    fraction is the run's bottleneck. When subsample < 1 we json-load once, take
    an evenly-spaced stride over the raw sample dicts (so the slice still spans
    every scaffold/item block, not just the first item), and build only the kept
    SesgoPromptSamples. The querier then runs with subsample=1.0.
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
    """Parse command-line arguments for selection-sample collection."""
    parser = argparse.ArgumentParser(
        description="Query the SELECTION prompt dataset into a SesgoDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset",
        type=Path,
        nargs="?",
        default=Path("out/sesgo/selection/prompt_dataset.json"),
        help="Path to the selection prompt_dataset.json (default: out/sesgo/selection/...)",
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
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; samples land at <out-dir>/sesgo/selection/<MODEL>/",
    )
    return parser.parse_args()


def _scaffold_label(scaffold_id: str | None) -> str:
    """Human label for a scaffold condition; None == the no-scaffold baseline."""
    return scaffold_id or _BASELINE


def _fmt(value: float | None) -> str:
    """Render an accuracy as a percentage, or n/a when undefined."""
    return f"{value:.1%}" if value is not None else "n/a"


def log_summary(dataset: SesgoDataset) -> None:
    """Report PER-SCAFFOLD non-thinking/thinking abstention accuracy.

    Accuracy = fraction of predictions that are UNKNOWN (the ambiguous gold).
    Predictions with no parsed thinking draw are excluded from the thinking rate
    (they have no decodable answer), so the two columns can rest on different
    denominators. The baseline sorts first; the four scaffolds follow in
    canonical (sorted) order. The actual SELECT happens in the visualizer, but
    we surface the table here so a collection run alone is interpretable.
    """
    # by_scaffold[label] = ([non-thinking flags], [thinking flags])
    nt_flags: dict[str, list[bool]] = defaultdict(list)
    th_flags: dict[str, list[bool]] = defaultdict(list)
    for s in dataset.samples:
        label = _scaffold_label(s.scaffold_id)
        if s.predicted_non_thinking is not None:
            nt_flags[label].append(s.correct_non_thinking)
        if s.predicted_thinking is not None:
            th_flags[label].append(s.correct_thinking)

    labels = set(nt_flags) | set(th_flags)
    rest = sorted(labels - {_BASELINE})
    ordered = ([_BASELINE] if _BASELINE in labels else []) + rest

    log_section("per-scaffold summary (accuracy = fraction predicted UNKNOWN)")
    log(f"  samples: {len(dataset.samples)}")
    log(f"  {'scaffold':<34} {'non-thinking':>16} {'thinking':>16}")
    for label in ordered:
        nt = nt_flags.get(label, [])
        th = th_flags.get(label, [])
        nt_acc = sum(nt) / len(nt) if nt else None
        th_acc = sum(th) / len(th) if th else None
        nt_txt = f"{_fmt(nt_acc)} (n={len(nt)})"
        th_txt = f"{_fmt(th_acc)} (n={len(th)})"
        log(f"  {label:<34} {nt_txt:>16} {th_txt:>16}")
    log("  NOTE: ambiguous gold is always UNKNOWN, so abstention == accuracy.")
    log("  SELECT (best scaffold) is computed by visualize_selection_samples.py.")


def main() -> None:
    """Load selection prompts, query the model, and persist the SesgoDataset."""
    args = parse_args()
    log_header(f"COLLECT SELECTION SAMPLES ({args.model})")

    # Stride the raw json before deserializing when subsampling (fast path).
    prompt_dataset = load_prompt_dataset(args.prompt_dataset, args.subsample)
    prompt_dataset = apply_shard(prompt_dataset, args.shard_index, args.shard_count)
    log(f"[collect] loaded {len(prompt_dataset.samples)} prompts")
    n_scaffolds = len({s.scaffold_id for s in prompt_dataset.samples})
    log(f"[collect] {n_scaffolds} scaffold condition(s) present (baseline + scaffolds)")

    # Already subsampled at load, so the querier runs over all loaded prompts.
    # do_non_thinking + do_thinking + do_greedy mirror the house defaults so both
    # levels are recorded for every scaffold condition.
    config = SesgoQueryConfig(
        do_non_thinking=True,
        do_thinking=True,
        n_thinking_samples=args.n_thinking,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_greedy=True,
        subsample=1.0,
        batch_size=args.batch_size,
    )
    with P("query_dataset"):
        dataset = SesgoQuerier(config).query_dataset(prompt_dataset, args.model)

    log_summary(dataset)

    # out/sesgo/selection/<MODEL>/response_samples.json (per-shard subdir when sharded).
    out_dir = shard_out_dir(
        args.out_dir, "selection", dataset.model_name, args.shard_index, args.shard_count
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "response_samples.json"
    dataset.save_as_json(out_path)
    log(f"[collect] wrote {out_path}")


if __name__ == "__main__":
    main()
