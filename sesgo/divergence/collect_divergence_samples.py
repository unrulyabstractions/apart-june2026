"""Run ONE divergence prompt through a model into a SesgoDataset.

Run-by-path driver for the DIVERGENCE study. Picks ONE representative ambiguous
SESGO prompt and reads it at every level: the non-thinking teacher-forced 3-way
softmax (remapped to roles), the 2-option forced choice, the greedy-thinking
baseline, and — the headline — the THINKING level sampled MANY times.

DIVERGENCE is "forking at a single position, but sampled deeply": it holds BOTH
the item AND the surface format fixed and draws the free-form THINKING generation
``--n-thinking`` times (100 by default) to estimate, by Monte-Carlo, the model's
outcome distribution over [TARGET, OTHER, UNKNOWN] on that one ambiguous prompt.
The question is how spread out / how far from correct abstention that
distribution is — computed downstream by visualize_divergence_samples.py.

Each of the ~100 CoT draws ALSO records its mean next-token VOCAB ENTROPY (the
mean Shannon entropy of the model's next-token distribution over the generated
tokens), captured for free during sampling. The per-draw entropies are stored on
the thinking summary so the ~100-draw entropy distribution is available
downstream — relating a draw's per-token uncertainty to the outcome it commits to.

On ambiguous SESGO items the gold answer is always UNKNOWN, so "accuracy" is the
fraction of predictions that abstain (predict UNKNOWN). The summary logs that plus
the role-mix entropy AND the mean/spread of per-draw vocab entropy as sanity checks.

Output lands at out/sesgo/divergence/<MODEL>/response_samples.json (MODEL == bare name).

Usage:
  uv run python sesgo/divergence/collect_divergence_samples.py \
      --model Qwen/Qwen3-0.6B --n-thinking 100 --temperature 1.0
  uv run python sesgo/divergence/collect_divergence_samples.py \
      out/sesgo/divergence/prompt_dataset.json --model Qwen/Qwen3-0.6B \
      --n-thinking 4 --n-items 1
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
from sesgo.shard_output_paths import apply_shard, shard_out_dir  # noqa: E402


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


def select_representative_ambiguous(
    dataset: SesgoPromptDataset, n_items: int
) -> SesgoPromptDataset:
    """Keep the first ``n_items`` representative AMBIGUOUS prompts (deterministic).

    DIVERGENCE samples ONE prompt deeply, so we narrow the full grid to a single
    representative ambiguous item: ambiguous context (gold == abstain) with BOTH
    named identities present, so the [target, other, unknown] split and the
    branching-tree view are well defined. Sorted by ``sample_idx`` for a stable,
    seedless choice; falls back to any ambiguous prompt if none carry both names.
    """
    ambiguous = [s for s in dataset.samples if s.context_condition == "ambig"]
    complete = [s for s in ambiguous if s.target_identity and s.other_identity]
    pool = sorted(complete or ambiguous, key=lambda s: s.sample_idx)
    kept = pool[: max(1, n_items)]
    return SesgoPromptDataset(
        dataset_id=dataset.dataset_id,
        config=dataset.config,
        scaffold_ids=dataset.scaffold_ids,
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
        default=100,
        # DIVERGENCE estimates the reasoning DISTRIBUTION on ONE prompt, so it
        # samples DEEP — 100 draws by default to pin down the outcome distribution.
        help="Sampled thinking generations per prompt (default: 100 — deep sampling of one prompt)",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=1,
        # DIVERGENCE is "fork at one position, sampled deeply": ONE prompt, many
        # draws. Keep the smallest representative-ambiguous slice (default 1).
        help="How many representative ambiguous prompts to query (default: 1 — a single prompt)",
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
        "--shard-index", type=int, default=0, help="This box's shard index (0-based)"
    )
    parser.add_argument(
        "--shard-count", type=int, default=1, help="Total shards (1 == full grid)"
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


def _vocab_entropy_summary(dataset: SesgoDataset) -> tuple[float | None, float | None, int]:
    """Pool every draw's per-generation vocab entropy and report mean/std/count.

    Each item's ``thinking.vocab_entropies`` holds ONE mean next-token entropy per
    CoT draw; pooled across items they describe the per-generation uncertainty
    distribution the study tracks. Returns (mean, population std, n_draws).
    """
    ents: list[float] = []
    for s in dataset.samples:
        if s.thinking is not None:
            ents.extend(s.thinking.vocab_entropies)
    if not ents:
        return None, None, 0
    mean = sum(ents) / len(ents)
    std = (sum((e - mean) ** 2 for e in ents) / len(ents)) ** 0.5
    return mean, std, len(ents)


def log_summary(dataset: SesgoDataset) -> None:
    """Report thinking abstention + role-mix entropy + per-draw vocab entropy.

    Accuracy = fraction of predictions that are UNKNOWN (the ambiguous gold).
    Predictions with no parsed thinking draw are excluded from the thinking rate.
    The mean role-mix entropy is the DIVERGENCE signal (how spread the outcome
    distribution is); the per-draw VOCAB entropy mean/spread is the headline
    per-generation uncertainty the study now also tracks.
    """
    nt = [s.correct_non_thinking for s in dataset.samples if s.predicted_non_thinking is not None]
    th = [s.correct_thinking for s in dataset.samples if s.predicted_thinking is not None]
    nt_acc = sum(nt) / len(nt) if nt else None
    th_acc = sum(th) / len(th) if th else None
    mean_ent, n_ent = _mean_thinking_entropy(dataset)
    ve_mean, ve_std, ve_n = _vocab_entropy_summary(dataset)

    log_section("summary (accuracy = fraction predicted UNKNOWN)")
    log(f"  samples: {len(dataset.samples)}")
    log(f"  non-thinking abstention: {_fmt(nt_acc):>6} (n={len(nt)})")
    log(f"  thinking abstention:     {_fmt(th_acc):>6} (n={len(th)})")
    ent_str = f"{mean_ent:.3f}" if mean_ent is not None else "n/a"
    log(f"  mean role-mix entropy:   {ent_str:>6} nats (n={n_ent})")
    if ve_mean is not None:
        log(f"  per-draw vocab entropy:  {ve_mean:.3f} +/- {ve_std:.3f} nats (n_draws={ve_n})")
    else:
        log("  per-draw vocab entropy:  n/a (no draws captured)")


def main() -> None:
    """Load divergence prompts, query the model, and persist the SesgoDataset."""
    args = parse_args()
    log_header(f"COLLECT DIVERGENCE SAMPLES ({args.model})")

    # Stride the raw json before deserializing when subsampling (fast path).
    prompt_dataset = load_prompt_dataset(args.prompt_dataset, args.subsample)
    # DIVERGENCE samples ONE prompt deeply: narrow to the representative ambiguous
    # slice BEFORE sharding so every box agrees on the same item set.
    prompt_dataset = select_representative_ambiguous(prompt_dataset, args.n_items)
    prompt_dataset = apply_shard(prompt_dataset, args.shard_index, args.shard_count)
    chosen = ", ".join(f"idx={s.sample_idx} qid={s.question_id}" for s in prompt_dataset.samples)
    log(f"[collect] selected {len(prompt_dataset.samples)} ambiguous prompt(s): {chosen}")

    # Already subsampled at load, so the querier runs over all loaded prompts.
    # DIVERGENCE reads out FOUR levels per prompt: the 3-option teacher-forced
    # non-thinking softmax, the 2-option forced choice (target vs other, no
    # UNKNOWN), the single greedy-thinking baseline (deterministic reasoning), and
    # the MANY sampled thinking draws (the selection/divergence distribution). The
    # greedy NON-thinking decode is the only extra generation we skip (do_greedy).
    config = SesgoQueryConfig(
        do_non_thinking=True,
        do_two_option=True,
        do_greedy_thinking=True,
        do_thinking=True,
        n_thinking_samples=args.n_thinking,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_greedy=False,
        subsample=1.0,
        batch_size=args.batch_size,
    )

    # Resolve the FINAL output path up front so the periodic checkpoint writes the
    # SAME file: a crash leaves a valid partial response_samples.json to resume from.
    bare_model = args.model.split("/")[-1]
    out_dir = shard_out_dir(
        args.out_dir, "divergence", bare_model, args.shard_index, args.shard_count
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
