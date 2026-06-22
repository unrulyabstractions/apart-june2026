"""PHASE 2 of the SHARDED forking-paths pipeline: fork ONE position shard.

Run-by-path driver, run on each of N fleet boxes. Loads the selected item +
``base_path.json`` (the shared plan decoded in phase 1), takes this box's slice of
positions ``[t for t in range(P) if t % N == shard_index]``, and samples
continuations for ONLY those positions in one batched decode (``fork_plan_positions``).
ONLY shard 0 also computes the prior o_0 (``resample_prior``). Writes a partial
``ForkingShard`` the local merge driver reassembles into a full trajectory.

Output: out/sesgo/forking/<MODEL>/forking_shard_<k>_of_<N>.json (a ForkingShard).

Usage:
  uv run python sesgo/forking/collect_forking_shard.py --model Qwen/Qwen3-14B \
      --shard-index 0 --num-shards 5 --n-prior 50 --max-new-tokens 768
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiment.shard_output_paths import shard_out_dir  # noqa: E402
from src.common.file_io import load_json, save_json_atomic  # noqa: E402
from src.common.logging import log, log_header  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.dynamics.forking_paths import (  # noqa: E402
    ForkingShard,
    SerializedBranchPlan,
    fork_plan_positions,
    resample_prior,
    serialized_to_branch_plan,
)
from src.inference.backends import ModelBackend  # noqa: E402
from src.ternary_choice import TernaryChoiceRunner  # noqa: E402

from experiment.forking.forking_item_io import load_selected_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI args for one shard's forking."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name")
    p.add_argument("--shard-index", type=int, required=True, help="this box's shard index k (0-based)")
    p.add_argument("--num-shards", type=int, required=True, help="total shard count N")
    p.add_argument("--n-prior", type=int, default=50, help="full-resample draws for o_0 (shard 0 only)")
    p.add_argument("--max-new-tokens", type=int, default=768, help="tokens per continuation rollout")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    return p.parse_args()


def main() -> None:
    """Fork this box's position slice and persist the partial ForkingShard."""
    args = parse_args()
    k, n = args.shard_index, args.num_shards
    log_header(f"COLLECT FORKING SHARD {k}/{n} ({args.model})")

    out_dir = shard_out_dir(args.out_dir, "forking", args.model.split("/")[-1], 0, 1)
    sample, outcome_set = load_selected_sample(out_dir / "selected_item.json")
    plan = serialized_to_branch_plan(
        SerializedBranchPlan.from_dict(load_json(out_dir / "base_path.json"))
    )

    n_positions = len(plan.rows_per_position)
    position_indices = [t for t in range(n_positions) if t % n == k]
    log(f"[shard{k}] {len(position_indices)}/{n_positions} positions: {position_indices[:8]}...")

    runner = TernaryChoiceRunner(model_name=args.model, backend=ModelBackend.HUGGINGFACE)
    dump_dir = out_dir / f"forking_positions_shard_{k}_of_{n}"
    with P(f"fork_shard_{k}"):
        positions = fork_plan_positions(
            runner, plan, sample, outcome_set, position_indices,
            args.max_new_tokens, args.temperature, dump_dir=dump_dir,
        )

    # Only shard 0 pays the prior's N-draw cost; every other shard leaves it empty.
    prior = []
    if k == 0:
        with P(f"prior_shard_{k}"):
            prior = resample_prior(
                runner, sample, outcome_set, args.n_prior,
                args.max_new_tokens, args.temperature,
            )
        log(f"[shard{k}] o_0={[round(x,2) for x in prior]}")

    shard = ForkingShard(
        shard_index=k, num_shards=n, model=runner.model_name,
        item_question_id=sample.question_id, prompt_text=sample.text,
        base_path_text=plan.base_path_text, base_token_texts=plan.base_token_texts,
        prompt_token_count=plan.prompt_token_count, outcome_labels=outcome_set.labels,
        prior_histogram=prior, positions=positions,
    )
    out_path = out_dir / f"forking_shard_{k}_of_{n}.json"
    save_json_atomic(shard.to_dict(), out_path)
    log(f"[shard{k}] wrote {out_path} ({len(positions)} positions)")


if __name__ == "__main__":
    main()
