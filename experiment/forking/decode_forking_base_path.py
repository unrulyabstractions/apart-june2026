"""PHASE 1 of the SHARDED forking-paths pipeline: decode the base path ONCE.

Run-by-path driver. Greedily decodes the FULL chain-of-thought for the SELECTED
item, enumerates every (position t, alternate token w) branch PREFIX (the
expensive batched forking is NOT done here), and serializes the whole plan to
``<out_dir>/base_path.json``. The N shard boxes each load this plan and fork only
their slice of positions, so the base path is decoded exactly once per item.

Output: out/sesgo/forking/<MODEL>/base_path.json (a SerializedBranchPlan).

Usage:
  uv run python sesgo/forking/decode_forking_base_path.py --model Qwen/Qwen3-14B \
      --base-max-new-tokens 768 --n-samples 50 --near-window 0
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiment.shard_output_paths import shard_out_dir  # noqa: E402
from src.common.file_io import save_json_atomic  # noqa: E402
from src.common.logging import log, log_header  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.dynamics.forking_paths import (  # noqa: E402
    branch_plan_to_serialized,
    build_branch_plan,
)
from src.inference import ModelRunner  # noqa: E402
from src.inference.backends import ModelBackend  # noqa: E402

from experiment.forking.forking_item_io import load_selected_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI args for base-path decode."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name")
    p.add_argument("--n-samples", type=int, default=50, help="per-(t,w) continuation budget recorded in the plan")
    p.add_argument("--near-window", type=int, default=0, help="+50%% samples within this radius of the peak-entropy token")
    p.add_argument("--base-max-new-tokens", type=int, default=768, help="tokens for the greedy base path decode")
    p.add_argument("--thinking", action="store_true", help="decode the base path in THINKING mode (enable_thinking for Qwen3.5 etc.)")
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    return p.parse_args()


def main() -> None:
    """Decode the base path, enumerate branches, and persist the SerializedBranchPlan."""
    args = parse_args()
    log_header(f"DECODE FORKING BASE PATH ({args.model})")

    out_dir = shard_out_dir(args.out_dir, "forking", args.model.split("/")[-1], 0, 1)
    sample, _outcome_set = load_selected_sample(out_dir / "selected_item.json")
    log(f"[base] item idx={sample.sample_idx} q={sample.question_id[:12]}")

    runner = ModelRunner(model_name=args.model, backend=ModelBackend.HUGGINGFACE)
    runner.force_thinking = args.thinking  # enable_thinking for the base-path decode
    log(f"[base] thinking={args.thinking} reasoning_model={runner.is_reasoning_model}")
    with P("build_branch_plan"):
        plan = build_branch_plan(
            runner, sample, args.near_window, args.n_samples,
            base_max_new_tokens=args.base_max_new_tokens, max_positions=0,
        )

    n_positions = len(plan.rows_per_position)
    n_branches = sum(len(rows) for rows in plan.rows_per_position)
    log(f"[base] base path: {n_positions} positions, {n_branches} total branches")

    out_path = out_dir / "base_path.json"
    save_json_atomic(branch_plan_to_serialized(plan).to_dict(), out_path)
    log(f"[base] wrote {out_path}")


if __name__ == "__main__":
    main()
