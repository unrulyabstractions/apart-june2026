"""Capture the per-token outcome distribution {O_t} for the SELECTED item.

Run-by-path driver, STAGE 1-3 of the forking-paths study. Loads the item chosen
by select_forking_item.py, greedily decodes its base thinking path, and at EACH
base-path token position samples N continuations per sufficiently-probable
alternate token, parsing each to a categorical outcome to estimate o_{t,w}/o_t
(Eqs. 1-2). EVERY (position, alternate) branch is decoded in ONE batched call
(continue_from_text_batch) so vLLM continuous batching saturates the cloud GPU;
the HuggingFace backend runs the same batched path for the local pilot.

Output: out/sesgo/forking/<MODEL>/forking_trajectory.json (a ForkingTrajectory).

Usage:
  uv run python sesgo/forking/collect_forking_rollouts.py            # full (cloud)
  uv run python sesgo/forking/collect_forking_rollouts.py --n-samples 6 \
      --n-prior 8 --max-new-tokens 96 --model Qwen/Qwen3-0.6B        # tiny pilot
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
from src.dynamics.forking_paths import capture_forking_trajectory  # noqa: E402
from src.inference.backends import ModelBackend  # noqa: E402
from src.ternary_choice import TernaryChoiceRunner  # noqa: E402

from experiment.forking.forking_item_io import load_selected_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI args for rollout capture."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name")
    p.add_argument("--n-samples", type=int, default=40, help="continuations per (t,w) (paper: 30-40)")
    p.add_argument("--n-prior", type=int, default=60, help="full-resample draws for the prior o_0")
    p.add_argument("--near-window", type=int, default=0, help="+50%% samples within this radius of the peak-entropy token")
    p.add_argument("--max-new-tokens", type=int, default=256, help="tokens per continuation rollout")
    p.add_argument("--base-max-new-tokens", type=int, default=512, help="tokens for the greedy base path decode")
    p.add_argument("--max-positions", type=int, default=0, help="cap on branched base-path positions (0 = all; local-pilot knob)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    # Output subdir suffix selecting the scaffold/baseline condition (default: none).
    p.add_argument("--run-tag", default="", help="output subdir suffix (default: none)")
    return p.parse_args()


def main() -> None:
    """Load the selected item, capture {O_t}, and persist the ForkingTrajectory."""
    args = parse_args()
    log_header(f"COLLECT FORKING ROLLOUTS ({args.model})")

    out_dir = shard_out_dir(args.out_dir, "forking", args.model.split("/")[-1] + args.run_tag, 0, 1)
    sample, outcome_set = load_selected_sample(out_dir / "selected_item.json")
    log(f"[collect] item idx={sample.sample_idx} q={sample.question_id[:12]} "
        f"outcomes={outcome_set.labels}")

    # Per-position RAW rollout dump: one pos_<NNN>.json per base-path token,
    # recording every alternate's raw continuation text + parsed label + token
    # info (written incrementally, crash-safe) so each entry maps back to its
    # forking_trajectory position for unparseable / outcome audits.
    dump_dir = out_dir / "forking_positions"

    runner = TernaryChoiceRunner(model_name=args.model, backend=ModelBackend.HUGGINGFACE)
    with P("capture_forking_trajectory"):
        traj = capture_forking_trajectory(
            runner, sample, outcome_set,
            n_samples=args.n_samples, n_prior=args.n_prior,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            near_window=args.near_window,
            base_max_new_tokens=args.base_max_new_tokens,
            max_positions=args.max_positions,
            dump_dir=dump_dir,
        )

    log(f"[collect] base path: {len(traj.positions)} tokens, "
        f"{sum(len(p.alternates) for p in traj.positions)} branches")
    log(f"[collect] o_0={[round(x,2) for x in traj.prior_histogram]} "
        f"o_T={[round(x,2) for x in traj.final_histogram]}")

    out_path = out_dir / "forking_trajectory.json"
    save_json_atomic(traj.to_dict(), out_path)
    log(f"[collect] wrote {out_path}")
    log(f"[collect] wrote {len(traj.positions)} per-position raw dumps to {dump_dir}")


if __name__ == "__main__":
    main()
