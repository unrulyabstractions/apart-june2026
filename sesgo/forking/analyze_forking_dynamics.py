"""Analyze a captured {O_t} trajectory: change point, states, diversity, survival.

Run-by-path driver, STAGE 4 of the forking-paths study. Loads the
ForkingTrajectory written by collect_forking_rollouts.py and runs the full
analysis: it reduces {O_t} to the univariate semantic-drift series, runs Bayesian
multiple-change-point detection to localize the forking token, computes the three
dynamic states (pull / drift / potential) plus the per-position diversity series,
and the survival curve. Writes one forking_analysis.json for the plotting driver.

Usage:
  uv run python sesgo/forking/analyze_forking_dynamics.py
  uv run python sesgo/forking/analyze_forking_dynamics.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from sesgo.shard_output_paths import shard_out_dir  # noqa: E402
from src.common.file_io import save_json_atomic  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.dynamics.forking_paths import (  # noqa: E402
    ForkingTrajectory,
    analyze_forking_trajectory,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for trajectory analysis."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name (path key only)")
    p.add_argument("--n-iter", type=int, default=6000, help="RJ-MCMC iterations")
    p.add_argument("--burn-in", type=int, default=1500, help="RJ-MCMC burn-in")
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    return p.parse_args()


def _log_summary(analysis) -> None:
    """Print the headline change-point / forking-token result."""
    cp = analysis.change_points
    st = analysis.dynamic_states
    log_section("forking analysis summary")
    log(f"  Bayes factor p(m>=1)/p(m=0): {cp.bayes_factor:.2f} (threshold 9)")
    log(f"  forking token (CPD argmax):  idx={cp.forking_token_index} "
        f"(significant={cp.significant})")
    log(f"  most-forking token (Delta):  idx={st.most_forking_index}")
    log(f"  p(m)[0:3]: {[round(v, 2) for v in cp.num_changepoints_posterior[:3]]}")


def main() -> None:
    """Load the trajectory, run the full analysis, and persist it."""
    args = parse_args()
    log_header(f"ANALYZE FORKING DYNAMICS ({args.model})")

    out_dir = shard_out_dir(args.out_dir, "forking", args.model.split("/")[-1], 0, 1)
    traj = ForkingTrajectory.from_json(out_dir / "forking_trajectory.json")
    log(f"[analyze] loaded {len(traj.positions)} positions for item "
        f"{traj.item_question_id[:12]}")

    analysis = analyze_forking_trajectory(traj, n_iter=args.n_iter, burn_in=args.burn_in)
    _log_summary(analysis)

    out_path = out_dir / "forking_analysis.json"
    save_json_atomic(analysis.to_dict(), out_path)
    log(f"[analyze] wrote {out_path}")


if __name__ == "__main__":
    main()
