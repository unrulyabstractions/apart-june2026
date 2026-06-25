"""Stage 1-3 forking-paths capture for ONE prompt, BATCHED through the backend.

Pipeline for a single ambiguous SESGO prompt:
  1. greedily decode the base thinking path and read each position's top-K
     alternate tokens (forking_top_k_tokens),
  2. for every (position t, alternate token w) build the forced prefix
     prompt + x*_{<t} + w and sample S continuations,
  3. parse each continuation to an outcome and build o_{t,w} / o_t (Eqs. 1-2).

EVERY (t, w) branch is sampled in ONE batched ``continue_from_text_batch`` call
(expanded S times) by ``fork_plan_positions`` — here called over ALL positions so
single-box output is byte-for-byte the per-position forker's. The same forker
takes a position SUBSET on each sharded fleet box. The prior o_0 is a separate
N-sample full-resample batch (``resample_prior``) from the bare prompt.
"""

from __future__ import annotations

from pathlib import Path

from src.datasets.prompt import SesgoPromptSample
from src.inference import ModelRunner

from .forking_branch_plan import BranchPlan, build_branch_plan, build_branch_plan_from_text
from .forking_outcome_set import ForkOutcomeSet
from .forking_path_types import ForkingTrajectory, ForkPosition
from .forking_plan_forker import fork_plan_positions
from .forking_prior_resample import resample_prior


def strided_positions(n_positions: int, stride: int) -> list[int]:
    """Every ``stride``-th base-path position, always including the last (committed
    answer). ``stride<=1`` returns all positions (the paper-default full sweep)."""
    if n_positions <= 0:
        return []
    if stride <= 1:
        return list(range(n_positions))
    idxs = list(range(0, n_positions, stride))
    if idxs[-1] != n_positions - 1:
        idxs.append(n_positions - 1)
    return idxs


def capture_forking_trajectory(
    runner: ModelRunner,
    sample: SesgoPromptSample,
    outcome_set: ForkOutcomeSet | None = None,
    n_samples: int = 40,
    n_prior: int = 60,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    near_window: int = 0,
    base_max_new_tokens: int = 256,
    max_positions: int = 0,
    position_stride: int = 1,
    shared_base_text: str | None = None,
    dump_dir: Path | None = None,
) -> ForkingTrajectory:
    """Capture the full {O_t} series for one prompt (greedy base path + branches).

    ``n_samples`` continuations per (t, w); ``near_window`` adds extra samples to
    the positions nearest the highest-entropy base token (more samples near
    suspected change points, per the paper). ``max_positions`` (0 == all) caps how
    many leading base-path tokens are branched (the local-pilot cost knob).
    ``position_stride`` (>=1) forks only every k-th base-path position (the dominant
    sampling cost is ``positions x alternates x n_samples``, and most positions
    barely move O_t); the final base-path position is always forked so the committed
    answer's histogram is captured.
    When ``dump_dir`` is given, EVERY position's RAW rollout texts (+ parsed label
    and token info) are written incrementally to ``dump_dir/pos_<NNN>.json`` so a
    crash mid-run keeps every completed position auditable.
    Returns a serializable ForkingTrajectory the analysis driver consumes.
    """
    outcome_set = outcome_set or ForkOutcomeSet()
    if shared_base_text is not None:
        # SHARED-trajectory mode: every model forks the SAME externally-supplied base path
        # (e.g. Qwen3.5-27B's chain of thought), teacher-forced under THIS model's own
        # prompt templating. No re-decode — we measure how each model's outcome distribution
        # evolves along the identical reasoning, so the dynamics are directly comparable.
        text = sample.text if isinstance(sample.text, str) else "\n".join(sample.text)
        templated = runner.apply_chat_template(text)
        plan = build_branch_plan_from_text(
            runner, templated, shared_base_text, near_window, n_samples, max_positions
        )
    else:
        plan = build_branch_plan(
            runner, sample, near_window, n_samples, base_max_new_tokens, max_positions
        )

    # Fork the chosen positions in one batched decode (single-box default == every
    # position; position_stride>1 subsamples them, always keeping the last so the
    # committed-answer histogram is present).
    positions: list[ForkPosition] = fork_plan_positions(
        runner, plan, sample, outcome_set,
        strided_positions(len(plan.rows_per_position), position_stride),
        max_new_tokens, temperature, dump_dir=dump_dir,
    )

    prior = resample_prior(
        runner, sample, outcome_set, n_prior, max_new_tokens, temperature
    )
    final = positions[-1].outcome_histogram if positions else prior
    return ForkingTrajectory(
        item_question_id=sample.question_id,
        model=runner.model_name,
        outcome_set=outcome_set,
        prompt_text=sample.text,
        base_path_text=plan.base_path_text,
        base_token_texts=plan.base_token_texts,
        prompt_token_count=plan.prompt_token_count,
        prior_histogram=prior,
        final_histogram=final,
        positions=positions,
    )
