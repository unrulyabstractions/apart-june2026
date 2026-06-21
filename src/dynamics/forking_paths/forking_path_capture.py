"""Stage 1-3 forking-paths capture for ONE prompt, BATCHED through the backend.

Pipeline for a single ambiguous SESGO prompt:
  1. greedily decode the base thinking path and read each position's top-K
     alternate tokens (forking_top_k_tokens),
  2. for every (position t, alternate token w) build the forced prefix
     prompt + x*_{<t} + w and sample S continuations,
  3. parse each continuation to an outcome and build o_{t,w} / o_t (Eqs. 1-2).

EVERY (t, w) branch is sampled in ONE ``generate_batch`` call (expanded S times),
so vLLM continuous batching saturates the GPU on the cloud box; the HF backend
runs the same batched path locally for the pilot. The prior o_0 is a separate
N-sample full-resample batch from the bare prompt.
"""

from __future__ import annotations

from pathlib import Path

from src.datasets.prompt import SesgoPromptSample
from src.inference import ModelRunner

from .forking_branch_plan import BranchPlan, build_branch_plan
from .forking_outcome_set import ForkOutcomeSet
from .forking_path_types import AltTokenRollouts, ForkingTrajectory, ForkPosition
from .forking_outcome_mapping import rollout_to_outcome_label
from .forking_position_dump_writer import build_position_dump, write_position_dump
from .outcome_histogram_builder import conditional_histogram, position_histogram


def _resample_prior(
    runner: ModelRunner,
    sample: SesgoPromptSample,
    outcome_set: ForkOutcomeSet,
    n_prior: int,
    max_new_tokens: int,
    temperature: float,
) -> list[float]:
    """o_0: full-resample prior from the bare prompt (N independent draws)."""
    rollouts = runner.generate_batch(
        [sample.text] * n_prior, max_new_tokens=max_new_tokens, temperature=temperature
    )
    labels = [rollout_to_outcome_label(r, sample) for r in rollouts]
    return conditional_histogram(labels, outcome_set)


def _branch_position(
    plan_rows: list[tuple],
    per_alt_labels: list[list[str]],
    outcome_set: ForkOutcomeSet,
) -> tuple[list[AltTokenRollouts], list[float]]:
    """Assemble one position's alternate records + its o_t from rollout labels."""
    alt_records: list[AltTokenRollouts] = []
    cond_hists: list[list[float]] = []
    weights: list[float] = []
    for (alt, _prefix, _n), labels in zip(plan_rows, per_alt_labels):
        cond = conditional_histogram(labels, outcome_set)
        alt_records.append(
            AltTokenRollouts(
                token_id=alt.token_id,
                token_text=alt.token_text,
                token_prob=alt.prob,
                is_base_token=alt.is_base,
                rollout_labels=labels,
                conditional_histogram=cond,
            )
        )
        cond_hists.append(cond)
        weights.append(alt.prob)
    o_t = position_histogram(cond_hists, weights, outcome_set)
    return alt_records, o_t


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
    dump_dir: Path | None = None,
) -> ForkingTrajectory:
    """Capture the full {O_t} series for one prompt (greedy base path + branches).

    ``n_samples`` continuations per (t, w); ``near_window`` adds extra samples to
    the positions nearest the highest-entropy base token (more samples near
    suspected change points, per the paper). ``max_positions`` (0 == all) caps how
    many leading base-path tokens are branched (the local-pilot cost knob).
    When ``dump_dir`` is given, EVERY position's RAW rollout texts (+ parsed label
    and token info) are written incrementally to ``dump_dir/pos_<NNN>.json`` so a
    crash mid-run keeps every completed position auditable.
    Returns a serializable ForkingTrajectory the analysis driver consumes.
    """
    outcome_set = outcome_set or ForkOutcomeSet()
    plan: BranchPlan = build_branch_plan(
        runner, sample, near_window, n_samples, base_max_new_tokens, max_positions
    )

    # ONE batched decode over every (t, w) branch, each expanded S times. The
    # prefixes are already fully templated, so use the raw-continuation path
    # (continue_from_text_batch), NOT generate_batch (which re-templates).
    flat_prompts: list[str] = []
    for _pos_rows in plan.rows_per_position:
        for _alt, prefix, n in _pos_rows:
            flat_prompts.extend([prefix] * n)
    flat_rollouts = runner.continue_from_text_batch(
        flat_prompts, max_new_tokens=max_new_tokens, temperature=temperature
    )

    # Slice the flat rollouts back into per-(t, w) groups and build histograms.
    # The RAW per-alternate texts are kept alongside the labels so a dump can map
    # each rollout back to its (position, alternate, sample) for unparseable audits.
    positions: list[ForkPosition] = []
    cursor = 0
    for t, pos_rows in enumerate(plan.rows_per_position):
        per_alt_texts: list[list[str]] = []
        per_alt_labels: list[list[str]] = []
        for _alt, _prefix, n in pos_rows:
            chunk = flat_rollouts[cursor : cursor + n]
            cursor += n
            per_alt_texts.append(list(chunk))
            per_alt_labels.append([rollout_to_outcome_label(r, sample) for r in chunk])
        alt_records, o_t = _branch_position(pos_rows, per_alt_labels, outcome_set)
        positions.append(
            ForkPosition(
                position=t,
                base_token_id=plan.base_token_ids[t],
                base_token_text=plan.base_token_texts[t],
                alternates=alt_records,
                outcome_histogram=o_t,
            )
        )
        # Crash-safe: persist this position's raw dump the moment it's assembled.
        if dump_dir is not None:
            write_position_dump(
                dump_dir,
                build_position_dump(
                    t, plan.base_token_ids[t], plan.base_token_texts[t],
                    pos_rows, per_alt_texts, per_alt_labels,
                ),
            )

    prior = _resample_prior(
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
