"""Fork a SUBSET of a BranchPlan's base-path positions in ONE batched decode.

The forking-paths inner loop, factored out of ``forking_path_capture`` so a single
box can fork ALL positions (single-box, ``position_indices == range(P)``) while a
sharded fleet box forks only its slice ``positions[k::N]`` — same code, same
batched ``continue_from_text_batch`` call, identical per-position output. The REAL
base-path index t is preserved on every ``ForkPosition`` (never renumbered from 0),
so a downstream merge can reassemble the shards back into one ordered trajectory.
"""

from __future__ import annotations

from pathlib import Path

from src.datasets.prompt import SesgoPromptSample
from src.inference import ModelRunner

from .forking_branch_plan import BranchPlan
from .forking_outcome_set import ForkOutcomeSet
from .forking_outcome_mapping import rollout_to_outcome_label
from .forking_path_types import AltTokenRollouts, ForkPosition
from .forking_position_dump_writer import build_position_dump, write_position_dump
from .outcome_histogram_builder import conditional_histogram, position_histogram


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


def fork_plan_positions(
    runner: ModelRunner,
    plan: BranchPlan,
    sample: SesgoPromptSample,
    outcome_set: ForkOutcomeSet,
    position_indices: list[int],
    max_new_tokens: int,
    temperature: float,
    dump_dir: Path | None = None,
) -> list[ForkPosition]:
    """Fork ONLY ``position_indices`` of the plan; one batched decode for all.

    Flattens every (position t in ``position_indices``, alternate w) branch prefix
    expanded n_samples times, decodes them in ONE ``continue_from_text_batch`` call,
    slices the flat rollouts back per (position, alternate), and builds a
    ``ForkPosition`` carrying the REAL index t. With ``dump_dir`` set, each
    position's raw rollouts are written atomically the moment it is assembled.
    """
    flat_prompts: list[str] = []
    for t in position_indices:
        for _alt, prefix, n in plan.rows_per_position[t]:
            flat_prompts.extend([prefix] * n)
    flat_rollouts = runner.continue_from_text_batch(
        flat_prompts, max_new_tokens=max_new_tokens, temperature=temperature
    )

    positions: list[ForkPosition] = []
    cursor = 0
    for t in position_indices:
        pos_rows = plan.rows_per_position[t]
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
        if dump_dir is not None:
            write_position_dump(
                dump_dir,
                build_position_dump(
                    t, plan.base_token_ids[t], plan.base_token_texts[t],
                    pos_rows, per_alt_texts, per_alt_labels,
                ),
            )
    return positions
