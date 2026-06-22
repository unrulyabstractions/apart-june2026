"""Reassemble N position shards into one full-length, real-index ``ForkingTrajectory``.

The merge step of the sharded forking pipeline: validate that the shards agree on
model / item / num_shards, then reassemble EVERY base-path index 0..P-1 by its REAL
position (never compacting), so a 4-of-5-shard fleet still yields a correctly-indexed
figure. Any index no shard covered is filled with an explicit GAP sentinel
``ForkPosition`` (prior-valued o_0 histogram = zero drift, no alternates, gap-marked
token) so downstream change-point τ / fork-token indices stay aligned to
``base_token_texts`` — the padded indices are warned LOUDLY. If shard 0 (the prior
o_0) is absent the merge RAISES rather than silently emitting an empty drift
baseline. Returns the trajectory plus a flat list of human-readable warnings.
"""

from __future__ import annotations

from .forking_outcome_set import ForkOutcomeSet
from .forking_path_types import ForkingTrajectory, ForkPosition
from .forking_shard_record import ForkingShard

# Visible marker prepended to a padded position's token text so the gap shows in
# the figure's token strip (no shard measured this base-path token).
GAP_TOKEN_PREFIX = "⌀gap:"


def _gap_position(index: int, base_token_text: str, prior: list[float]) -> ForkPosition:
    """A sentinel position for an index no shard covered (prior o_0 => zero drift).

    Carries the REAL ``index`` and the real base-path token (gap-marked) so the
    token strip stays aligned; its ``outcome_histogram`` is the prior o_0 so the
    drift series reads L2(o_0, o_0)=0 (an honest "no signal" plateau, not a spike),
    and its empty ``alternates`` give zero hazard / zero Wilson n at the gap.
    """
    return ForkPosition(
        position=index,
        base_token_id=-1,
        base_token_text=f"{GAP_TOKEN_PREFIX}{base_token_text}",
        alternates=[],
        outcome_histogram=list(prior),
    )


def _duplicate_warnings(covered: list[ForkPosition]) -> tuple[dict[int, ForkPosition], list[str]]:
    """Index the covered positions; LOUDLY flag any index >1 shard claimed (last wins)."""
    by_index: dict[int, ForkPosition] = {}
    duplicated: list[int] = []
    for p in covered:
        if p.position in by_index:
            duplicated.append(p.position)
        by_index[p.position] = p
    warnings: list[str] = []
    if duplicated:
        warnings.append(f"DUPLICATE positions (covered by >1 shard, kept last): {sorted(set(duplicated))}")
    return by_index, warnings


def _reassemble_full_length(
    by_index: dict[int, ForkPosition], base_token_texts: list[str], prior: list[float]
) -> tuple[list[ForkPosition], list[str]]:
    """Rebuild positions 0..P-1 by REAL index, padding missing indices with gaps."""
    p_max = max(by_index) if by_index else -1
    n_positions = max(len(base_token_texts), p_max + 1)
    positions: list[ForkPosition] = []
    padded: list[int] = []
    for t in range(n_positions):
        if t in by_index:
            positions.append(by_index[t])
        else:
            tok = base_token_texts[t] if t < len(base_token_texts) else ""
            positions.append(_gap_position(t, tok, prior))
            padded.append(t)
    warnings = [f"PADDED missing positions with gap sentinels (real index preserved): {padded}"] if padded else []
    return positions, warnings


def _agreement_warnings(shards: list[ForkingShard], ref: ForkingShard) -> list[str]:
    """LOUDLY flag any shard disagreeing with the reference on model / item / N."""
    warnings: list[str] = []
    for s in shards:
        if s.num_shards != ref.num_shards:
            warnings.append(f"num_shards mismatch: shard {s.shard_index} has {s.num_shards} != {ref.num_shards}")
        if s.model != ref.model:
            warnings.append(f"model mismatch: shard {s.shard_index} model={s.model} != {ref.model}")
        if s.item_question_id != ref.item_question_id:
            warnings.append(f"item mismatch: shard {s.shard_index} item={s.item_question_id} != {ref.item_question_id}")
    return warnings


def _require_prior(shards: list[ForkingShard]) -> list[float]:
    """Return shard 0's prior o_0, or RAISE — a missing prior corrupts every drift."""
    prior_shard = next((s for s in shards if s.shard_index == 0), None)
    if prior_shard is None or not prior_shard.prior_histogram:
        raise ValueError(
            "shard 0 (prior o_0) MISSING — refusing to merge: the prior is the drift "
            "baseline for every position; rerun shard 0 before merging."
        )
    return prior_shard.prior_histogram


def merge_forking_shards(
    shards: list[ForkingShard],
) -> tuple[ForkingTrajectory, list[str]]:
    """Merge shards into one full-length, real-index ``ForkingTrajectory`` + warnings."""
    ref = shards[0]
    prior = _require_prior(shards)  # raises LOUDLY if shard 0 / prior is absent

    covered = [p for s in shards for p in s.positions]
    by_index, dup_warnings = _duplicate_warnings(covered)
    positions, pad_warnings = _reassemble_full_length(by_index, ref.base_token_texts, prior)
    warnings = _agreement_warnings(shards, ref) + dup_warnings + pad_warnings

    final = positions[-1].outcome_histogram if positions else prior
    trajectory = ForkingTrajectory(
        item_question_id=ref.item_question_id,
        model=ref.model,
        outcome_set=ForkOutcomeSet(labels=list(ref.outcome_labels)),
        prompt_text=ref.prompt_text,
        base_path_text=ref.base_path_text,
        base_token_texts=ref.base_token_texts,
        prompt_token_count=ref.prompt_token_count,
        prior_histogram=prior,
        final_histogram=final,
        positions=positions,
    )
    return trajectory, warnings
