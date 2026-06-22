"""Serialize / reload a ``BranchPlan`` so the base path is decoded ONCE per item.

The sharded forking fleet decodes the greedy base path + enumerates every branch
prefix on ONE box (``decode_forking_base_path``), then ships the plan to N shard
boxes that each fork only their slice — none of them re-decode the base path. The
plan's ``rows_per_position`` is a nested ``list[list[tuple]]`` (illegal across a
serialization boundary), so it is FLATTENED into a 1D ``list[ForkingBranchRow]``
(one BaseSchema per (position, alternate) branch) and regrouped on reload.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema

from .forking_branch_plan import BranchPlan
from .forking_top_k_tokens import AltToken


@dataclass
class ForkingBranchRow(BaseSchema):
    """ONE (position t, alternate token w) branch: its forced prefix + sample budget.

    ``is_base`` flags the greedy base-path token w* at this position; ``prefix`` is
    the fully-templated forced-continuation string the forker samples from;
    ``n_samples`` is this position's per-alternate continuation budget.
    """

    position: int
    token_id: int
    token_text: str
    token_prob: float
    is_base: bool
    prefix: str
    n_samples: int


@dataclass
class SerializedBranchPlan(BaseSchema):
    """A ``BranchPlan`` flattened to a 1D list of branch rows (serialization-safe).

    ``rows`` holds every (position, alternate) branch as a flat ``ForkingBranchRow``
    list; ``serialized_to_branch_plan`` regroups them by ``position`` back into the
    plan's ``rows_per_position``. The base-path strip fields mirror ``BranchPlan``.
    """

    base_path_text: str
    base_token_ids: list[int]
    base_token_texts: list[str]
    prompt_token_count: int
    rows: list[ForkingBranchRow] = field(default_factory=list)


def branch_plan_to_serialized(plan: BranchPlan) -> SerializedBranchPlan:
    """Flatten a ``BranchPlan`` into a 1D ``SerializedBranchPlan`` for transport."""
    rows: list[ForkingBranchRow] = []
    for t, pos_rows in enumerate(plan.rows_per_position):
        for alt, prefix, n in pos_rows:
            rows.append(
                ForkingBranchRow(
                    position=t,
                    token_id=alt.token_id,
                    token_text=alt.token_text,
                    token_prob=alt.prob,
                    is_base=alt.is_base,
                    prefix=prefix,
                    n_samples=n,
                )
            )
    return SerializedBranchPlan(
        base_path_text=plan.base_path_text,
        base_token_ids=list(plan.base_token_ids),
        base_token_texts=list(plan.base_token_texts),
        prompt_token_count=plan.prompt_token_count,
        rows=rows,
    )


def serialized_to_branch_plan(s: SerializedBranchPlan) -> BranchPlan:
    """Regroup a flat ``SerializedBranchPlan`` back into a position-indexed plan."""
    n_positions = len(s.base_token_ids)
    grouped: list[list[tuple[AltToken, str, int]]] = [[] for _ in range(n_positions)]
    for row in s.rows:
        alt = AltToken(row.token_id, row.token_text, row.token_prob, row.is_base)
        grouped[row.position].append((alt, row.prefix, row.n_samples))
    return BranchPlan(
        base_path_text=s.base_path_text,
        base_token_ids=list(s.base_token_ids),
        base_token_texts=list(s.base_token_texts),
        prompt_token_count=s.prompt_token_count,
        rows_per_position=grouped,
    )
