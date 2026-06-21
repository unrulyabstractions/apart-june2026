"""Index a GeometryDataset into balanced (scaffold vs no-scaffold) pairs.

Pairing rule (per the geometry spec): group samples by ``question_id`` — each id
appears EXACTLY twice, once with ``has_scaffold=False`` and once with
``has_scaffold=True`` — and split the two members on ``has_scaffold``. We pair by
question_id, NOT scaffold_id (which is the constant 'interpretive_direction'
whenever present, so useless as a key). The pair carries its shared
``context_condition`` (constant within a pair) so the test driver can filter to
the ambiguous items where UNKNOWN is the unbiased answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.datasets.sesgo_eval import GeometrySample


@dataclass
class ContrastivePair:
    """The two members of one question_id: with vs without the scaffold."""

    question_id: str
    context_condition: str  # "ambig" | "disambig" (constant within the pair)
    no_scaffold: GeometrySample
    scaffold: GeometrySample


def build_pairs(samples: list[GeometrySample]) -> list[ContrastivePair]:
    """Group samples by question_id into scaffold/no-scaffold pairs.

    Skips any question_id that does not have exactly one scaffold and one
    no-scaffold member (defensive — the balanced dataset always has both).
    """
    by_qid: dict[str, list[GeometrySample]] = {}
    for s in samples:
        by_qid.setdefault(s.question_id, []).append(s)

    pairs: list[ContrastivePair] = []
    for qid, members in by_qid.items():
        scaffold = next((m for m in members if m.has_scaffold), None)
        no_scaffold = next((m for m in members if not m.has_scaffold), None)
        if scaffold is None or no_scaffold is None:
            continue
        pairs.append(
            ContrastivePair(
                question_id=qid,
                context_condition=no_scaffold.context_condition,
                no_scaffold=no_scaffold,
                scaffold=scaffold,
            )
        )
    return pairs
