"""BinaryFork: a pairwise comparison between two branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base_schema import BaseSchema

# The `analysis` subpackage only exists in some forks of this codebase
# (e.g. "temporal-manifolds"). Import ForkAnalysis when available, but fall
# back gracefully so this module imports cleanly in forks that lack it
# (e.g. "queering-nlp-bias").
try:
    from .analysis.metrics import ForkAnalysis
except Exception:  # pragma: no cover - optional dependency
    ForkAnalysis = Any  # type: ignore[assignment,misc]


@dataclass
class BinaryFork(BaseSchema):
    """A pairwise comparison between two branches at a divergence point.

    Attributes:
        next_token_ids: The two token IDs being compared (branch_a, branch_b)
        next_token_logprobs: Log-probabilities for each token
        fork_idx: Index in the parent tree's forks tuple
        group_idx: Which groups/arms the two branches belong to (group_a, group_b).
            ``arm_idx`` is a backward-compatible alias for this field.
        vocab_logits: Full logits over vocabulary at this fork position (for raw
            logit extraction)
        analysis: Optional per-fork analysis metrics (when the analysis
            subpackage is available)
    """

    next_token_ids: tuple[int, int]
    next_token_logprobs: tuple[float, float]
    fork_idx: int | None = None  # Index in parent tree's forks tuple
    group_idx: tuple[int, int] | None = None
    arm_idx: tuple[int, int] | None = None
    vocab_logits: list[float] | None = None
    analysis: ForkAnalysis | None = None

    def __post_init__(self) -> None:
        # Reconcile the `group_idx` / `arm_idx` aliases so callers in either
        # fork can read or write whichever name they expect.
        if self.group_idx is None and self.arm_idx is not None:
            self.group_idx = self.arm_idx
        elif self.arm_idx is None and self.group_idx is not None:
            self.arm_idx = self.group_idx

    @property
    def next_token_logits(self) -> tuple[float, float] | None:
        """Raw logits for the two tokens, extracted from vocab_logits."""
        if self.vocab_logits is None:
            return None
        id_a, id_b = self.next_token_ids
        return (self.vocab_logits[id_a], self.vocab_logits[id_b])
