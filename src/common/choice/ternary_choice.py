"""Ternary (3-way) choice result.

A TernaryChoice records the three divergent conditional logprobs the model
assigned to three option-label tokens (under teacher forcing), the raw model
logits of those same tokens (from the one shared predicting row), and derives a
probability distribution over the options via a 3-way softmax. This is the
N=3 generalization of SimpleBinaryChoice: where binary compares two logprobs
at the fork, ternary softmaxes over three.

Kept as a clean BaseSchema (labels + three logprobs + three logits are the only
identity fields — no heavy tensors), so it serializes/roundtrips cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base_schema import BaseSchema

# Reuse the project's numerically-stable primitives rather than re-deriving a
# softmax/entropy here (see src/common/CLAUDE.md: search before writing utils).
from ..math import normalize_log_probs, probs_to_logprobs, shannon_entropy


@dataclass
class TernaryChoice(BaseSchema):
    """Model's 3-way preference over three single-token-distinct labels.

    Attributes:
        labels:   The three option labels, e.g. ("0", "1", "2").
        logprobs: Conditional logprobs of each label's option-label token at
            the position where the three forced continuations first diverge.
        logits:   RAW (pre-softmax) model logit of each label's first-divergent
            token, read from the SAME shared predicting row. Because all three
            forced continuations share the prefix up to divergence, that
            full-vocab row is identical across the three trajectories, so these
            logits are directly comparable. logprobs == log_softmax(full row)
            indexed at each token; logits are that full row's raw values.
    """

    labels: tuple[str, str, str]
    logprobs: tuple[float, float, float]
    logits: tuple[float, float, float]

    @property
    def probs(self) -> tuple[float, float, float]:
        """3-way softmax over the divergent logprobs (sums to 1)."""
        # normalize_log_probs uses the logsumexp trick for stability.
        p = normalize_log_probs(list(self.logprobs))
        return (p[0], p[1], p[2])

    @property
    def choice_idx(self) -> int:
        """Argmax option index, or -1 if the top probability is tied."""
        p = self.probs
        top = max(p)
        # A tie means the max is not unique → no decisive choice.
        if p.count(top) > 1:
            return -1
        return p.index(top)

    @property
    def chosen_label(self) -> str | None:
        """Label of the chosen option, or None on a tie."""
        idx = self.choice_idx
        if idx == -1:
            return None
        return self.labels[idx]

    @property
    def entropy(self) -> float:
        """Shannon entropy (nats) of the normalized 3-way distribution.

        shannon_entropy takes logprobs, so convert the normalized probs back
        to logprobs first (consistent with the rest of src.common.math).
        """
        return float(shannon_entropy(probs_to_logprobs(list(self.probs))))
