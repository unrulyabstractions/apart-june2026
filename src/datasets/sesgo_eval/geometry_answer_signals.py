"""Derive the continuous answer-distribution signals for a geometry sample.

The geometry viz wants to colour points by the model's CONFIDENCE / UNCERTAINTY
at the answer, not just its categorical pick. We read those scalars off the
readout's answer distribution (the renormalized softmax over the option roles)
and its raw logits, reusing the shared information-theory helpers in
``src.common.math`` rather than re-deriving entropy / perplexity here:

  top_choice_prob   - probability mass on the argmax role (peakedness)
  top_choice_logit  - raw logit of the argmax role (absolute confidence/scale)
  vocab_entropy     - Shannon entropy (nats) of the answer distribution
  answer_diversity  - Hill D_1 = effective number of roles the mass spreads over
  inv_perplexity    - geometric-mean per-option probability (=exp(mean logprob))

All five are derived from the SAME role distribution so they stay mutually
consistent. ``argmax`` (top option) anchors the prob/logit; the entropy /
diversity / inv-perplexity describe the whole distribution's shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.common.math import (
    inv_perplexity,
    probs_to_logprobs,
    q_diversity,
    shannon_entropy,
)


@dataclass
class GeometryAnswerSignals(BaseSchema):
    """The five continuous answer-distribution scalars for one readout."""

    top_choice_prob: float = 0.0
    top_choice_logit: float = 0.0
    vocab_entropy: float = 0.0  # Shannon entropy (nats) of the answer distribution
    answer_diversity: float = 0.0  # Hill D_1 (effective #roles)
    inv_perplexity: float = 0.0  # geometric-mean per-option probability

    @classmethod
    def from_distribution(
        cls, prob: list[float], logit: list[float] | None = None
    ) -> "GeometryAnswerSignals":
        """Build the five scalars from a role probability vector (+ raw logits).

        ``prob`` is the renormalized softmax over option roles; ``logit`` is the
        matching raw logit vector (mean-centred or raw — only the argmax entry is
        read). All reductions reuse ``src.common.math`` so entropy/perplexity are
        never reimplemented. Empty / degenerate input yields all-zero signals.
        """
        if not prob:
            return cls()
        logprobs = probs_to_logprobs(list(prob))
        top = max(prob)
        top_i = prob.index(top)
        return cls(
            top_choice_prob=float(top),
            top_choice_logit=float(logit[top_i]) if logit else 0.0,
            vocab_entropy=float(shannon_entropy(logprobs)),
            answer_diversity=float(q_diversity(logprobs, 1.0)),
            inv_perplexity=float(inv_perplexity(logprobs)),
        )
