"""Rich non-thinking SESGO readout: per-option evidence in canonical role order.

The non-thinking level teacher-forces the three option tokens and reads the
model's scores at the single shared predicting position. This schema keeps every
per-option quantity (prob / logprob / raw logit / mean-centered logit / inverse
perplexity) as a length-3 list in the canonical role order [TARGET, OTHER,
UNKNOWN], so downstream analysis can slice by role without re-joining. The
position->role remap (defeats position bias) happens once in `from_ternary`.

Kept a clean BaseSchema: five length-3 vectors + two scalars + the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from src.common import BaseSchema
from src.common.math import (
    normalize_log_probs,
    probs_to_logprobs,
    q_diversity,
    shannon_entropy,
)
from src.datasets.sesgo import SesgoLabel
from src.common.choice import TernaryChoice

# Canonical role order every vector follows; argmax ties resolve to UNKNOWN.
_ROLE_ORDER = (SesgoLabel.TARGET, SesgoLabel.OTHER, SesgoLabel.UNKNOWN)


@dataclass
class SesgoNonThinking(BaseSchema):
    """Per-option non-thinking evidence, vectors ordered [TARGET, OTHER, UNKNOWN].

    Two complementary readouts, like temporal-manifolds' binary-choice preference
    sample: (1) the per-option-PATH teacher-forced scores below, and (2) a GREEDY
    decode — the option the model actually emits when it answers without reasoning
    (skip-thinking prefill, temperature 0). ``decoding_mismatch`` flags when the
    greedy pick disagrees with the teacher-forced argmax (``predicted``).
    """

    prob: list[float]  # 3-way renormalized softmax over the 3 option logprobs
    logprob: list[float]  # full-vocab conditional logprob per option token
    logit: list[float]  # raw model logit per option token (shared row)
    normalized_logit: list[float]  # mean-centered logits (see from_ternary)
    inv_ppl: list[float]  # inverse single-token perplexity = exp(logprob)
    greedy_label: SesgoLabel | None = None  # role the greedy non-thinking decode chose
    greedy_text: str = ""  # the greedy decoded answer (short)
    decoding_mismatch: bool = False  # greedy pick != teacher-forced argmax

    @property
    def entropy(self) -> float:
        """Shannon entropy (nats) of `prob` (reuses src.common.math)."""
        # shannon_entropy takes logprobs, so convert the probs back first.
        return float(shannon_entropy(probs_to_logprobs(list(self.prob))))

    @property
    def diversity(self) -> float:
        """Hill number D_1 (effective #roles) of `prob`."""
        return float(q_diversity(probs_to_logprobs(list(self.prob)), 1.0))

    @property
    def predicted(self) -> SesgoLabel:
        """Argmax role over `prob`; a non-unique max → UNKNOWN (indecisive)."""
        top = max(self.prob)
        if self.prob.count(top) > 1:
            return SesgoLabel.UNKNOWN
        return _ROLE_ORDER[self.prob.index(top)]

    @classmethod
    def from_ternary(
        cls,
        choice: TernaryChoice,
        position_labels: tuple[SesgoLabel, SesgoLabel, SesgoLabel],
    ) -> SesgoNonThinking:
        """Remap the choice's POSITION-indexed scores into canonical role order.

        The chooser returns scores per displayed position; position i shows role
        ``position_labels[i]``. We scatter each position's (logprob, logit) into
        its role slot, then derive the rest. `prob` is the 3-way renormalized
        softmax of the role-ordered logprobs.

        normalized_logit = logit_i - mean(logits): we mean-CENTER rather than
        softmax the logits because softmax(logits) is mathematically identical
        to `prob` (logits and logprobs differ only by the constant log-partition,
        which softmax cancels), so a softmax form would be redundant. Centering
        keeps the raw logit SCALE/spread (an absolute-confidence signal `prob`
        discards) while removing the arbitrary per-row offset.
        """
        # Scatter position-indexed scores into canonical [TARGET, OTHER, UNKNOWN].
        logprob = [0.0, 0.0, 0.0]
        logit = [0.0, 0.0, 0.0]
        for i, role in enumerate(position_labels):
            slot = _ROLE_ORDER.index(role)
            logprob[slot] = float(choice.logprobs[i])
            logit[slot] = float(choice.logits[i])

        prob = normalize_log_probs(logprob)  # 3-way renormalized softmax
        mean_logit = sum(logit) / len(logit)
        normalized_logit = [x - mean_logit for x in logit]
        inv_ppl = [exp(lp) for lp in logprob]  # full-vocab mass of each token

        return cls(
            prob=list(prob),
            logprob=logprob,
            logit=logit,
            normalized_logit=normalized_logit,
            inv_ppl=inv_ppl,
        )
