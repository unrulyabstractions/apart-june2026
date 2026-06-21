"""The 2-option forced-choice SESGO readout (no UNKNOWN; target vs other).

The forced-choice level teacher-forces only the two GROUP option tokens and reads
the binary preference at the shared predicting position. There is no abstention
option, so this isolates which group the model leans toward when pushed to commit
— the bias DIRECTION. The position->role remap (defeats position bias) happens
once in ``from_binary``: the chosen displayed position is decoded through the
prompt's 2-option ``position_labels_2opt`` into a role.

Kept a clean BaseSchema: two length-2 vectors + the picked role.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.common.choice import LabeledSimpleBinaryChoice
from src.common.math import normalize_log_probs
from src.datasets.sesgo import SesgoLabel

# Canonical role order for the two group options every vector follows.
_ROLE_ORDER = (SesgoLabel.OTHER, SesgoLabel.TARGET)


@dataclass
class SesgoTwoOption(BaseSchema):
    """Binary forced-choice readout, vectors ordered [OTHER, TARGET]."""

    prob: list[float]  # 2-way renormalized softmax over the 2 group logprobs
    logprob: list[float]  # conditional logprob per group option token
    picked: SesgoLabel  # role of the higher-probability group (ties -> OTHER)

    @classmethod
    def from_binary(
        cls,
        choice: LabeledSimpleBinaryChoice,
        position_labels: tuple[SesgoLabel, SesgoLabel],
    ) -> "SesgoTwoOption":
        """Remap the binary choice's POSITION-indexed scores into role order.

        position i shows role ``position_labels[i]``; we scatter each position's
        logprob into its role slot ([OTHER, TARGET]), renormalize a 2-way softmax,
        and take the argmax role as the forced-choice pick (ties resolve to OTHER).
        """
        lp_a, lp_b = choice.divergent_logprobs
        logprob = [0.0, 0.0]
        for i, (role, lp) in enumerate(zip(position_labels, (lp_a, lp_b))):
            logprob[_ROLE_ORDER.index(role)] = float(lp)
        prob = normalize_log_probs(logprob)
        picked = _ROLE_ORDER[0] if prob[0] >= prob[1] else _ROLE_ORDER[1]
        return cls(prob=list(prob), logprob=logprob, picked=picked)
