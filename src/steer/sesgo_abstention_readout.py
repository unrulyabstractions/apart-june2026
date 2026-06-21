"""Reconstruct a SESGO 3-option readout from a GeometrySample and score abstention.

The geometry sample stores the fully-rendered ``prompt_text`` (options block +
the trailing choice prefix) and the ``label_style`` markers, but not the
position->role decoding tuple. This dataset uses the canonical, unpermuted role
order (label_style is always 'a)b)c)'), so position->role is the canonical
(OTHER, TARGET, UNKNOWN). We split the trailing choice prefix off ``prompt_text``
(``choose3`` re-appends it) and rebuild the exact (prompt, choice_prefix, labels)
triple the original querier used, then remap scores through ``from_ternary``.

Abstention = the model preferring the UNKNOWN ("not enough information") option:
on AMBIGUOUS items UNKNOWN is the unbiased gold, so raising its probability under
+v is the causal effect we test. We report both the teacher-forced UNKNOWN
probability and a hard abstain flag (teacher-forced argmax == UNKNOWN).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.choice import TernaryChoice
from src.datasets.prompt.sesgo_prompt_localization import sesgo_choice_prefix
from src.datasets.sesgo import SesgoLabel
from src.datasets.sesgo_eval import GeometrySample
from src.datasets.sesgo_eval.sesgo_non_thinking import SesgoNonThinking

# This dataset emits only the canonical, unpermuted role ordering (label_style is
# always 'a)b)c)'), so displayed position i shows this role.
_CANONICAL_ROLES = (SesgoLabel.OTHER, SesgoLabel.TARGET, SesgoLabel.UNKNOWN)
_MARKERS = ("a)", "b)", "c)")
_UNKNOWN_SLOT = _CANONICAL_ROLES.index(SesgoLabel.UNKNOWN)


@dataclass
class AbstentionReadout:
    """The reconstructed choose3 inputs plus where to read UNKNOWN mass."""

    prompt: str  # prompt_text with the trailing choice prefix stripped
    choice_prefix: str  # the trailing prefix choose3 will re-append
    labels: tuple[str, str, str]  # option markers, e.g. ("a)","b)","c)")


def build_readout(sample: GeometrySample) -> AbstentionReadout:
    """Recover the (prompt, choice_prefix, labels) triple for ``choose3``."""
    prefix = sesgo_choice_prefix(sample.language)
    text = sample.prompt_text
    prompt = text[: -len(prefix)] if prefix and text.endswith(prefix) else text
    return AbstentionReadout(prompt=prompt, choice_prefix=prefix, labels=_MARKERS)


def unknown_probability(choice: TernaryChoice) -> float:
    """Teacher-forced probability mass on the UNKNOWN option (the abstain choice).

    ``choice`` is position-indexed; with the canonical role order the UNKNOWN
    option sits at the fixed canonical slot, so ``from_ternary`` lands its mass in
    that slot of ``prob``.
    """
    nt = SesgoNonThinking.from_ternary(choice, _CANONICAL_ROLES)
    return float(nt.prob[_UNKNOWN_SLOT])


def is_abstained(choice: TernaryChoice) -> bool:
    """True iff the teacher-forced argmax role is UNKNOWN (a hard abstention)."""
    nt = SesgoNonThinking.from_ternary(choice, _CANONICAL_ROLES)
    return nt.predicted == SesgoLabel.UNKNOWN
