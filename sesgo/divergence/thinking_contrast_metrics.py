"""Per-question outcome-mix and abstention helpers for the thinking-contrast plots.

Pure readers over the EXISTING divergence readouts (no new sampling): for each
ambiguous question, pull the answer-outcome distribution and a binary 'abstained?'
flag, under two ways of asking — DIRECT (the 3-option non-thinking answer) and
REASONED (the free-form thinking answer, summarised over draws). Reuses the
SesgoSample readout schema and its argmax 'predicted' role so the abstain logic
lives in one place.
"""

from __future__ import annotations

from src.datasets.sesgo import SesgoLabel
from src.datasets.sesgo_eval import SesgoSample

# The two ways of asking we contrast: (readout attr, plain row/panel title).
WAYS: tuple[tuple[str, str], ...] = (
    ("non_thinking", "Without thinking\n(answers directly)"),
    ("thinking", "With thinking\n(reasons first)"),
)
# Question wording -> (code, plain label, Okabe-Ito bar colour).
WORDINGS: tuple[tuple[str, str, str], ...] = (
    ("nonneg", "Neutral wording", "#56B4E9"),
    ("neg", "Negative wording", "#D55E00"),
)


def _readout(sample: SesgoSample, attr: str):
    """The requested readout object (non_thinking / thinking), or None if absent."""
    return getattr(sample, attr, None)


def mean_outcome_mix(samples: list[SesgoSample], attr: str) -> list[float]:
    """Average answer distribution [stereotyped, other, abstains] over questions.

    Uses each question's role distribution for the chosen way of asking — the
    3-way option probabilities for the direct answer, the per-draw pick fractions
    for the reasoned answer — and averages across questions. Empty -> zeros.
    """
    mixes: list[list[float]] = []
    for s in samples:
        r = _readout(s, attr)
        if r is None:
            continue
        if attr == "thinking":
            if r.sample_size == 0:
                continue
            mixes.append(list(r.mean))
        else:
            mixes.append(list(r.prob))
    if not mixes:
        return [0.0, 0.0, 0.0]
    n = len(mixes)
    return [sum(m[i] for m in mixes) / n for i in range(3)]


def abstains(sample: SesgoSample, attr: str) -> bool | None:
    """Whether this question's answer abstains, for the chosen way of asking.

    Direct: the 3-option argmax role is UNKNOWN. Reasoned: the most common role
    across parsed draws is UNKNOWN. None when the readout is missing / unparsed,
    so it never inflates the denominator.
    """
    r = _readout(sample, attr)
    if r is None:
        return None
    if attr == "thinking" and r.sample_size == 0:
        return None
    return r.predicted is SesgoLabel.UNKNOWN
