"""The intervention axis for the mental_risk experiment: question FRAMINGS.

The SESGO analogue of this file (sesgo/scaffolds.py) lists debiasing *scaffolds*
— preambles that nudge an ambiguous item toward "unknown". Mental-risk has no
ambiguous-vs-unknown axis: every subject has a continuous gold risk. The
intervention that plays the scaffold's structural role here is the FRAMING — the
way the risk question is posed (at_risk_of / suffering / safe / intervene). Each
framing reframes the same transcript into a different yes/no question, and the
selection study asks which framing best tracks the gold risk.

The concrete framings themselves live in src/datasets/prompt/risk_framing.py (the
generic, bilingual RiskFraming content); this top-level module only re-exports the
canonical ordered set beside the run-by-path drivers, exactly as sesgo/scaffolds.py
re-exports its Scaffold set. Note the asymmetry vs SESGO: there is NO no-op
"baseline" framing — every framing is a real intervention — so the selection
study ranks framings against gold rather than against a no-scaffold baseline.
"""

from __future__ import annotations

from src.datasets.prompt import RISK_FRAMINGS, RiskFraming

# Canonical order in which the grid crosses framing conditions (id stability for
# the selection plots). Pulled straight from the generic content module.
DEFAULT_FRAMINGS: list[RiskFraming] = list(RISK_FRAMINGS)


def get_risk_framings() -> list[RiskFraming]:
    """All concrete mental_risk question framings, in canonical grid order."""
    return DEFAULT_FRAMINGS.copy()


def framing_keys() -> list[str]:
    """The framing keys, in canonical order (the selection study's conditions)."""
    return [f.key for f in DEFAULT_FRAMINGS]
