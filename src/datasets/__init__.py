"""Dataset generation for risk assessment.

Subpackages:
- ``mental_risk/``: MentalRiskES corpus loader + the baseline generation
  scripts that turn it into prompt datasets.
- ``prompt/``: risk prompt construction (``risk_*`` builders) plus the shared
  formatting-variation helpers.
- ``risk/``: querying a risk prompt dataset through a model into a ``RiskDataset``.
- ``other/``: parametric scenario datasets (``generate.py`` / phrasings /
  templates), parameterised by a planning horizon with phrasing-group splits.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.
"""

from ..common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
