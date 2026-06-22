"""Common utilities for experiments (temporal-awareness and bias analysis).

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.

The explicit re-exports below are intentional: auto_export swallows
ImportError, so these statements make failures in load-bearing modules loud
and document the required import order (analysis must come before choice,
which imports from analysis).
"""

from src.common.auto_export import auto_export

# Re-export from subpackages for flat access (e.g., from src.common import SimpleBinaryChoice)
# NOTE: Import order matters! analysis must come before choice (choice imports from analysis)
from .math import *
from .analysis import *
from .choice import *
from .profiler import *
from .time_value import TimeValue, TIME_UNITS, TIME_UNIT_TO_YEARS, DEFAULT_TIME_UNIT

# Position mapping base types (generic, no domain-specific builders)
from .position_info import TokenPositionInfo
from .position_mapping_base import DatasetPositionMappingBase, SamplePositionMappingBase
from .token_positions import *

__all__ = auto_export(__file__, __name__, globals())
