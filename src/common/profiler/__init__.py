"""Simple profiling utilities.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.

Usage:
    from src.common.profiler import P
    from src.common import P  # also works (via re-export)

    with P("section_name"):
        work()

    P.report()

    # Or use the @profile decorator
    from src.common.profiler import profile

    @profile  # uses function name
    def load_data():
        return load()

    @profile("custom_name")  # custom identifier
    def other_func():
        pass

    # Track memory usage around a function
    from src.common.profiler import track_memory

    @track_memory
    def cleanup():
        ...
"""

from src.common.auto_export import auto_export

# Explicit imports needed: 'profile' is excluded by auto_export (stdlib
# collision). The profiling_* modules are the superset (memory tracking), so we
# bind the canonical P/Profiler/profile/track_memory from them.
from .profiling_timer import P, Profiler
from .profiling_decorators import profile, track_memory

__all__ = auto_export(__file__, __name__, globals())

# Add back names excluded or guaranteed by the explicit imports above.
for _name in ("P", "Profiler", "profile", "track_memory"):
    if _name not in __all__:
        __all__.append(_name)
