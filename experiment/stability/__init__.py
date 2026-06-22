"""SESGO stability study: consistency of answers across superficial format variants.

Collects one answer per rendered prompt across the 18 label-style x role-order
variants of each item (gold fixed), then measures how often the prediction is
invariant, splits sensitivity by format axis, and contrasts the smallest vs the
biggest model within each family.
"""

from src.common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
