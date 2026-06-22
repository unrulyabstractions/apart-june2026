"""Column-major legend ordering so families stay grouped in the figure legend.

Matplotlib fills a multi-column legend ROW-major: with ``ncol`` columns it lays
entries left-to-right across the first row, then the second, and so on. We want
each family clustered in its own column (size-ordered top-to-bottom), so we
pre-permute the size-ordered keys into the order matplotlib needs to RENDER them
column-major. Pad short columns with ``None`` then drop padding at the end so an
uneven final column doesn't shuffle later families across columns.
"""

from __future__ import annotations


def legend_columns(order: list[str], ncol: int) -> list[str]:
    """Reorder size-ordered ``order`` so matplotlib renders it column-major.

    ``order`` is already grouped by family (families contiguous, size-ordered
    within each). We chunk it into ``ncol`` near-equal columns and interleave them
    so matplotlib's row-major fill reproduces the columns. Padding keeps columns
    aligned; trailing ``None`` placeholders are stripped from the result.
    """
    n = len(order)
    rows = -(-n // ncol)  # ceil division: rows per column
    cols = [order[c * rows:(c + 1) * rows] for c in range(ncol)]
    cols = [col + [None] * (rows - len(col)) for col in cols]  # pad to equal height
    interleaved = [cols[c][r] for r in range(rows) for c in range(ncol)]
    return [k for k in interleaved if k is not None]
