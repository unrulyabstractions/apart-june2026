"""Uncertainty estimators for plotting: Wilson proportion CIs, SEM, bootstrap CIs.

Small, dependency-light helpers so every plot can carry honest error bars even at
the tiny sample sizes typical of a subsampled probe:

  - ``wilson_interval``  : Wilson score interval for a binomial proportion. Unlike
    the normal (Wald) interval it stays inside [0, 1] and behaves sanely at p≈0/1
    and small n — exactly the regime of per-condition accuracy / flip rates here.
  - ``sem``              : standard error of the mean (std / sqrt(n)), the natural
    error bar for a mean of per-item values.
  - ``bootstrap_ci``     : percentile bootstrap CI for an arbitrary statistic of a
    sample (used for mean shifts / spreads where no closed form is convenient).

All return plain floats / tuples so callers can hand them straight to matplotlib's
asymmetric ``yerr``/``xerr`` (as ``[point - lo, hi - point]``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Two-sided z for a 95% interval; the only constant the Wilson form needs.
_Z_95 = 1.959963984540054


def wilson_interval(
    successes: int, total: int, z: float = _Z_95
) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(p_hat, lo, hi)`` with the point estimate and the lower/upper bounds
    clamped to [0, 1]. ``total == 0`` yields ``(nan, nan, nan)`` so callers can skip
    drawing a bar that has no data behind it.
    """
    if total <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    return (p, max(0.0, center - half), min(1.0, center + half))


def wilson_err(successes: int, total: int) -> tuple[float, float]:
    """Wilson interval as asymmetric ``(below, above)`` offsets for ``yerr``.

    ``(0.0, 0.0)`` when there is no data, so an empty bar gets no whisker.
    """
    p, lo, hi = wilson_interval(successes, total)
    if np.isnan(p):
        return (0.0, 0.0)
    return (p - lo, hi - p)


def sem(values: Sequence[float]) -> float:
    """Standard error of the mean; 0.0 for fewer than two points (no spread)."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of ``statistic`` over ``values``.

    Returns ``(point, lo, hi)``; for an empty sample returns all-nan, and for a
    single value collapses the interval to that value (zero-width, honest n=1).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(arr))
    if arr.size == 1:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.array([statistic(arr[row]) for row in idx])
    half_alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [half_alpha, 1.0 - half_alpha])
    return (point, float(lo), float(hi))


def bootstrap_labelled_ci(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of a STATISTIC over a labelled point cloud.

    Resamples ROWS of ``matrix`` together with their parallel ``labels`` (so each
    draw keeps the row->label pairing) and recomputes ``statistic(rows, labels)``.
    This is the right resampling for geometry statistics with no closed-form
    variance — centroid-shift magnitude, silhouette — where the statistic depends
    on the group structure, not just a flat list of values.

    Returns ``(point, lo, hi)`` with ``point`` the statistic on the full sample.
    Resamples whose statistic is non-finite (e.g. a group vanished) are dropped;
    if every resample is degenerate, or there are fewer than two rows, the interval
    collapses to the point estimate.
    """
    arr = np.asarray(matrix, dtype=float)
    labs = np.asarray(labels, dtype=object)
    point = float(statistic(arr, labs))
    if arr.shape[0] < 2:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.shape[0], size=arr.shape[0])
        val = statistic(arr[idx], labs[idx])
        if np.isfinite(val):
            boots.append(float(val))
    if not boots:
        return (point, point, point)
    half_alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [half_alpha, 1.0 - half_alpha])
    return (point, float(lo), float(hi))
