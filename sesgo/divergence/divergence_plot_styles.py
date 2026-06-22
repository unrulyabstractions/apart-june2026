"""Shared style constants + tiny drawing primitives for the divergence plots.

Kept separate from the plot bodies so both the panel module and the driver can
import the palette / role order / strip helper without a cycle. Everything here is
presentation-only (colors, a jittered strip, a titled-axes helper); the numbers
come from src.common.math.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from src.common.math import bootstrap_ci
from sesgo.common.plain_language_labels import ROLE_LABEL

# Canonical role order for every length-3 vector in a SesgoThinking readout.
ROLES = ("target", "other", "unknown")
# Plain-language tick labels for the three answer outcomes (short, two-line).
ROLE_TICKS = {
    "target": "Picks the\nstereotyped group",
    "other": "Picks the\nother group",
    "unknown": "Abstains\n('unknown')",
}
# Gold one-hot for the ambiguous SESGO context: always abstain ("safe answer").
GOLD_UNKNOWN = [0.0, 0.0, 1.0]
# Colorblind-safe per-role palette (Okabe-Ito): target / other / unknown.
ROLE_COLORS = {"target": "#D55E00", "other": "#E69F00", "unknown": "#0072B2"}
ACCENT = "#CC79A7"  # mean / deviation accent
REF = "#555555"     # reference / max lines
# The four ways the model was asked, in escalating "effort" order, plain-language
# two-line labels for crowded x-axes, plus stable Okabe-Ito colors.
READOUTS = ("non_thinking", "non_thinking_2opt", "greedy_thinking", "thinking")
READOUT_LABELS = {
    "non_thinking": "Without thinking\n(answers directly)",
    "non_thinking_2opt": "Forced two-way choice\n(no 'unknown')",
    "greedy_thinking": "With thinking\n(reasons first)",
    "thinking": "Free-form thinking\n(many tries)",
}
READOUT_COLORS = {
    "non_thinking": "#56B4E9",
    "non_thinking_2opt": "#009E73",
    "greedy_thinking": "#E69F00",
    "thinking": "#CC79A7",
}
BREAKDOWN_AXES = ("bias_category", "question_polarity", "language")
# Plain-language axis names for the breakdown figures (no pipeline field names).
AXIS_LABEL = {
    "bias_category": "social-group category",
    "question_polarity": "question wording",
    "language": "language",
}
LN3, LN2 = float(np.log(3)), float(np.log(2))


def role_tick(role: str) -> str:
    """Short two-line plain-language tick for one answer outcome."""
    return ROLE_TICKS.get(role, ROLE_LABEL.get(role, role))


def save_fig(fig, path):
    """Save tight at publication dpi and close — never leak a figure handle."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def titled(ax, title: str) -> None:
    """Short bold axes title, padded clear of the data. No subtitle."""
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)


def boot_band(ax, values, *, vertical: bool, color: str = ACCENT) -> tuple[float, float]:
    """Shade the bootstrap 95% CI of the mean and return (mean, ci_halfspan).

    The shaded span makes the mean's uncertainty explicit on a strip/lollipop where
    a bare mean line would hide the small-n spread. NaN (empty) -> no band, (0, 0).
    """
    m, lo, hi = bootstrap_ci(list(values))
    if np.isnan(m):
        return 0.0, 0.0
    span = ax.axvspan if vertical else ax.axhspan
    span(lo, hi, color=color, alpha=0.16, lw=0, zorder=0)
    return m, max(m - lo, hi - m)


def jittered_strip(ax, xs, *, y0: float, color: str, jitter: float = 0.12) -> None:
    """Draw values as a horizontal jittered strip with a faint density curve."""
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return
    rng = np.random.default_rng(0)
    jit = (rng.random(arr.size) - 0.5) * jitter
    if arr.size >= 3 and np.unique(arr).size >= 2:  # density only if it means something
        lo, hi = float(arr.min()), float(arr.max())
        pad = max(1e-3, (hi - lo) * 0.15)
        grid = np.linspace(lo - pad, hi + pad, 200)
        try:
            dens = gaussian_kde(arr)(grid)
            dens = 0.32 * dens / dens.max()
            ax.fill_between(grid, y0, y0 + dens, color=color, alpha=0.18, lw=0)
        except np.linalg.LinAlgError:
            pass
    ax.scatter(arr, np.full(arr.size, y0) + jit, s=64, color=color,
               edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3)
