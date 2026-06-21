"""Presentation-only helpers for the SESGO steering-test figure.

Kept apart from the driver so the palette / save helper / single-panel drawer can
be imported without re-deriving any numbers — every value plotted comes straight
out of ``steering_test.json`` (the held-out causal sweep). Colours are the same
colourblind-safe Okabe-Ito family the rest of the SESGO plots use. The abstention
metric is a proportion, so its markers carry Wilson 95% whiskers; the mean-
probability metric is a mean with no per-item spread on disk, so it is drawn
without whiskers (an honest choice, not a missing bar).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.common.math.confidence_intervals import wilson_err
from src.steer import SteeringTestResult

# Okabe-Ito: the held-out curve is the headline; the fitted-items curve is the echo.
HELDOUT_COLOR = "#0072B2"   # held-out questions (the causal claim)
FITTED_COLOR = "#009E73"    # questions used to build the steering direction
TARGET_COLOR = "#D55E00"    # the scaffold's own abstention (the behaviour to match)
BASELINE_COLOR = "#444444"  # alpha = 0, no steering applied
PUSH_AWAY_FILL = "#CC79A7"  # negative-alpha control (steering the opposite way)

# Plain-language curve labels reused by the panel drawer and the shared legend.
HELDOUT_LABEL = "Held-out questions (never used to build the direction)"
FITTED_LABEL = "Questions used to build the direction"


def save_fig(fig, path):
    """Save tight at publication dpi on white, then close the handle."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _sweep_xy(sweep, attr: str):
    """(alphas, metric, n_items) arrays for a sweep, sorted by alpha."""
    pts = sorted(sweep, key=lambda p: p.alpha)
    return (
        np.array([p.alpha for p in pts], dtype=float),
        np.array([getattr(p, attr) for p in pts], dtype=float),
        np.array([p.n_items for p in pts], dtype=int),
    )


def _wilson_band(rates: np.ndarray, ns: np.ndarray):
    """Asymmetric (below, above) Wilson 95% offsets for a rate-vs-alpha curve."""
    errs = [wilson_err(round(r * n), n) for r, n in zip(rates, ns)]
    below = np.array([e[0] for e in errs])
    above = np.array([e[1] for e in errs])
    # Clamp away tiny negative float artefacts at p=0/1 (matplotlib rejects yerr<0).
    return np.clip(np.vstack([below, above]), 0.0, None)


def _draw_curve(ax, x, y, ns, *, color, label, lw, ms, z, metric):
    """One sweep curve; Wilson whiskers only when the metric is a proportion."""
    if metric == "abstain_rate":
        ax.errorbar(
            x, y, yerr=_wilson_band(y, ns), fmt="-o", color=color, lw=lw, ms=ms,
            capsize=3, elinewidth=1.0, zorder=z, label=label,
        )
    else:
        ax.plot(x, y, "-o", color=color, lw=lw, ms=ms, zorder=z, label=label)


def draw_panel(ax, result: SteeringTestResult, metric: str) -> None:
    """One model's abstention-vs-steering panel: held-out + fitted curves.

    ``metric`` is ``abstain_rate`` or ``mean_unknown_prob``. The no-steering point,
    the scaffold's own abstention, and the push-the-opposite-way control region are
    all marked so the panel is self-explanatory.
    """
    short = result.model.split("/")[-1]
    alphas = sorted(p.alpha for p in result.sweep)
    left = alphas[0] - 0.4  # a touch past the most-negative alpha for a clean edge

    # Shade the negative-alpha region: there we steer the OPPOSITE way (the control).
    ax.axvspan(left, 0.0, color=PUSH_AWAY_FILL, alpha=0.08, lw=0, zorder=0)
    ax.text(left / 2.0, 0.02, "steered the\nopposite way", ha="center", va="bottom",
            fontsize=8, color="#8a4d6f", style="italic", zorder=1)

    # The scaffold's own abstention: the behaviour steering is trying to reproduce.
    ref = getattr(result.scaffold_reference, metric)
    ax.axhline(ref, color=TARGET_COLOR, ls="--", lw=1.6, zorder=1,
               label=f"Scaffold's own level ({ref:.0%})")

    # No-steering reference at alpha = 0.
    base_pt = next(p for p in result.sweep if p.alpha == 0.0)
    base = getattr(base_pt, metric)
    ax.axhline(base, color=BASELINE_COLOR, ls=":", lw=1.2, zorder=1,
               label=f"No steering ({base:.0%})")
    ax.axvline(0.0, color=BASELINE_COLOR, ls=":", lw=0.9, alpha=0.6, zorder=1)

    # Fitted-items curve first (in-sample echo), then held-out on top (headline).
    if result.train_sweep:
        fx, fy, fn = _sweep_xy(result.train_sweep, metric)
        _draw_curve(ax, fx, fy, fn, color=FITTED_COLOR, label=FITTED_LABEL,
                    lw=1.6, ms=5, z=2, metric=metric)
    hx, hy, hn = _sweep_xy(result.sweep, metric)
    _draw_curve(ax, hx, hy, hn, color=HELDOUT_COLOR, label=HELDOUT_LABEL,
                lw=2.4, ms=7, z=3, metric=metric)

    ax.set_title(f"{short}   ({result.n_ambiguous_test_items} held-out questions)",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Steering strength  (higher = push harder toward the scaffold)",
                  fontsize=9.5)
    ax.set_ylim(-0.03, 1.08)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticklabels([f"{t:.0%}" for t in np.arange(0.0, 1.01, 0.2)])
    ax.set_xticks(hx)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, lw=0.6)
