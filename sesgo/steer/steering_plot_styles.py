"""Presentation-only helpers for the SESGO steering-test figure.

Kept apart from the driver so the palette / save helper / single-panel drawer can
be imported without re-deriving any numbers — every value plotted comes straight
out of ``steering_test.json`` (the held-out causal sweep). Colours are the same
colourblind-safe Okabe-Ito family the rest of the SESGO plots use.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.steer import SteeringTestResult

# Okabe-Ito: TEST (held-out) is the headline curve; TRAIN is the in-sample echo.
TEST_COLOR = "#0072B2"   # held-out test split (the causal claim)
TRAIN_COLOR = "#009E73"  # in-sample train split (generalization check)
SCAFFOLD_COLOR = "#D55E00"  # real-scaffold reference (the target behaviour)
BASELINE_COLOR = "#555555"  # alpha=0 unsteered baseline
CONTROL_FILL = "#CC79A7"    # negative-alpha control region


def save_fig(fig, path):
    """Save tight at publication dpi on white, then close the handle."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _sweep_xy(sweep, attr: str):
    """(alphas, metric) arrays for a sweep, sorted by alpha."""
    pts = sorted(sweep, key=lambda p: p.alpha)
    return (
        np.array([p.alpha for p in pts], dtype=float),
        np.array([getattr(p, attr) for p in pts], dtype=float),
    )


def draw_panel(ax, result: SteeringTestResult, metric: str) -> None:
    """One model's abstention-vs-alpha panel: TEST + TRAIN curves with markers.

    ``metric`` is ``abstain_rate`` or ``mean_unknown_prob``. The alpha=0 baseline,
    the real-scaffold reference, and the negative-alpha control region are all
    marked so the figure is self-explanatory.
    """
    short = result.model.split("/")[-1]

    # Shade the negative-alpha (control) region: +v should raise abstention, -v drop it.
    ax.axvspan(ax.get_xlim()[0], 0.0, color=CONTROL_FILL, alpha=0.07, lw=0, zorder=0)

    # Real-scaffold reference: the behaviour the steering vector is trying to reproduce.
    ref = getattr(result.scaffold_reference, metric)
    ax.axhline(ref, color=SCAFFOLD_COLOR, ls="--", lw=1.6, zorder=1,
               label=f"real scaffold ({ref:.2f})")

    # Unsteered baseline at alpha=0.
    base_pt = next(p for p in result.sweep if p.alpha == 0.0)
    base = getattr(base_pt, metric)
    ax.axhline(base, color=BASELINE_COLOR, ls=":", lw=1.2, zorder=1,
               label=f"unsteered baseline ({base:.2f})")
    ax.axvline(0.0, color=BASELINE_COLOR, ls=":", lw=0.9, alpha=0.6, zorder=1)

    # TRAIN (in-sample) then TEST (held-out) so TEST draws on top.
    if result.train_sweep:
        tx, ty = _sweep_xy(result.train_sweep, metric)
        ax.plot(tx, ty, "-o", color=TRAIN_COLOR, lw=1.6, ms=5, alpha=0.75,
                zorder=2, label="train split (in-sample)")
    sx, sy = _sweep_xy(result.sweep, metric)
    ax.plot(sx, sy, "-o", color=TEST_COLOR, lw=2.4, ms=7, zorder=3,
            label="test split (held-out)")

    ax.set_title(f"{short}  (layer {result.layer}, n_test={result.n_ambiguous_test_items})",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("steering strength  alpha  (x diff-of-means v)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
