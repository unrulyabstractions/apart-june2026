"""Shared palette + tiny drawing primitives for the forking-paths figures.

Presentation-only (colors, the outcome-band order, the change-point heat colormap,
a token-text strip, and a saver); all numbers come from the captured trajectory /
analysis. Kept separate from the plot bodies so the driver and any panel module
import the same look without a cycle, mirroring the divergence study's style.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Outcome band order for the stacked area (matches ForkOutcomeSet.labels order).
OUTCOME_COLORS = {
    "target": "#D55E00",      # Okabe-Ito vermilion
    "other": "#F0E442",       # Okabe-Ito yellow (CVD-distinct from the vermilion target band)
    "unknown": "#0072B2",     # blue
    "unparseable": "#999999", # grey catch-all
}
# Change-point posterior heatmap: low p(tau) yellow -> high p(tau) red (paper).
TAU_CMAP = LinearSegmentedColormap.from_list("tau_yellow_red", ["#FFF7BC", "#FEC44F", "#D7301F"])
REF = "#555555"
SUBTITLE = "forking-paths O_t dynamics (arXiv:2412.07961 / arXiv:2601.06116)"


def save_fig(fig, path):
    """Save tight at publication dpi and close — never leak a figure handle."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def titled(ax, title: str, sub: str | None = None) -> None:
    """Bold title with the paper-nodding subtitle, padded clear of the data."""
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=20)
    ax.text(0.5, 1.01, sub or SUBTITLE, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, color=REF, style="italic")


def token_strip(ax, tokens, tau_posterior, fork_idx) -> None:
    """Render base-path tokens as text, background-colored by p(tau=t|y).

    The forking token (fork_idx) gets a bold red box; the others are shaded on the
    yellow->red heatmap. Prompt tokens are not passed here (they stay uncolored).
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = max(len(tokens), 1)
    peak = max(tau_posterior) if tau_posterior else 1.0
    for i, tok in enumerate(tokens):
        x = (i + 0.5) / n
        p = (tau_posterior[i] / peak) if (peak > 0 and i < len(tau_posterior)) else 0.0
        is_fork = i == fork_idx
        ax.text(
            x, 0.5, tok.replace("\n", "\\n").strip() or "·",
            ha="center", va="center", fontsize=7, rotation=90,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=TAU_CMAP(p),
                edgecolor="red" if is_fork else "none",
                linewidth=1.8 if is_fork else 0.0,
            ),
        )
