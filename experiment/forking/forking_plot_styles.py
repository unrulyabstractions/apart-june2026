"""Shared palette + tiny drawing primitives for the forking-paths figures.

Presentation-only (colors, the outcome-band order, the forking-jump heat colormap,
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
# Forking-jump heatmap: small jump pale -> large O_t jump red.
JUMP_CMAP = LinearSegmentedColormap.from_list("jump_pale_red", ["#FFF7BC", "#FEC44F", "#D7301F"])
REF = "#555555"  # neutral grey for reference text/edges


def save_fig(fig, path):
    """Save tight at publication dpi and close — never leak a figure handle."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def titled(ax, title: str) -> None:
    """Short bold panel title (minimal-text rule: no subtitle)."""
    ax.set_title(title, fontsize=11, fontweight="bold")


def token_strip(ax, tokens, fork_scores, fork_flags) -> None:
    """Render base-path tokens, highlighting the ones that flip the answer.

    Each token's background is shaded by its O_t jump magnitude (``fork_scores``,
    Delta_t = ||O_t - O_{t-1}||); tokens flagged as forking (``fork_flags``, the
    large-jump tokens) get a bold red box so the reader can see WHICH words flip
    the outcome distribution. Prompt tokens are not passed here.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = max(len(tokens), 1)
    peak = max(fork_scores) if fork_scores else 1.0
    # Long chains of thought (hundreds of tokens) cannot show readable rotated token
    # text, so degrade to a heat bar: each position is a cell shaded by its O_t jump,
    # with a red tick over every forking token. Short chains keep the labelled tokens.
    if n > 90:
        for i in range(n):
            x = i / n
            s = (fork_scores[i] / peak) if (peak > 0 and i < len(fork_scores)) else 0.0
            ax.axvspan(x, (i + 1) / n, ymin=0.2, ymax=0.8, color=JUMP_CMAP(s), lw=0)
            if i < len(fork_flags) and fork_flags[i]:
                ax.plot([(i + 0.5) / n], [0.9], marker="v", color="red", ms=5, clip_on=False)
        return
    for i, tok in enumerate(tokens):
        x = (i + 0.5) / n
        s = (fork_scores[i] / peak) if (peak > 0 and i < len(fork_scores)) else 0.0
        is_fork = i < len(fork_flags) and fork_flags[i]
        ax.text(
            x, 0.5, tok.replace("\n", "\\n").strip() or "·",
            ha="center", va="center", fontsize=7, rotation=90,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=JUMP_CMAP(s),
                edgecolor="red" if is_fork else "none",
                linewidth=1.8 if is_fork else 0.0,
            ),
        )
