"""Shared style for the stability figures: palette, titles, condition labels.

ONE place for the colourblind-safe Okabe-Ito hues, the bold-title/plain-subtitle
helper, the figure saver, and the plain-English names for the two context
conditions and the two answer-reading modes. Every stability plot imports from
here so the family reads as one consistent, jargon-free set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Colourblind-safe (Okabe-Ito): one hue per context condition, reused everywhere.
AMBIG_BLUE = "#0072B2"     # ambiguous question (no answer is stated)
DISAMBIG_ORANGE = "#E69F00"  # clear question (the answer is stated)
MEAN_RED = "#D55E00"       # mean / reference line
ZONE_GREY = "#BBBBBB"      # de-emphasised reference shading

# Plain-English names for the context condition, used in legends and axis labels.
COND_COLOR: dict[str, str] = {"ambig": AMBIG_BLUE, "disambig": DISAMBIG_ORANGE}
COND_LABEL: dict[str, str] = {
    "ambig": "Ambiguous question (no clear answer)",
    "disambig": "Clear question (answer is stated)",
}

# Plain-English names for the two ways the model's answer is read off.
READOUT_NAME: dict[bool, str] = {
    False: "Three options (with 'unknown')",
    True: "Forced two-way choice (no 'unknown')",
}


def titled(ax, title: str, how_to_read: str) -> None:
    """Bold plain-sentence title over an italic 'how to read this' subtitle."""
    ax.set_title(f"{title}\n", fontsize=12.5, fontweight="bold", pad=24)
    ax.text(0.5, 1.012, how_to_read, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=9.5, color="#444444", style="italic")


def save_figure(fig, out_path: Path) -> Path:
    """Persist a figure publication-clean: tight bbox, 150 dpi, then close it."""
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
