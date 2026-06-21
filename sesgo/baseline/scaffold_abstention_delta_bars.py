"""Does each debiasing scaffold improve abstention OVER the no-scaffold baseline?

ONE narrative full-data SESGO figure. On AMBIGUOUS questions the only correct
answer is to abstain ('unknown'); per bias category and readout we draw each
scaffold's CHANGE in that abstention rate vs running with NO scaffold. Diverging
horizontal bars: RIGHT (green) = the scaffold raises correct abstention, LEFT
(orange) = it lowers it. Whiskers are Newcombe 95% CIs (the principled interval
for a difference of two independent proportions, built on the shared Wilson core);
the per-arm n is in each column title. Rows = without/with thinking; columns =
Classism / Racism / Xenophobia / Gender. Run by path; writes the PNG under plots/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.base_schema import BaseSchema  # noqa: E402
from src.common.math import wilson_interval  # noqa: E402
from src.datasets.sesgo import SesgoLabel  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample  # noqa: E402

from sesgo.common.plain_language_labels import (  # noqa: E402
    CATEGORY_LABEL,
    CATEGORY_ORDER,
    READOUT_LABEL,
    SCAFFOLD_LABEL,
    SCAFFOLD_ORDER,
    normalize_scaffold,
)

# The two readouts full_data carries -> (sample attribute, plain row title).
_READOUTS = (
    ("correct_non_thinking", READOUT_LABEL["non_thinking"]),
    ("correct_greedy_thinking", READOUT_LABEL["greedy_thinking"]),
)
_SCAFFOLDS = SCAFFOLD_ORDER[1:]  # everything except the no-scaffold baseline
_HELP_COLOR = "#009E73"  # Okabe-Ito green  -> scaffold helps (bar points right)
_HURT_COLOR = "#D55E00"  # Okabe-Ito vermdn -> scaffold hurts (bar points left)


@dataclass
class DeltaBar(BaseSchema):
    """One scaffold's change in ambiguous-abstention rate vs the no-scaffold arm."""

    scaffold_id: str
    delta: float  # scaffold rate minus baseline rate, in [-1, 1]
    ci_lo: float  # Newcombe lower bound on the difference
    ci_hi: float  # Newcombe upper bound on the difference
    n_baseline: int
    n_scaffold: int


def _rate(samples: list[SesgoSample], attr: str) -> tuple[int, int]:
    """``(successes, n)`` correct-abstention over usable (parsed) samples."""
    flags = [getattr(s, attr) for s in samples]
    usable = [f for f in flags if f is not None]
    return sum(bool(f) for f in usable), len(usable)


def _newcombe_diff(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float, float]:
    """Newcombe 95% CI for proportion difference p2 - p1: square-and-add each
    arm's Wilson interval (sensible at small n and p near 0/1)."""
    p1, l1, u1 = wilson_interval(s1, n1)
    p2, l2, u2 = wilson_interval(s2, n2)
    diff = p2 - p1
    lo = diff - np.hypot(p2 - l2, u1 - p1)
    hi = diff + np.hypot(u2 - p2, p1 - l1)
    return diff, lo, hi


def _category_deltas(ambig: list[SesgoSample], attr: str, cat: str) -> list[DeltaBar]:
    """Per-scaffold abstention change vs baseline for one bias category."""
    cells = [s for s in ambig if s.bias_category == cat]
    base = [s for s in cells if normalize_scaffold(s.scaffold_id) is None]
    bs, bn = _rate(base, attr)
    bars: list[DeltaBar] = []
    for scaffold in _SCAFFOLDS:
        arm = [s for s in cells if normalize_scaffold(s.scaffold_id) == scaffold]
        ss, sn = _rate(arm, attr)
        diff, lo, hi = _newcombe_diff(bs, bn, ss, sn)
        bars.append(DeltaBar(scaffold, diff, lo, hi, bn, sn))
    return bars


def _panel(ax, bars: list[DeltaBar]) -> None:
    """Horizontal diverging delta bars (top scaffold first) with CI + n labels."""
    y = np.arange(len(bars))[::-1]  # first scaffold at the top of the panel
    for yi, bar in zip(y, bars):
        color = _HELP_COLOR if bar.delta >= 0 else _HURT_COLOR
        ax.barh(yi, bar.delta * 100, height=0.62, color=color, zorder=3)
        ax.errorbar(bar.delta * 100, yi,
                    xerr=[[(bar.delta - bar.ci_lo) * 100], [(bar.ci_hi - bar.delta) * 100]],
                    fmt="none", ecolor="#222222", elinewidth=1.1, capsize=3, zorder=4)
        # Headline change in percentage points, just past the bar end / CI whisker.
        end = bar.ci_hi if bar.delta >= 0 else bar.ci_lo
        ha = "left" if bar.delta >= 0 else "right"
        pad = 1.4 if bar.delta >= 0 else -1.4
        ax.text(end * 100 + pad, yi, f"{bar.delta * 100:+.0f} pts",
                ha=ha, va="center", fontsize=8.6, fontweight="bold")
    ax.axvline(0, color="#444444", lw=1.1, zorder=2)  # the 'no change' line
    ax.set_yticks(y)
    ax.set_ylim(-0.6, len(bars) - 0.4)
    ax.set_xlim(-45, 45)
    ax.set_xticks([-40, -20, 0, 20, 40])
    ax.grid(axis="x", color="#dddddd", lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_scaffold_deltas(ambig: list[SesgoSample], model: str, out_path: Path) -> Path:
    """Faceted diverging delta figure: readout (rows) x bias category (cols)."""
    cats = [c for c in CATEGORY_ORDER if any(s.bias_category == c for s in ambig)]
    nrow, ncol = len(_READOUTS), len(cats)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.3 * nrow),
                             sharex=True, squeeze=False)
    # _panel sets yticks to arange[::-1], paired positionally with these labels in
    # bar (data) order -- so label i lines up with bars[i]. Keep them un-reversed.
    ytick_labels = [SCAFFOLD_LABEL[s] for s in _SCAFFOLDS]
    for ri, (attr, row_title) in enumerate(_READOUTS):
        for ci, cat in enumerate(cats):
            ax = axes[ri][ci]
            bars = _category_deltas(ambig, attr, cat)
            _panel(ax, bars)
            n_per_arm = bars[0].n_scaffold if bars else 0
            ax.set_yticklabels(ytick_labels if ci == 0 else [""] * len(bars), fontsize=9)
            if ri == 0:
                ax.set_title(f"{CATEGORY_LABEL.get(cat, cat)}\n(~{n_per_arm} items/scaffold)",
                             fontsize=11.5, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(row_title, fontsize=11, fontweight="bold")
            if ri == nrow - 1:
                ax.set_xlabel("Change in correct-abstention rate\n"
                              "vs no scaffold (percentage points)", fontsize=9)
    fig.suptitle(
        f"Each debiasing scaffold mostly RAISES how often {model} correctly says "
        f"'unknown' on ambiguous questions\n"
        + fill("How to read this: each bar is one scaffold's abstention rate minus the "
               "no-scaffold rate. Bars to the RIGHT (green) mean the scaffold improves "
               "correct abstention; to the LEFT (orange) mean it hurts. Whiskers are "
               "Newcombe 95% confidence intervals.", width=118),
        fontsize=12.5, fontweight="bold",
    )
    # Footnote: the line crossing zero is the only honest 'no effect' reference.
    fig.text(0.5, 0.005, "Bars whose 95% interval crosses the central 0-line are not "
             "statistically distinguishable from no change.", ha="center", va="bottom",
             fontsize=8.5, color="#555555", style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Load full_data response_samples.json, keep ambiguous items, render the PNG."""
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path("out/sesgo/full_data/Qwen3-0.6B/response_samples.json"))
    dataset = SesgoDataset.from_json(src)
    scored = [s for s in dataset.samples if s.non_thinking is not None]
    ambig = [s for s in scored if s.gold_label is SesgoLabel.UNKNOWN]
    plots_dir = src.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plot_scaffold_deltas(ambig, dataset.model_name,
                               plots_dir / "scaffold_abstention_delta.png")
    print(f"wrote {out}  (ambiguous n={len(ambig)})")


if __name__ == "__main__":
    main()
