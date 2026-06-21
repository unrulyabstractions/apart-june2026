"""Smallest vs biggest model: how often does only reformatting change the answer?

A clean head-to-head for the paper. Each item is shown in 18 superficially
different FORMATS (option-label style x the order the groups are listed) that
never change the correct answer. An item is FORMAT-INVARIANT when the model gives
the very same answer across all 18 — its choice was independent of formatting.

We REUSE the existing stability consistency metric (``consistency_set``: per-item
modal-answer fraction over a format group); modal fraction == 1.0 means the answer
never moved. The format-invariance RATE is the share of such items, a binomial
proportion carrying an honest Wilson 95% interval + n. Bars are grouped by context
condition (ambiguous vs clear question), the one sub-dimension the stability
collection varies; it covers the Racism category only, so that scope is stated on
the figure. Taller bar = more robust. Run by path with the project venv:
  .venv/bin/python sesgo/stability/smallest_vs_biggest_format_invariance.py
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))  # repo root -> `src...`
sys.path.insert(0, str(_HERE.parent))  # this dir -> sibling helper modules

from src.common import BaseSchema  # noqa: E402
from src.common.math import wilson_err  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402
from stability_metrics_helpers import CONDITIONS, consistency_set  # noqa: E402
from stability_plot_style import COND_COLOR, COND_LABEL, save_figure, titled  # noqa: E402

# Two models to contrast, smallest first; their plain display names.
_MODELS = ("Qwen3-0.6B", "Llama-3.1-70B-Instruct")
_MODEL_LABEL = {"Qwen3-0.6B": "Smallest model\n(Qwen3 0.6B)",
                "Llama-3.1-70B-Instruct": "Biggest model\n(Llama 3.1 70B)"}
_DATA = _HERE.parents[2] / "out" / "sesgo" / "stability"


@dataclass
class InvarianceRate(BaseSchema):
    """Format-invariance tally for one (model, condition): invariant / total items."""

    model: str
    condition: str
    invariant: int  # items whose answer never changed across the 18 formats
    total: int  # items with >=2 comparable formats (a defined consistency)

    @property
    def rate(self) -> float:
        """Share of items that were fully format-invariant (0 when no items)."""
        return self.invariant / self.total if self.total else 0.0


def _invariance_rate(dataset: SesgoDataset, condition: str, model: str) -> InvarianceRate:
    """Count fully format-invariant items (3-option modal fraction == 1.0)."""
    cons = consistency_set(dataset, condition, two_opt=False).consistency
    invariant = sum(1 for c in cons if c >= 1.0)
    return InvarianceRate(model=model, condition=condition,
                          invariant=invariant, total=len(cons))


# Layout geometry (in data units along x): bars within a condition cluster sit
# this far apart, and the two clusters are separated by an extra gap.
_BAR_DX = 1.6
_CLUSTER_GAP = 2.4
_Y_TOP = 1.34  # headroom above 100% for the value + condition-header rows


def _bar_with_ci(ax, x: float, rate: InvarianceRate, color: str) -> None:
    """Draw one Wilson-capped bar and annotate its rate + item count beneath it."""
    below, above = wilson_err(rate.invariant, rate.total)
    below, above = max(0.0, below), max(0.0, above)  # guard tiny float negatives
    ax.bar(x, rate.rate, width=0.92, color=color, edgecolor="white", zorder=2)
    ax.errorbar(x, rate.rate, yerr=[[below], [above]], fmt="none",
                ecolor="#333333", elinewidth=1.6, capsize=5, zorder=3)
    # Rate label above the upper whisker; "(no bar)" hint when the bar is flat.
    tag = f"{rate.rate:.0%}" + ("  (no bar)" if rate.invariant == 0 else "")
    ax.text(x, rate.rate + above + 0.03, tag, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#222222")
    ax.text(x, -0.105, f"n={rate.total}", ha="center", va="top",
            fontsize=9, color="#666666")


def _bar_positions() -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """Map (condition, model) -> x and condition -> cluster-centre x."""
    n = len(_MODELS)
    span = (n - 1) * _BAR_DX
    pos: dict[tuple[str, str], float] = {}
    centres: dict[str, float] = {}
    for ci, cond in enumerate(CONDITIONS):
        base = ci * (span + _CLUSTER_GAP)
        centres[cond] = base + span / 2
        for mi, model in enumerate(_MODELS):
            pos[(cond, model)] = base + mi * _BAR_DX
    return pos, centres


def plot_invariance(rates: list[InvarianceRate], out_path: pathlib.Path) -> pathlib.Path:
    """Grouped bars: per-model format-invariance rate, split by context condition."""
    fig, ax = plt.subplots(figsize=(9.2, 6.6), layout="constrained")
    pos, centres = _bar_positions()
    by_key = {(r.condition, r.model): r for r in rates}
    for (cond, model), x in pos.items():
        _bar_with_ci(ax, x, by_key[(cond, model)], COND_COLOR[cond])

    ax.set_xticks([pos[(c, m)] for c in CONDITIONS for m in _MODELS])
    ax.set_xticklabels([_MODEL_LABEL[m] for _ in CONDITIONS for m in _MODELS],
                       fontsize=9.5)
    # Condition headers + hairline bracket, parked in the headroom above 100%.
    for cond, cx in centres.items():
        ax.text(cx, 1.235, COND_LABEL[cond], ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=COND_COLOR[cond])
        ax.plot([cx - _BAR_DX / 2, cx + _BAR_DX / 2], [1.215, 1.215],
                color=COND_COLOR[cond], lw=1.4, clip_on=False)

    ax.set_ylim(0, _Y_TOP)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Format-invariance rate\n(share of items whose answer never changed)",
                  fontsize=10.5)
    x_arrow = -1.15  # "higher = more robust" arrow, parked left of the first bar
    ax.annotate("more robust", xy=(x_arrow, 1.0), xytext=(x_arrow, 0.3),
                rotation=90, ha="center", va="center", fontsize=10,
                color="#2E7D32", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=2.2))
    last = centres[CONDITIONS[-1]]
    ax.set_xlim(-1.7, last + (len(_MODELS) - 1) * _BAR_DX / 2 + 0.7)
    titled(ax,
           "The biggest model keeps its answer when only the wording changes; "
           "the smallest one flips",
           "taller bar = the answer stayed the same across 18 reworded formats "
           "(more robust). Items are the Racism category.")
    return save_figure(fig, out_path)


def main() -> pathlib.Path:
    """Load both models, compute the rates, render the comparison PNG."""
    rates: list[InvarianceRate] = []
    for model in _MODELS:
        dataset = SesgoDataset.from_json(_DATA / model / "response_samples.json")
        rates += [_invariance_rate(dataset, c, model) for c in CONDITIONS]
    out_path = _DATA / "smallest_vs_biggest_format_invariance.png"
    return plot_invariance(rates, out_path)


if __name__ == "__main__":
    print(main())
