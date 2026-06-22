"""Format-invariance split by the TYPE of format change, smallest vs biggest model.

Each stability item is shown in 18 superficially different FORMATS that never
change the correct answer: 3 LABEL styles crossed with 6 ROLE orders. Here we
break invariance out by axis — does the answer survive a LABEL-style change, and
does it survive a ROLE-order change — reusing the existing per-bucket flip logic
(``axis_invariance``: other axis held fixed, share of buckets that never moved).

Four within-family models (Qwen 0.6B/32B, Llama 1B/70B); the larger model is the
darker shade so "does scale help?" reads light->dark within each family. Bars
carry a Wilson 95% interval. Run with the project venv:
  .venv/bin/python sesgo/stability/smallest_vs_biggest_format_invariance.py
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))  # repo root -> `src...`
sys.path.insert(0, str(_HERE.parent))  # this dir -> sibling helper modules

from src.common.math import wilson_err  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402
from format_axis_invariance_helpers import (  # noqa: E402
    AXIS_LABEL,
    AXIS_ORDER,
    AxisInvariance,
    axis_invariances,
)
from stability_plot_style import save_figure  # noqa: E402

# Four models: within EACH family the smallest vs the largest, so the scale effect
# is isolated from the family. Grouped Qwen (small->large) then Llama (small->large).
_MODELS = ("Qwen3-0.6B", "Qwen3-32B", "Llama-3.2-1B-Instruct", "Llama-3.1-70B-Instruct")
_MODEL_LABEL = {"Qwen3-0.6B": "Qwen\n0.6B", "Qwen3-32B": "Qwen\n32B",
                "Llama-3.2-1B-Instruct": "Llama\n1B", "Llama-3.1-70B-Instruct": "Llama\n70B"}
# Colour by family; the larger model is the darker shade (Qwen greens, Llama blues;
# Okabe-Ito anchors) so light->dark within a family answers "does scale help?".
_MODEL_COLOR = {"Qwen3-0.6B": "#9bd5c6", "Qwen3-32B": "#009E73",
                "Llama-3.2-1B-Instruct": "#a6cee8", "Llama-3.1-70B-Instruct": "#0072B2"}
_DATA = _HERE.parents[2] / "out" / "sesgo" / "stability"

# Layout geometry (data units): bars within an axis cluster sit this far apart and
# the two clusters are separated by an extra gap.
_BAR_DX = 1.5
_CLUSTER_GAP = 2.2
_Y_TOP = 1.16  # headroom above 100% for value labels


def _bar_with_ci(ax, x: float, inv: AxisInvariance, color: str) -> None:
    """Draw one Wilson-capped bar; tag its rate above and its item count below."""
    below, above = wilson_err(inv.invariant, inv.total)
    below, above = max(0.0, below), max(0.0, above)  # guard tiny float negatives
    ax.bar(x, inv.rate, width=0.92, color=color, edgecolor="white", zorder=2)
    ax.errorbar(x, inv.rate, yerr=[[below], [above]], fmt="none",
                ecolor="#333333", elinewidth=1.4, capsize=4, zorder=3)
    ax.text(x, inv.rate + above + 0.02, f"{inv.rate:.0%}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#222222")
    ax.text(x, -0.085, f"n={inv.total}", ha="center", va="top",
            fontsize=8, color="#777777")


def _positions() -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """Map (axis, model) -> x and axis -> cluster-centre x."""
    span = (len(_MODELS) - 1) * _BAR_DX
    pos: dict[tuple[str, str], float] = {}
    centres: dict[str, float] = {}
    for ai, axis in enumerate(AXIS_ORDER):
        base = ai * (span + _CLUSTER_GAP)
        centres[axis] = base + span / 2
        for mi, model in enumerate(_MODELS):
            pos[(axis, model)] = base + mi * _BAR_DX
    return pos, centres


def plot_axis_invariance(rows: list[AxisInvariance], out_path: pathlib.Path) -> pathlib.Path:
    """Grouped bars: per-model invariance to a LABEL-style vs a ROLE-order change."""
    fig, ax = plt.subplots(figsize=(9.5, 5.6), layout="constrained")
    pos, centres = _positions()
    by_key = {(r.axis, r.model): r for r in rows}
    for (axis, model), x in pos.items():
        _bar_with_ci(ax, x, by_key[(axis, model)], _MODEL_COLOR[model])

    ax.set_xticks([pos[(a, m)] for a in AXIS_ORDER for m in _MODELS])
    ax.set_xticklabels([_MODEL_LABEL[m] for _ in AXIS_ORDER for m in _MODELS], fontsize=9)
    # Format-axis headers + hairline bracket, parked in the headroom above 100%.
    for axis, cx in centres.items():
        ax.text(cx, 1.075, AXIS_LABEL[axis], ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#222222")
        ax.plot([cx - _BAR_DX / 2, cx + _BAR_DX / 2], [1.06, 1.06],
                color="#999999", lw=1.2, clip_on=False)

    ax.set_ylim(0, _Y_TOP)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Invariance rate", fontsize=11)
    last = centres[AXIS_ORDER[-1]]
    ax.set_xlim(-0.9, last + (len(_MODELS) - 1) * _BAR_DX / 2 + 0.7)
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, out_path)


def _restrict_to(dataset: SesgoDataset, question_ids: set[str]) -> SesgoDataset:
    """A copy of `dataset` keeping only samples whose question_id is in the set."""
    kept = [s for s in dataset.samples if s.question_id in question_ids]
    return dataclasses.replace(dataset, samples=kept)


def _common_question_ids(datasets: dict[str, SesgoDataset]) -> set[str]:
    """The question_ids present in EVERY model, so all comparisons share items."""
    per_model = (set(s.question_id for s in d.samples) for d in datasets.values())
    return set.intersection(*per_model)


def main() -> pathlib.Path:
    """Render the comparison PNG over the question_id intersection of all models.

    The models were collected on slightly different item subsamples, so the
    smallest-vs-biggest contrast must be restricted to the shared items first —
    otherwise 1B-vs-70B and 0.6B-vs-32B would be measured on non-identical sets.
    """
    datasets = {
        m: SesgoDataset.from_json(_DATA / m / "response_samples.json") for m in _MODELS
    }
    common = _common_question_ids(datasets)
    print(f"shared question_ids across all {len(_MODELS)} models: {len(common)}")
    rows: list[AxisInvariance] = []
    for model, dataset in datasets.items():
        rows += axis_invariances(_restrict_to(dataset, common), model)
    out_path = _DATA / "smallest_vs_biggest_format_invariance.png"
    return plot_axis_invariance(rows, out_path)


if __name__ == "__main__":
    print(main())
