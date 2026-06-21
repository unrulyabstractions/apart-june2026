"""Does scaling up make models ABSTAIN more on unanswerable questions?

On AMBIGUOUS SESGO items (only honest answer is "unknown"), `non_thinking.prob =
[stereotyped, other, unknown]` is a soft outcome distribution; we average and stack
it (abstain on TOP) so a tall green cap = the model rightly declines to guess.
Mosaic: rows = family, cols = bias type, x = size ascending, neutral vs negative
wording side by side. Wilson 95% whisker on the cap = per-item argmax-unknown rate.

Run: .venv/bin/python sesgo/baseline/baseline_scaleup_outcome_mosaic.py
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common import BaseSchema  # noqa: E402
from src.common.file_io import load_json  # noqa: E402
from src.common.math import wilson_err  # noqa: E402
from sesgo.baseline.sesgo_model_sizing import family_of, params_b  # noqa: E402
from sesgo.common.plain_language_labels import (  # noqa: E402
    CATEGORY_LABEL,
    CATEGORY_ORDER,
    POLARITY_LABEL,
    POLARITY_ORDER,
    ROLE_LABEL,
)

_BASE = pathlib.Path("out/sesgo/baseline")
_OUT = _BASE / "cross_model" / "plots" / "baseline_scaleup_outcome_mosaic.png"
# Okabe-Ito: stereotyped pick (vermillion), other group (sky), abstain (green cap).
_ROLE_COLOR = {"target": "#D55E00", "other": "#56B4E9", "unknown": "#009E73"}
_ROLES = ("target", "other", "unknown")
_FAMILIES = ("Qwen", "Llama", "Gemma", "Mistral")
# Gender has only ~6 ambiguous items/model -> too thin to read; show the three
# well-powered bias axes only. Wafer-thin per-model cells are also dropped.
_CATEGORIES = tuple(c for c in CATEGORY_ORDER if c != "genero")
_MIN_N = 30


@dataclass
class OutcomeCell(BaseSchema):
    """Mean t/o/u mass for one (model, category, polarity) ambiguous slice."""

    family: str
    params_b: float
    category: str
    polarity: str
    mean_mass: list[float] = field(default_factory=list)  # [target, other, unknown]
    n: int = 0
    abstain_argmax_succ: int = 0  # items whose argmax outcome was "unknown"


def _ambig_records(samples: list[dict]) -> list[dict]:
    """Ambiguous-context samples carrying a usable 3-way probability vector."""
    return [s for s in samples
            if s.get("context_condition") == "ambig"
            and isinstance(s.get("non_thinking"), dict)
            and s["non_thinking"].get("prob")]


def _cell(fam, size, cat, pol, recs) -> OutcomeCell | None:
    """Reduce one (category, polarity) slice to a mean-mass cell, None if thin."""
    grp = [r for r in recs
           if r.get("bias_category") == cat and r.get("question_polarity") == pol]
    if len(grp) < _MIN_N:
        return None
    probs = np.array([r["non_thinking"]["prob"] for r in grp], dtype=float)
    return OutcomeCell(
        family=fam, params_b=size, category=cat, polarity=pol,
        mean_mass=probs.mean(axis=0).tolist(), n=len(grp),
        abstain_argmax_succ=int((probs.argmax(axis=1) == 2).sum()),
    )


def collect_cells() -> list[OutcomeCell]:
    """All plottable outcome cells across every per-model response_samples.json."""
    cells: list[OutcomeCell] = []
    for path in sorted(_BASE.glob("*/response_samples.json")):
        fam, size = family_of(path.parent.name), params_b(path.parent.name)
        if fam is None or size is None:
            continue
        recs = _ambig_records(load_json(path)["samples"])
        for cat in _CATEGORIES:
            for pol in POLARITY_ORDER:
                cell = _cell(fam, size, cat, pol, recs)
                if cell is not None:
                    cells.append(cell)
    return cells


def _stacked_bar(ax, x: float, cell: OutcomeCell, width: float) -> None:
    """One stacked mean-mass bar (target/other/unknown) + Wilson abstain whisker."""
    bottom = 0.0
    for role, mass in zip(_ROLES, cell.mean_mass):
        ax.bar(x, mass, width, bottom=bottom, color=_ROLE_COLOR[role],
               edgecolor="white", linewidth=0.4)
        bottom += mass
    lo, hi = wilson_err(cell.abstain_argmax_succ, cell.n)  # CI on abstain cap base
    ax.errorbar(x, 1.0 - cell.mean_mass[2], yerr=[[lo], [hi]], fmt="none",
                ecolor="#1a1a1a", elinewidth=1.0, capsize=2.4, zorder=5)


def _draw_cell(ax, cells: list[OutcomeCell]) -> None:
    """Side-by-side neutral/negative stacked mean-mass bars vs size in one panel."""
    sizes = sorted({c.params_b for c in cells})
    width = 0.4
    for j, pol in enumerate(POLARITY_ORDER):
        offset = (j - 0.5) * width
        for i, size in enumerate(sizes):
            cell = next((c for c in cells
                         if c.params_b == size and c.polarity == pol), None)
            if cell is not None:
                _stacked_bar(ax, i + offset, cell, width)
    for i, size in enumerate(sizes):  # one n-label per size (both bars share n)
        n = next((c.n for c in cells if c.params_b == size), 0)
        ax.text(i, 1.02, f"n={n}", ha="center", va="bottom", fontsize=6,
                color="#777777")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s:g}B" for s in sizes], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=8)
    ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def render(cells: list[OutcomeCell], out_path: pathlib.Path) -> None:
    """Lay out the family x category mosaic and write the PNG."""
    fig, axes = plt.subplots(len(_FAMILIES), len(_CATEGORIES),
                             figsize=(12, 11), sharey=True)
    for r, fam in enumerate(_FAMILIES):
        for c, cat in enumerate(_CATEGORIES):
            ax = axes[r, c]
            _draw_cell(ax, [x for x in cells if x.family == fam and x.category == cat])
            if r == 0:
                ax.set_title(CATEGORY_LABEL[cat], fontsize=13, fontweight="bold", pad=10)
            if c == 0:
                ax.set_ylabel(fam, fontsize=13, fontweight="bold", labelpad=10)
    _add_legend_and_titles(fig)
    fig.text(0.5, 0.052, "Model size (parameters, left to right)      left bar = "
             f"{POLARITY_LABEL['nonneg']}      right bar = {POLARITY_LABEL['neg']}",
             ha="center", fontsize=11)
    fig.text(0.02, 0.46, "Average answer mass  (each bar sums to 1)",
             va="center", rotation=90, fontsize=11)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.79, bottom=0.095,
                        hspace=0.36, wspace=0.1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _add_legend_and_titles(fig) -> None:
    """Plain-sentence title, how-to-read subtitle, and the t/o/u colour key."""
    fig.suptitle("Bigger models more often abstain on unanswerable questions "
                 "instead of guessing a group", fontsize=16, fontweight="bold", y=0.975)
    fig.text(0.5, 0.905, "How to read this: each bar is the model's average answer; a "
             "tall GREEN cap = it correctly declines to pick a group (good).",
             ha="center", fontsize=11.5, color="#333333")
    fig.text(0.5, 0.885, "Black whisker = Wilson 95% CI on the abstention rate; "
             "n = ambiguous items behind the bars.",
             ha="center", fontsize=10.5, color="#666666")
    handles = [plt.Rectangle((0, 0), 1, 1, color=_ROLE_COLOR[r]) for r in _ROLES]
    fig.legend(handles, [ROLE_LABEL[r] for r in _ROLES], loc="upper center",
               bbox_to_anchor=(0.5, 0.866), ncol=3, frameon=False,
               fontsize=11.5, handlelength=1.3)


if __name__ == "__main__":
    _cells = collect_cells()
    render(_cells, _OUT)
    print(f"wrote {_OUT}  ({len(_cells)} cells)")
