r"""ONE page-filling figure for the SHARED-BASE forking experiment: every Qwen3.5 thinking
model size (rows) forks the SAME Qwen3.5-27B abstaining chain of thought for each bias
category (columns), and we plot the outcome distribution $O_t$ along that shared reasoning.

Reading down a column shows whether the larger model's careful (abstaining) reasoning
"rescues" the smaller models toward \emph{unknown} (the gold answer on these ambiguous
negative items) or whether they still commit to a named group along the identical tokens.

  uv run python -m experiment.forking.shared_base_grid_figure \
      --forking-dir out/forking --out paper/figures/forking_shared_grid.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from experiment.common.sweep_models import parse_model
from experiment.forking.forking_plot_styles import OUTCOME_COLORS

# Column tag -> human category name (shared-base runs are all negative-polarity ambiguous).
_CATEGORIES = [("-racsh", "racismo"), ("-xensh", "xenofobia"),
               ("-clash", "clasismo"), ("-gensh", "género")]
_LEGEND = {"target": "target group", "other": "other group",
           "unknown": "unknown (gold)", "unparseable": "unparseable"}


def _cell(ax, traj: dict) -> None:
    """One O_t stacked area (outcome distribution over the shared 27B base path)."""
    labels = traj["outcome_set"]["labels"]
    o = np.array([p["outcome_histogram"] for p in traj["positions"]])
    xs = np.arange(len(o))
    ax.stackplot(xs, o.T, colors=[OUTCOME_COLORS[l] for l in labels])
    ax.set_xlim(0, max(len(o) - 1, 1)); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])


def _load(d: Path) -> dict | None:
    t = d / "forking_trajectory.json"
    return json.load(t.open()) if t.exists() else None


def build(forking_dir: Path, out_path: Path) -> Path:
    # Rows = model sizes present (size-sorted); columns = the four bias categories.
    sizes: dict[float, str] = {}
    for d in forking_dir.iterdir():
        for tag, _ in _CATEGORIES:
            if d.is_dir() and d.name.endswith(tag):
                sm = parse_model(d.name[: -len(tag)])
                if sm:
                    sizes.setdefault(sm.size_b, sm.name)
    rows = sorted(sizes)
    if not rows:
        print(f"[shared-grid] no shared-base runs under {forking_dir}"); return out_path

    fig = plt.figure(figsize=(11, 2.2 * len(rows) + 1.0))
    gs = GridSpec(len(rows), len(_CATEGORIES), figure=fig,
                  hspace=0.12, wspace=0.06, top=0.90, bottom=0.04, left=0.10, right=0.985)
    legend_labels = None
    for r, size in enumerate(rows):
        for c, (tag, cat) in enumerate(_CATEGORIES):
            ax = fig.add_subplot(gs[r, c])
            # parse_model names are like "Qwen3.5 0.8B"; dirs are "Qwen3.5-0.8B-<tag>".
            cand = [p for p in forking_dir.iterdir()
                    if p.name.endswith(tag) and parse_model(p.name[: -len(tag)])
                    and parse_model(p.name[: -len(tag)]).size_b == size]
            loaded = _load(cand[0]) if cand else None
            if loaded:
                _cell(ax, loaded)
                legend_labels = legend_labels or loaded["outcome_set"]["labels"]
            else:
                ax.set_xticks([]); ax.set_yticks([]); ax.text(0.5, 0.5, "—", ha="center", va="center")
            if r == 0:
                ax.set_title(cat, fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(sizes[size], fontsize=10, fontweight="bold",
                              rotation=0, ha="right", va="center", labelpad=30)

    handles = [Line2D([], [], marker="s", ls="", mfc=OUTCOME_COLORS[l], mec="none", label=_LEGEND[l])
               for l in (legend_labels or ["target", "other", "unknown", "unparseable"])]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Forking a shared Qwen3.5-27B abstaining chain of thought "
                 "(rows: model size; columns: bias category)\n"
                 "all items ambiguous, negative framing; correct answer = abstain (unknown / gold)",
                 fontsize=11.5, fontweight="bold", y=0.999)
    fig.text(0.5, 0.015, "chain-of-thought token position (shared 27B base path) "
             "$\\rightarrow$ outcome distribution $O_t$", ha="center", fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[shared-grid] wrote {out_path}  ({len(rows)} sizes x {len(_CATEGORIES)} categories)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forking-dir", type=Path, default=Path("out/forking"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures/forking_shared_grid.png"))
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    build(a.forking_dir, a.out)


if __name__ == "__main__":
    main()
