"""ONE page-filling mega figure of the forking-paths dynamics for EVERY model present
under a forking output dir (``out/forking/<bare>/``). Auto-discovers whatever models have
both a ``forking_trajectory.json`` and a ``forking_analysis.json``, so re-running it as
cloud results land grows the figure row by row.

One ROW per model (size-sorted): the outcome distribution O_t stacked over chain-of-thought
token position, the forking token marked (red dashed line + a heat strip of per-token O_t
jump with red markers on the forking tokens), and a narrow survival / change-point panel on
the right showing how sharply the committed answer firms up. Shared outcome legend on top.

  uv run python -m experiment.forking.forking_dynamics_grid_figure --forking-dir out/forking \
      --out paper/figures/forking_dynamics_grid.png
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
from experiment.forking.forking_plot_styles import OUTCOME_COLORS, save_fig, token_strip


def _load(model_dir: Path) -> tuple[dict, dict] | None:
    t, a = model_dir / "forking_trajectory.json", model_dir / "forking_analysis.json"
    if not (t.exists() and a.exists()):
        return None
    return json.load(t.open()), json.load(a.open())


def _fork_flags(jumps: list[float]) -> list[bool]:
    """Forking tokens = O_t-jump spikes above mean+std (matches plot_forking_dynamics)."""
    if not jumps:
        return []
    m, s = float(np.mean(jumps)), float(np.std(jumps))
    return [i > 0 and v > m + s for i, v in enumerate(jumps)]


def _row(fig, gs_row, traj: dict, analysis: dict, label: str) -> None:
    """Draw one model's row: O_t stacked area + token heat strip (left) and survival (right)."""
    labels = traj["outcome_set"]["labels"]
    o = np.array([p["outcome_histogram"] for p in traj["positions"]])  # (T, n_outcomes)
    xs = np.arange(len(o))
    cp = analysis["change_points"]
    fork_idx = cp["forking_token_index"]

    inner = gs_row.subgridspec(2, 2, width_ratios=[3.4, 1.0], height_ratios=[3.0, 0.5],
                               hspace=0.05, wspace=0.16)
    ax_o = fig.add_subplot(inner[0, 0])
    ax_o.stackplot(xs, o.T, colors=[OUTCOME_COLORS[l] for l in labels])
    ax_o.set_xlim(0, max(len(o) - 1, 1)); ax_o.set_ylim(0, 1)
    ax_o.set_xticklabels([]); ax_o.set_ylabel(label, fontsize=9, fontweight="bold", rotation=0,
                                              ha="right", va="center", labelpad=42)
    ax_o.tick_params(labelsize=7)
    if cp["significant"] and fork_idx >= 0:
        ax_o.axvline(fork_idx, color="red", ls="--", lw=1.3, zorder=5)

    ax_strip = fig.add_subplot(inner[1, 0])
    token_strip(ax_strip, traj["base_token_texts"],
                analysis["dynamic_states"]["forking_magnitude"],
                _fork_flags(analysis["dynamic_states"]["forking_magnitude"]))
    ax_strip.set_xlabel("chain-of-thought token position", fontsize=8)

    ax_s = fig.add_subplot(inner[0, 1])
    sv = analysis["survival"]["survival"]; tau = cp["tau_posterior"]
    ax_s.plot(range(len(sv)), sv, color="#0072B2", lw=1.3, label="survival")
    ax_s.plot(range(len(tau)), tau, color="red", lw=1.0, alpha=0.8, label=r"p($\tau$)")
    ax_s.set_xlim(0, max(len(sv) - 1, 1)); ax_s.set_ylim(-0.02, 1.05)
    ax_s.tick_params(labelsize=7)
    bf = cp.get("bayes_factor", 0.0)
    ax_s.set_title(f"commit @ t={fork_idx}  (BF {bf:.0e})" if cp["significant"] else "no commit",
                   fontsize=8)
    fig.add_subplot(inner[1, 1]).axis("off")


# Known per-item run-tag suffixes so the default (untagged) item does not pick up tagged dirs.
# The "*sh" tags are the SHARED-BASE runs (every model forks the SAME Qwen3.5-27B abstaining
# chain of thought for that bias category), kept distinct from the per-model untagged runs.
_KNOWN_TAGS = ("-xeno", "-racsh", "-xensh", "-clash", "-gensh")

# Human-readable item name per run tag (for the figure title).
_ITEM_NAME = {"": "racismo", "-xeno": "xenofobia",
              "-racsh": "racismo", "-xensh": "xenofobia", "-clash": "clasismo", "-gensh": "género"}


def build(forking_dir: Path, out_path: Path, run_tag: str = "", title: str = "") -> Path:
    rows = []
    for d in sorted(forking_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if run_tag:                       # this item: only dirs carrying the tag
            if not name.endswith(run_tag):
                continue
            base = name[: -len(run_tag)]
        else:                             # default item: skip any tagged dir
            if any(name.endswith(t) for t in _KNOWN_TAGS):
                continue
            base = name
        sm = parse_model(base)
        loaded = _load(d) if sm else None
        # forking output dirs are bare model names (no -thinking suffix); these runs were
        # all captured with --thinking, so include every model that has results.
        if sm and loaded:
            rows.append((sm.size_b, f"{sm.name} (think)", *loaded))
    rows.sort(key=lambda r: r[0])
    if not rows:
        print(f"[grid] no thinking-model forking results under {forking_dir}"); return out_path

    fig = plt.figure(figsize=(11, 2.6 * len(rows) + 0.8))
    gs = GridSpec(len(rows), 1, figure=fig, hspace=0.42, top=0.93, bottom=0.05, left=0.12, right=0.98)
    for i, (_size, name, traj, analysis) in enumerate(rows):
        _row(fig, gs[i], traj, analysis, name)
    handles = [Line2D([], [], marker="s", ls="", mfc=OUTCOME_COLORS[l], mec="none",
                      label={"target": "target group", "other": "other group",
                             "unknown": "unknown (correct)", "unparseable": "unparseable"}[l])
               for l in rows[0][2]["outcome_set"]["labels"]]
    handles.append(Line2D([], [], color="red", ls="--", lw=1.3, label="forking token"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.985))
    item = _ITEM_NAME.get(run_tag, run_tag.lstrip("-") or "item")
    suptitle = title or (f"Forking-paths dynamics across scale (Qwen3.5 thinking): {item} item, "
                         "ambiguous, negative framing; correct answer = abstain (unknown)")
    fig.suptitle(suptitle, fontsize=11.5, fontweight="bold", y=0.999)
    save_fig(fig, out_path)
    print(f"[grid] wrote {out_path}  ({len(rows)} models: {', '.join(r[1] for r in rows)})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forking-dir", type=Path, default=Path("out/forking"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures/forking_dynamics_grid.png"))
    ap.add_argument("--run-tag", default="", help="per-item output suffix (e.g. -xeno); empty = the default item")
    ap.add_argument("--title", default="", help="override the figure suptitle")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    build(a.forking_dir, a.out, a.run_tag, a.title)


if __name__ == "__main__":
    main()
