"""Does step-by-step thinking change what the model believes? (divergence study).

ONE narrative figure: per item, the chance the model abstains ('unknown') BEFORE
it reasons (a single direct read) vs AFTER it reasons step by step (averaged over
its free-form tries). Points ON the y=x diagonal mean thinking left the belief
unchanged; points OFF mean thinking moved it. The main panel shows the abstain
mass; two small companions repeat the picture for the stereotyped-group mass and
the other-group mass. Coloured by social-group category, Wilson 95% CIs on the
post-think (count-backed) axis, n annotated. Existing data only — nothing sampled.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write the PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Run-by-path: put the repo root on sys.path so first-party imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import BaseSchema  # noqa: E402
from src.common.file_io import load_json  # noqa: E402
from src.common.math import wilson_err  # noqa: E402
from sesgo.common.plain_language_labels import CATEGORY_LABEL, CATEGORY_ORDER  # noqa: E402

# Index of each outcome inside a [target, other, unknown] mix vector.
ROLE_IDX = {"target": 0, "other": 1, "unknown": 2}
# Plain-language panel titles per outcome (no pipeline jargon).
ROLE_TITLE = {
    "unknown": "Chance of abstaining ('unknown')",
    "target": "Chance of picking the stereotyped group",
    "other": "Chance of picking the other group",
}
# Okabe-Ito colourblind-safe colour per social-group category.
CATEGORY_COLOR = {
    "clasismo": "#0072B2",
    "racismo": "#D55E00",
    "xenofobia": "#009E73",
    "genero": "#CC79A7",
}
PRE_LABEL = "Before thinking\n(answers directly)"
POST_LABEL = "After thinking\n(reasons step by step)"


@dataclass
class ItemAgreement(BaseSchema):
    """One item's pre-think vs post-think mass for a single outcome, with a CI."""

    category: str
    pre: float          # direct-read probability of the outcome
    post: float         # mean over the free-form thinking tries
    n_tries: int        # rollouts the post-think mean is averaged over
    post_err_lo: float  # Wilson 95% lower offset on the post-think mass
    post_err_hi: float  # Wilson 95% upper offset on the post-think mass


def _scored_items(samples: list[dict]) -> list[dict]:
    """Items whose post-think mean is backed by >=1 parsed reasoning try."""
    return [s for s in samples if s["thinking"]["sample_size"] >= 1]


def _agreement(sample: dict, role: str) -> ItemAgreement:
    """Pre-/post-think mass for one outcome, with a Wilson CI on the post count."""
    idx = ROLE_IDX[role]
    pre = float(sample["non_thinking"]["prob"][idx])
    post = float(sample["thinking"]["mean"][idx])
    n = int(sample["thinking"]["sample_size"])
    lo, hi = wilson_err(round(post * n), n)  # post mean is a fraction of n tries
    return ItemAgreement(sample["bias_category"], pre, post, n, lo, hi)


def _draw_panel(ax, items: list[ItemAgreement], role: str, *,
                marker: float = 58) -> None:
    """One pre-vs-post scatter with the y=x diagonal and FAINT Wilson whiskers."""
    ax.plot([0, 1], [0, 1], color="#555555", lw=1.4, ls="--", zorder=1)
    for cat in CATEGORY_ORDER:
        grp = [it for it in items if it.category == cat]
        if not grp:
            continue
        xs, ys = [it.pre for it in grp], [it.post for it in grp]
        yerr = np.array([[it.post_err_lo for it in grp], [it.post_err_hi for it in grp]])
        # Faint NEUTRAL whiskers: small-n CIs are wide, so keep them recessive grey
        # so they convey uncertainty without burying the coloured points/finding.
        ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor="#bbbbbb", alpha=0.5,
                    lw=0.6, capsize=0, zorder=2)
        ax.scatter(xs, ys, s=marker, color=CATEGORY_COLOR[cat], edgecolor="white",
                   linewidth=0.7, alpha=0.92, zorder=3)
    ax.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04), aspect="equal")
    ax.set_xlabel(PRE_LABEL, fontsize=8.5)
    ax.set_ylabel(POST_LABEL, fontsize=8.5)
    ax.set_title(ROLE_TITLE[role], fontsize=10.5, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.18, lw=0.6)


def build_figure(samples: list[dict]):
    """Main abstain panel + two companions (target / other), shared legend."""
    items = _scored_items(samples)
    fig = plt.figure(figsize=(13.0, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1],
                          left=0.07, right=0.985, top=0.95, bottom=0.16,
                          wspace=0.26, hspace=0.5)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_tgt = fig.add_subplot(gs[0, 1])
    ax_oth = fig.add_subplot(gs[1, 1])

    _draw_panel(ax_main, [_agreement(s, "unknown") for s in items], "unknown")
    _draw_panel(ax_tgt, [_agreement(s, "target") for s in items], "target", marker=32)
    _draw_panel(ax_oth, [_agreement(s, "other") for s in items], "other", marker=32)

    # ONE shared legend below the panels, out of the data area.
    counts = {c: sum(1 for s in items if s["bias_category"] == c) for c in CATEGORY_ORDER}
    handles = [plt.Line2D([0], [0], marker="o", ls="", markersize=8, markeredgecolor="white",
               markerfacecolor=CATEGORY_COLOR[c], label=f"{CATEGORY_LABEL[c]} (n={counts[c]})")
               for c in CATEGORY_ORDER if counts[c]]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.02))
    return fig


def main() -> None:
    """Load existing divergence samples and write the agreement-scatter PNG."""
    base = Path("out/sesgo/divergence/Qwen3-0.6B")
    data = load_json(base / "response_samples.json")
    fig = build_figure(data["samples"])
    out_dir = base / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "thinking_belief_agreement_scatter.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
