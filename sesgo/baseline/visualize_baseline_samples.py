"""Compute and plot ALL NON_THINKING_BASELINE statistics for a SesgoDataset.

Run-by-path driver for the NON_THINKING_BASELINE study. Loads a samples.json
produced by collect_baseline_samples.py (a SesgoDataset) and answers
one question: at the NON-THINKING level, on the single un-varied, un-scaffolded
rendering of each item, how often does the model ABSTAIN (predict UNKNOWN, the
ambiguous gold), and where does its 3-way mass sit across roles?

Because the baseline has one rendering per item — no format variation, no scaffold
— every number here is the raw, pre-reasoning behaviour with nothing to average
over per item. We therefore slice the population instead:
  - OVERALL non-thinking abstention accuracy (fraction predicting UNKNOWN).
  - abstention accuracy BY bias_category / question polarity / language.
  - mean non-thinking role-prob vector [TARGET, OTHER, UNKNOWN] overall and per
    every slice axis — where the un-abstained mass leaks (target vs other group).

Plots land at out/sesgo/baseline/<MODEL>/plots/. A stats table is
logged. Robust to subsampled data — a slice may hold only a few items.

Usage:
  uv run python sesgo/baseline/visualize_baseline_samples.py \
      out/sesgo/baseline/Qwen3-0.6B/samples.json
  uv run python sesgo/baseline/visualize_baseline_samples.py \
      SAMPLES.json --out-dir out
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/baseline/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample  # noqa: E402

# Non-thinking prob vector is ordered [TARGET, OTHER, UNKNOWN].
_ROLE_NAMES = ("target", "other", "unknown")
# The population axes we slice by (everything is one rendering/item).
_SLICE_AXES = ("bias_category", "question_polarity", "language")
# Human-friendly section titles for each axis (small-multiple panel headers).
_AXIS_TITLES = {
    "OVERALL": "overall",
    "bias_category": "by bias category",
    "question_polarity": "by question polarity",
    "language": "by language",
}
# Colorblind-safe palette (Okabe–Ito): one hue per role, reused everywhere.
_ROLE_COLORS = {"target": "#E69F00", "other": "#56B4E9", "unknown": "#009E73"}
_BAR_COLOR = "#0072B2"  # single-series abstention bars (Okabe–Ito blue)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for non-thinking-baseline visualization."""
    parser = argparse.ArgumentParser(
        description="Compute and plot all NON_THINKING_BASELINE statistics for a SesgoDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "samples",
        type=Path,
        help="Path to samples.json (a SesgoDataset) from collect_baseline_samples.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; plots land at <out-dir>/sesgo/baseline/<MODEL>/plots/",
    )
    return parser.parse_args()


def _scored(dataset: SesgoDataset) -> list[SesgoSample]:
    """Samples that carry a non-thinking prediction (exclude readout-less ones)."""
    return [s for s in dataset.samples if s.predicted_non_thinking is not None]


def _group_by(samples: list[SesgoSample], axis: str) -> dict[str, list[SesgoSample]]:
    """Partition samples by the string value of `axis`, key-sorted."""
    groups: dict[str, list[SesgoSample]] = defaultdict(list)
    for s in samples:
        groups[str(getattr(s, axis))].append(s)
    return {k: groups[k] for k in sorted(groups)}


def _abstention_accuracy(samples: list[SesgoSample]) -> tuple[float | None, int]:
    """Overall (fraction predicting UNKNOWN, n) over the given samples."""
    flags = [s.correct_non_thinking for s in samples]
    return (sum(flags) / len(flags) if flags else None), len(flags)


def _mean_role_prob(samples: list[SesgoSample]) -> list[float] | None:
    """Mean non-thinking prob vector [TARGET, OTHER, UNKNOWN], or None if empty."""
    vecs = [s.non_thinking.prob for s in samples if s.non_thinking is not None]
    if not vecs:
        return None
    return [float(np.mean(col)) for col in zip(*vecs)]


def _sections(scored: list[SesgoSample]) -> list[tuple[str, dict[str, list[SesgoSample]]]]:
    """Ordered (axis, {value: samples}) sections: OVERALL then each slice axis."""
    out: list[tuple[str, dict[str, list[SesgoSample]]]] = [("OVERALL", {"all items": scored})]
    out += [(axis, _group_by(scored, axis)) for axis in _SLICE_AXES]
    return out


def _suptitle(fig, model: str, subtitle: str, n: int) -> None:
    """Two-line figure title: study + breakdown, with n and model for context."""
    fig.suptitle(
        f"SESGO non-thinking baseline · {model}  (n={n} scored items)\n{subtitle}",
        fontsize=13, fontweight="bold",
    )


# --------------------------------------------------------------------------- #
# Plots — both use small-multiples so each breakdown dimension reads on its own.
# --------------------------------------------------------------------------- #
def plot_overall_accuracy(scored: list[SesgoSample], model: str, out_path: Path) -> Path:
    """Small-multiples of abstention accuracy: one panel per breakdown dimension."""
    sections = _sections(scored)
    overall_acc, n = _abstention_accuracy(scored)
    widths = [max(1, len(grp)) for _, grp in sections]
    fig, axes = plt.subplots(
        1, len(sections), figsize=(11, 4.6), sharey=True,
        gridspec_kw={"width_ratios": widths}, constrained_layout=True,
    )
    for ax, (axis, grp) in zip(np.atleast_1d(axes), sections):
        keys = list(grp)
        vals, ns = zip(*((_abstention_accuracy(grp[k])) for k in keys))
        vals = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(range(len(keys)), vals, color=_BAR_COLOR, width=0.7)
        for bar, v, kn in zip(bars, vals, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.0%}\nn={kn}", ha="center", va="bottom", fontsize=8.5)
        # Dashed reference at the population mean so slices read as above/below.
        if overall_acc is not None:
            ax.axhline(overall_acc, ls="--", lw=1, color="#555555", alpha=0.7, zorder=0)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
        ax.set_title(_AXIS_TITLES[axis], fontsize=11)
        ax.margins(x=0.15)
    axes_flat = np.atleast_1d(axes)
    axes_flat[0].set_ylim(0, 1.18)
    axes_flat[0].set_ylabel("abstention accuracy\n(fraction predicting UNKNOWN)", fontsize=10)
    axes_flat[-1].text(
        0.99, overall_acc, f" overall {overall_acc:.0%}", transform=axes_flat[-1].get_yaxis_transform(),
        ha="right", va="bottom", fontsize=8, color="#555555", clip_on=False,
    )
    _suptitle(fig, model, "abstention accuracy by population slice", n)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _role_bars(ax, keys: list[str], vecs: list[list[float]]) -> None:
    """Draw grouped target/other/unknown bars for one panel."""
    width = 0.8 / len(_ROLE_NAMES)
    x = np.arange(len(keys))
    for r, role in enumerate(_ROLE_NAMES):
        offset = (r - (len(_ROLE_NAMES) - 1) / 2) * width
        ax.bar(x + offset, [v[r] for v in vecs], width, label=role,
               color=_ROLE_COLORS[role])
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
    ax.margins(x=0.1)


def plot_role_prob(scored: list[SesgoSample], model: str, out_path: Path) -> Path:
    """Small-multiples of mean role-prob [target/other/unknown] per breakdown."""
    sections = _sections(scored)
    panels = [
        (axis, list(grp), [v for k in grp if (v := _mean_role_prob(grp[k])) is not None])
        for axis, grp in sections
    ]
    widths = [max(1, len(keys)) for _, keys, _ in panels]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(12, 4.8), sharey=True,
        gridspec_kw={"width_ratios": widths}, constrained_layout=True,
    )
    for ax, (axis, keys, vecs) in zip(np.atleast_1d(axes), panels):
        _role_bars(ax, keys, vecs)
        ax.set_title(_AXIS_TITLES[axis], fontsize=11)
    axes_flat = np.atleast_1d(axes)
    axes_flat[0].set_ylim(0, 1.0)
    axes_flat[0].set_ylabel("mean non-thinking role probability", fontsize=10)
    # Reserve a strip of whitespace below the panels for the legend so it never
    # collides with the rotated x-tick labels, then anchor the legend into it.
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.92))
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="role", ncol=len(_ROLE_NAMES),
               loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False,
               fontsize=10, title_fontsize=10)
    _, n = _abstention_accuracy(scored)
    _suptitle(fig, model, "mean role-probability mass [target / other / unknown]", n)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _fmt(v: float | None, pct: bool = False) -> str:
    """Render a metric: percentage / float / n/a."""
    if v is None:
        return "n/a"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def _fmt_vec(vec: list[float] | None) -> str:
    """Render a role-prob vector as [t, o, u], or n/a."""
    if vec is None:
        return "n/a"
    return "[" + ", ".join(f"{x:.3f}" for x in vec) + "]"


def _log_stats(scored: list[SesgoSample]) -> None:
    """Emit the full baseline stats table to the log."""
    log_section("NON_THINKING_BASELINE STATS")
    log(f"  scored samples (non-thinking present): {len(scored)}")
    acc, n = _abstention_accuracy(scored)
    log(f"  overall abstention accuracy: {_fmt(acc, pct=True)} (n={n})")
    log(f"  mean role-prob [target, other, unknown] overall: {_fmt_vec(_mean_role_prob(scored))}")
    for axis in _SLICE_AXES:
        log(f"  by {axis}:")
        for key, grp in _group_by(scored, axis).items():
            a, kn = _abstention_accuracy(grp)
            log(f"    {key:<12}: abstain {_fmt(a, pct=True):>6} (n={kn})  "
                f"role {_fmt_vec(_mean_role_prob(grp))}")
    log("  NOTE: ambiguous gold is always UNKNOWN, so abstention == accuracy.")


def main() -> None:
    """Load the SesgoDataset, compute every baseline statistic, plot, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO NON_THINKING_BASELINE")

    dataset = SesgoDataset.from_json(args.samples)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")
    n_items = len({s.question_id for s in dataset.samples})
    log(f"[viz] {n_items} distinct question_id(s)")

    scored = _scored(dataset)
    _log_stats(scored)

    plots_dir = args.out_dir / "sesgo" / "baseline" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    written = [
        plot_overall_accuracy(scored, dataset.model_name, plots_dir / "abstention_accuracy.png"),
        plot_role_prob(scored, dataset.model_name, plots_dir / "role_prob.png"),
    ]
    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
