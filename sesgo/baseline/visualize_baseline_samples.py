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
  - mean non-thinking role-prob vector [TARGET, OTHER, UNKNOWN] overall and BY
    bias_category — where the un-abstained mass leaks (target vs other group).

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
# regardless of cwd. From <repo>/sesgo/baseline/x.py, parents[2] is
# the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample  # noqa: E402

# Non-thinking prob vector is ordered [TARGET, OTHER, UNKNOWN].
_ROLE_NAMES = ("target", "other", "unknown")
# The population axes we slice abstention by (everything is one rendering/item).
_SLICE_AXES = ("bias_category", "question_polarity", "language")


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


def _abstention_accuracy(samples: list[SesgoSample]) -> tuple[float | None, int]:
    """Overall (fraction predicting UNKNOWN, n) over the given samples."""
    flags = [s.correct_non_thinking for s in samples]
    return (sum(flags) / len(flags) if flags else None), len(flags)


def _accuracy_by(
    samples: list[SesgoSample], axis: str
) -> dict[str, tuple[float | None, int]]:
    """Map each value of `axis` -> (abstention accuracy, n), sorted by key."""
    groups: dict[str, list[SesgoSample]] = defaultdict(list)
    for s in samples:
        groups[str(getattr(s, axis))].append(s)
    return {k: _abstention_accuracy(groups[k]) for k in sorted(groups)}


def _mean_role_prob(samples: list[SesgoSample]) -> list[float] | None:
    """Mean non-thinking prob vector [TARGET, OTHER, UNKNOWN], or None if empty."""
    vecs = [s.non_thinking.prob for s in samples if s.non_thinking is not None]
    if not vecs:
        return None
    return [float(np.mean(col)) for col in zip(*vecs)]


def _mean_role_prob_by(
    samples: list[SesgoSample], axis: str
) -> dict[str, list[float]]:
    """Map each value of `axis` -> mean role-prob vector, sorted by key."""
    groups: dict[str, list[SesgoSample]] = defaultdict(list)
    for s in samples:
        groups[str(getattr(s, axis))].append(s)
    out: dict[str, list[float]] = {}
    for k in sorted(groups):
        vec = _mean_role_prob(groups[k])
        if vec is not None:
            out[k] = vec
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_overall_accuracy(
    overall: tuple[float | None, int],
    by_axis: dict[str, dict[str, tuple[float | None, int]]],
    model: str,
    out_path: Path,
) -> Path:
    """Bar chart: overall abstention accuracy plus each slice's per-value bars."""
    labels: list[str] = []
    vals: list[float] = []
    ns: list[int] = []
    acc, n = overall
    labels.append("OVERALL")
    vals.append(acc if acc is not None else 0.0)
    ns.append(n)
    for axis in _SLICE_AXES:
        for key, (a, kn) in by_axis[axis].items():
            labels.append(f"{axis}={key}")
            vals.append(a if a is not None else 0.0)
            ns.append(kn)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    colors = ["#30638e"] + sns.color_palette("viridis", len(labels) - 1)
    bars = ax.bar(range(len(labels)), vals, color=colors)
    for bar, a, kn in zip(bars, vals, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{a:.0%}\nn={kn}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("non-thinking abstention accuracy (fraction predicting UNKNOWN)")
    ax.set_title(f"SESGO non-thinking baseline: abstention accuracy ({model})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_role_prob(
    overall: list[float] | None,
    by_category: dict[str, list[float]],
    model: str,
    out_path: Path,
) -> Path:
    """Grouped bars: mean non-thinking role-prob [target,other,unknown].

    One group of three role bars for OVERALL, then one per bias_category.
    """
    groups: list[tuple[str, list[float]]] = []
    if overall is not None:
        groups.append(("OVERALL", overall))
    for key in by_category:
        groups.append((f"cat={key}", by_category[key]))

    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.8), 5))
    n_roles = len(_ROLE_NAMES)
    width = 0.8 / n_roles
    palette = sns.color_palette("Set2", n_roles)
    x = np.arange(len(groups))
    for r, role in enumerate(_ROLE_NAMES):
        heights = [g[1][r] for g in groups]
        ax.bar(x + (r - (n_roles - 1) / 2) * width, heights, width,
               label=role, color=palette[r])
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mean non-thinking role probability")
    ax.set_title(f"SESGO non-thinking baseline: mean role-prob [target/other/unknown] ({model})")
    ax.legend(title="role")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
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


def main() -> None:
    """Load the SesgoDataset, compute every baseline statistic, plot, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO NON_THINKING_BASELINE")

    dataset = SesgoDataset.from_json(args.samples)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")
    n_items = len({s.question_id for s in dataset.samples})
    log(f"[viz] {n_items} distinct question_id(s)")

    scored = _scored(dataset)

    # Overall + per-axis abstention accuracy.
    overall_acc = _abstention_accuracy(scored)
    acc_by = {axis: _accuracy_by(scored, axis) for axis in _SLICE_AXES}

    # Mean role-prob vector overall + by bias_category.
    overall_prob = _mean_role_prob(scored)
    prob_by_cat = _mean_role_prob_by(scored, "bias_category")

    # ----- stats table to the log ----------------------------------------- #
    log_section("NON_THINKING_BASELINE STATS")
    log(f"  scored samples (non-thinking present): {len(scored)}")
    acc, n = overall_acc
    log(f"  overall abstention accuracy: {_fmt(acc, pct=True)} (n={n})")
    for axis in _SLICE_AXES:
        log(f"  abstention accuracy by {axis}:")
        for key, (a, kn) in acc_by[axis].items():
            log(f"    {key:<12}: {_fmt(a, pct=True):>6} (n={kn})")
    log(f"  mean role-prob [target, other, unknown] overall: {_fmt_vec(overall_prob)}")
    log("  mean role-prob [target, other, unknown] by bias_category:")
    for key, vec in prob_by_cat.items():
        log(f"    {key:<12}: {_fmt_vec(vec)}")
    log("  NOTE: ambiguous gold is always UNKNOWN, so abstention == accuracy.")

    # ----- plots ---------------------------------------------------------- #
    plots_dir = args.out_dir / "sesgo" / "baseline" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    written: list[Path] = []
    written.append(plot_overall_accuracy(
        overall_acc, acc_by, dataset.model_name, plots_dir / "abstention_accuracy.png"))
    written.append(plot_role_prob(
        overall_prob, prob_by_cat, dataset.model_name, plots_dir / "role_prob.png"))

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
