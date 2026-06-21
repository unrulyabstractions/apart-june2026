"""Plot a collected SesgoDataset into PNGs under out/sesgo/<MODEL>/plots/.

Run-by-path driver. Loads a responses.json produced by collect_llm_responses.py
(a SesgoDataset), and renders the headline with/without-scaffold comparison plus
a handful of grouped/faceted breakdowns. On ambiguous SESGO items the gold is
always UNKNOWN, so "accuracy" is the fraction of predictions that abstain
(predict UNKNOWN). Thinking predictions with no parsed draw (sample_size == 0)
have no decodable answer and are excluded from thinking accuracy.

Usage:
  uv run python sesgo/baseline/visualize_llm_responses.py \
      out/sesgo/Qwen3-0.6B/responses.json
  uv run python sesgo/baseline/visualize_llm_responses.py RESPONSES.json --out-dir out
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

from src.common.logging import log, log_header  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402

# Role order for the non-thinking probability vector [TARGET, OTHER, UNKNOWN].
_ROLE_NAMES = ["target", "other", "unknown"]
_BASELINE = "(baseline)"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SESGO visualization."""
    parser = argparse.ArgumentParser(
        description="Plot a collected SesgoDataset into PNGs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "responses",
        type=Path,
        help="Path to responses.json (a SesgoDataset) from collect_llm_responses.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output directory; plots land at <out-dir>/sesgo/<MODEL>/plots/",
    )
    return parser.parse_args()


def _scaffold_label(scaffold_id: str | None) -> str:
    """Human label for a scaffold condition; None == the no-scaffold baseline."""
    return scaffold_id or _BASELINE


def _accuracy(flags: list[bool]) -> float:
    """Fraction of True flags (== predicted UNKNOWN); 0.0 when empty."""
    return sum(flags) / len(flags) if flags else 0.0


def _ordered_scaffolds(dataset: SesgoDataset) -> list[str]:
    """Scaffold labels with the baseline first, the rest sorted after it."""
    labels = {_scaffold_label(s.scaffold_id) for s in dataset.samples}
    rest = sorted(labels - {_BASELINE})
    return ([_BASELINE] if _BASELINE in labels else []) + rest


def plot_accuracy_by_scaffold(
    dataset: SesgoDataset, level: str, out_path: Path
) -> Path:
    """Headline bar chart: abstention accuracy by scaffold (baseline + 4)."""
    scaffolds = _ordered_scaffolds(dataset)
    flags: dict[str, list[bool]] = defaultdict(list)
    for s in dataset.samples:
        if level == "non_thinking" and s.predicted_non_thinking is not None:
            flags[_scaffold_label(s.scaffold_id)].append(s.correct_non_thinking)
        elif level == "thinking" and s.predicted_thinking is not None:
            flags[_scaffold_label(s.scaffold_id)].append(s.correct_thinking)

    accs = [_accuracy(flags.get(sc, [])) for sc in scaffolds]
    ns = [len(flags.get(sc, [])) for sc in scaffolds]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#888888"] + sns.color_palette("viridis", len(scaffolds) - 1)
    bars = ax.bar(range(len(scaffolds)), accs, color=colors[: len(scaffolds)])
    for bar, acc, n in zip(bars, accs, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.0%}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(scaffolds)))
    ax.set_xticklabels(scaffolds, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy (fraction predicted UNKNOWN)")
    pretty = "non-thinking" if level == "non_thinking" else "thinking"
    ax.set_title(f"SESGO {pretty} abstention accuracy by scaffold ({dataset.model_name})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_accuracy_grouped(
    dataset: SesgoDataset, facet_attr: str, out_path: Path
) -> Path:
    """Grouped bars: non-thinking accuracy by scaffold, grouped by `facet_attr`."""
    scaffolds = _ordered_scaffolds(dataset)
    groups = sorted({getattr(s, facet_attr) for s in dataset.samples})
    # acc[group][scaffold] = list of correctness flags.
    acc: dict[str, dict[str, list[bool]]] = {
        g: defaultdict(list) for g in groups
    }
    for s in dataset.samples:
        if s.predicted_non_thinking is None:
            continue
        acc[getattr(s, facet_attr)][_scaffold_label(s.scaffold_id)].append(
            s.correct_non_thinking
        )

    fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(groups)), 5))
    width = 0.8 / max(len(scaffolds), 1)
    x = np.arange(len(groups))
    palette = ["#888888"] + sns.color_palette("viridis", len(scaffolds) - 1)
    for i, sc in enumerate(scaffolds):
        vals = [_accuracy(acc[g].get(sc, [])) for g in groups]
        ax.bar(x + i * width, vals, width, label=sc, color=palette[i])
    ax.set_xticks(x + width * (len(scaffolds) - 1) / 2)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("non-thinking accuracy (fraction UNKNOWN)")
    ax.set_title(
        f"SESGO non-thinking accuracy by scaffold, grouped by {facet_attr} "
        f"({dataset.model_name})"
    )
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_role_mass_by_scaffold(dataset: SesgoDataset, out_path: Path) -> Path:
    """Stacked bars of mean non-thinking role probability [target,other,unknown]."""
    scaffolds = _ordered_scaffolds(dataset)
    # accumulate mean prob vector per scaffold.
    sums: dict[str, np.ndarray] = {sc: np.zeros(3) for sc in scaffolds}
    counts: dict[str, int] = dict.fromkeys(scaffolds, 0)
    for s in dataset.samples:
        if s.non_thinking is None:
            continue
        label = _scaffold_label(s.scaffold_id)
        sums[label] += np.asarray(s.non_thinking.prob, dtype=float)
        counts[label] += 1
    means = np.array(
        [sums[sc] / counts[sc] if counts[sc] else np.zeros(3) for sc in scaffolds]
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(scaffolds))
    role_colors = ["#d1495b", "#edae49", "#30638e"]  # target, other, unknown
    for r, role in enumerate(_ROLE_NAMES):
        ax.bar(
            range(len(scaffolds)),
            means[:, r],
            bottom=bottom,
            label=role,
            color=role_colors[r],
        )
        bottom += means[:, r]
    ax.set_xticks(range(len(scaffolds)))
    ax.set_xticklabels(scaffolds, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("mean non-thinking role probability")
    ax.set_title(
        f"SESGO mean non-thinking probability mass by role and scaffold "
        f"({dataset.model_name})"
    )
    ax.legend(title="role")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    """Load the SesgoDataset and render all plots into the plots/ dir."""
    args = parse_args()
    log_header("VISUALIZE SESGO RESPONSES")

    dataset = SesgoDataset.from_json(args.responses)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    plots_dir = args.out_dir / "sesgo" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    written: list[Path] = []
    # Headline with/without-scaffold comparison, both levels.
    written.append(
        plot_accuracy_by_scaffold(
            dataset, "non_thinking", plots_dir / "accuracy_by_scaffold_non_thinking.png"
        )
    )
    has_thinking = any(s.predicted_thinking is not None for s in dataset.samples)
    if has_thinking:
        written.append(
            plot_accuracy_by_scaffold(
                dataset, "thinking", plots_dir / "accuracy_by_scaffold_thinking.png"
            )
        )
    else:
        log("[viz] no parsed thinking draws; skipping thinking-accuracy chart")

    # Grouped/faceted breakdowns.
    for attr, fname in (
        ("bias_category", "accuracy_by_scaffold_x_category.png"),
        ("question_polarity", "accuracy_by_scaffold_x_polarity.png"),
        ("language", "accuracy_by_scaffold_x_language.png"),
    ):
        written.append(plot_accuracy_grouped(dataset, attr, plots_dir / fname))

    # Where the probability mass goes.
    written.append(
        plot_role_mass_by_scaffold(dataset, plots_dir / "role_mass_by_scaffold.png")
    )

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
