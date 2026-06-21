"""Compute and plot ALL STABILITY statistics for a collected SesgoDataset.

Run-by-path driver for the STABILITY half. Loads a response_samples.json produced
by collect_stability_samples.py (Spanish, ORIGINAL items, BOTH ambiguous and
disambiguated) and asks: how CONSISTENT is the model's answer across the
superficial FORMAT variations of the same item, and how does that interact with
the context condition?

Within a stability GROUP — (question_id, context_condition, polarity) — the gold
answer is fixed, so the 18 members differ only in label style x role permutation;
any change in the prediction is pure format sensitivity. We report, split by
context condition (ambig vs disambig) and for BOTH readouts (3-OPTION non-thinking
argmax and 2-OPTION forced choice), four publication-quality figures:

  - consistency.png        : per-item modal-answer fraction, both readouts stacked.
  - format_sensitivity.png : per-axis flip rate (label_style vs permutation) + CIs.
  - p_unknown_spread.png   : per-item std of non-thinking p(unknown), ambig/disambig.
  - accuracy.png           : per-condition accuracy vs gold, both readouts, Wilson CIs.

Filenames stay stable; the rendered titles/labels are plain English (see
sesgo/common/plain_language_labels.py).

Every bar/mean carries an honest uncertainty band (Wilson / SEM / bootstrap) and
its sample size. greedy_thinking / thinking readouts are plotted only if present;
the cheap stability collection disables them, so they are normally absent.

Usage:
  uv run python sesgo/stability/visualize_stability_samples.py \
      out/sesgo/stability/Qwen3-0.6B/response_samples.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import seaborn as sns  # noqa: E402

# Repo root (for `src...`) and this dir (for the sibling helper modules) on path.
_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parent))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402
from stability_metrics_helpers import (  # noqa: E402
    AXES,
    CONDITIONS,
    accuracy_count,
    consistency_set,
    flip_rate,
    p_unknown_spread,
)
from stability_abstention_plot import plot_p_unknown_spread  # noqa: E402
from stability_consistency_plot import plot_consistency  # noqa: E402
from stability_sensitivity_plot import (  # noqa: E402
    plot_accuracy,
    plot_format_sensitivity,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for stability visualization."""
    parser = argparse.ArgumentParser(
        description="Compute and plot all STABILITY statistics for a SesgoDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("samples", type=Path, help="response_samples.json (a SesgoDataset)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"),
        help="Base output dir; plots land at <out-dir>/sesgo/stability/<MODEL>/plots/")
    return parser.parse_args()


def _log_stats(dataset: SesgoDataset, model: str) -> None:
    """Log the headline stability numbers (n's + per-condition means) to console."""
    log_section("STABILITY STATS")
    n_items = len({(s.question_id, s.context_condition) for s in dataset.samples})
    log(f"  samples={len(dataset.samples)}  distinct (item,condition)={n_items}  model={model}")
    for cond in CONDITIONS:
        cs3 = consistency_set(dataset, cond, two_opt=False)
        acc3 = accuracy_count(dataset, cond, two_opt=False)
        acc2 = accuracy_count(dataset, cond, two_opt=True)
        mean_c = f"{sum(cs3.consistency)/len(cs3.consistency):.0%}" if cs3.consistency else "n/a"
        log(f"  [{cond}] groups={len(cs3.consistency)} mean 3-opt consistency={mean_c}"
            f"  acc 3-opt={acc3.correct}/{acc3.total}"
            f"  acc 2-opt={acc2.correct}/{acc2.total}")


def main() -> None:
    """Load the SesgoDataset, compute every stability statistic, plot, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO STABILITY")

    dataset = SesgoDataset.from_json(args.samples)
    model = dataset.model_name
    log(f"[viz] loaded {len(dataset.samples)} samples (model={model})")
    _log_stats(dataset, model)

    # Per-readout, per-condition metrics (consistency unit = format group of 18).
    three_opt = {c: consistency_set(dataset, c, two_opt=False) for c in CONDITIONS}
    two_opt = {c: consistency_set(dataset, c, two_opt=True) for c in CONDITIONS}
    flips = {c: [flip_rate(dataset, a, c) for a in AXES] for c in CONDITIONS}
    spreads = {c: p_unknown_spread(dataset, c) for c in CONDITIONS}
    acc_counts = [
        accuracy_count(dataset, c, t) for t in (False, True) for c in CONDITIONS
    ]

    plots_dir = args.out_dir / "sesgo" / "stability" / model / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    written = [
        plot_consistency(three_opt, two_opt, model, plots_dir / "consistency.png"),
        plot_format_sensitivity(flips, model, plots_dir / "format_sensitivity.png"),
        plot_p_unknown_spread(spreads, model, plots_dir / "p_unknown_spread.png"),
        plot_accuracy(acc_counts, model, plots_dir / "accuracy.png"),
    ]
    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
