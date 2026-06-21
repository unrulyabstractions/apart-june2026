"""Cross-model DISTRIBUTION figures for the SESGO baseline SIZE SWEEP.

Run-by-path driver that scans ``out/sesgo/baseline/*/response_samples.json`` (each
subdir is one model named by its bare repo id) and renders a rich set of
cross-model DISTRIBUTIONAL comparisons — not just accuracy lines but the full
shapes: outcome (role) mass, abstention spread, per-category abstention, the
target-vs-other bias gap, three-readout agreement, and disambiguated accuracy
spread. It scales to whatever model dirs exist, flags partial runs, and skips dirs
it can't size/place.

Usage:
  uv run python sesgo/baseline/visualize_cross_model_distributions.py
  uv run python sesgo/baseline/visualize_cross_model_distributions.py --base-dir out/sesgo/baseline
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.math import wilson_interval  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402

from sesgo.baseline.cross_model_distribution_builder import (  # noqa: E402
    build_model_distribution,
)
from sesgo.baseline.cross_model_distribution_stats import (  # noqa: E402
    ModelDistribution,
    is_degenerate_readout,
)
from sesgo.baseline.cross_model_outcome_plots import (  # noqa: E402
    plot_outcome_distribution,
    plot_readout_agreement,
    plot_target_other_gap,
)
from sesgo.baseline.cross_model_plot_styles import is_partial, order_by_size  # noqa: E402
from sesgo.baseline.cross_model_spread_plots import (  # noqa: E402
    plot_abstention_spread,
    plot_category_heatmap,
    plot_disambig_accuracy_spread,
)

_SAMPLES_FILE = "response_samples.json"

# Filename -> plotter. Stable order so the log reads top-to-bottom like the report.
_FIGURES = (
    ("outcome_distribution.png", plot_outcome_distribution),
    ("abstention_spread.png", plot_abstention_spread),
    ("category_abstention_heatmap.png", plot_category_heatmap),
    ("target_other_gap.png", plot_target_other_gap),
    ("readout_agreement.png", plot_readout_agreement),
    ("disambig_accuracy_spread.png", plot_disambig_accuracy_spread),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the cross-model distribution visualization."""
    parser = argparse.ArgumentParser(
        description="Cross-model SESGO baseline DISTRIBUTION figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir", type=Path, default=Path("out/sesgo/baseline"),
        help="Dir whose immediate subdirs own per-model response_samples.json",
    )
    return parser.parse_args()


def _model_dirs(base_dir: Path) -> list[Path]:
    """Per-model sample files: every ``<base>/<model>/response_samples.json``."""
    return sorted(p for p in base_dir.glob(f"*/{_SAMPLES_FILE}") if p.is_file())


def _collect(sample_files: list[Path]) -> list[ModelDistribution]:
    """Load each model dataset and reduce it to its distribution bundle."""
    out: list[ModelDistribution] = []
    for path in sample_files:
        dataset = SesgoDataset.from_json(path)
        dist = build_model_distribution(dataset.model_name, dataset.samples)
        if dist is None:
            ambig = [s for s in dataset.samples if s.context_condition == "ambig"]
            reason = ("DEGENERATE 3-opt readout (all-uniform logits)"
                      if is_degenerate_readout(ambig) else "unknown size/family")
            log(f"  {dataset.model_name:<34} SKIPPED — {reason}")
            continue
        flag = "  PARTIAL" if is_partial(dist) else ""
        log(f"  {dataset.model_name:<34} ambig={dist.n_ambig:>4} "
            f"disambig={dist.n_disambig:>4}{flag}")
        out.append(dist)
    return out


def _log_summary(models: list[ModelDistribution]) -> None:
    """Emit the per-model headline distribution numbers (size-ordered)."""
    log_section("CROSS-MODEL DISTRIBUTION SUMMARY (size-ordered)")
    for m in order_by_size(models):
        gap = (wilson(m.target_succ, m.target_total)
               - wilson(m.other_succ, m.other_total))
        log(f"  {m.model:<24} {m.params_b:>5}B  "
            f"unknown_mass={m.mean_role_mass[2]:.3f}  "
            f"abstain_3opt={wilson(m.abstain_3opt_succ, m.abstain_3opt_total):.2%} "
            f"greedy={wilson(m.abstain_greedy_succ, m.abstain_greedy_total):.2%}  "
            f"tgt-oth_gap={gap:+.2%}")


def wilson(succ: int, total: int) -> float:
    """Point estimate (Wilson center is for bars); 0.0 when there is no data."""
    p, _, _ = wilson_interval(succ, total)
    return 0.0 if p != p else p  # NaN guard for empty cells


def main() -> None:
    """Scan model dirs, build distribution bundles, render every figure, log."""
    args = parse_args()
    log_header("VISUALIZE SESGO BASELINE CROSS-MODEL DISTRIBUTIONS")

    sample_files = _model_dirs(args.base_dir)
    log(f"[viz] found {len(sample_files)} model dir(s) under {args.base_dir}")
    models = _collect(sample_files)
    if not models:
        log("[viz] no sizable/placeable models — nothing to plot")
        return
    _log_summary(models)

    plots_dir = args.base_dir / "cross_model" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    log_section(f"RENDERING {len(_FIGURES)} CROSS-MODEL DISTRIBUTION FIGURES")
    for filename, plotter in _FIGURES:
        out_path = plots_dir / filename
        plotter(models, out_path)
        log(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
