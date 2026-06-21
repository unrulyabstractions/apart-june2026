"""Compute, plot, and SELECT the best scaffold for a collected SesgoDataset.

Run-by-path driver for the SELECTION study. Loads a response_samples.json produced
by collect_selection_samples.py (a SesgoDataset spanning the no-scaffold baseline
plus four debiasing scaffolds, BOTH context conditions, four readout levels) and
answers one question: which scaffold best debiases the model? On AMBIGUOUS items the
gold is UNKNOWN, so accuracy IS abstention; on DISAMBIGUATED items the gold is the
ground-truth role, so accuracy is picking it. We rank scaffolds per condition and
SELECT the best — on ambiguous abstention at the thinking level (the model's actual
free-form behavior), falling back to non-thinking when no thinking draw survived.

Every figure is ONE file with Wilson 95% CIs and sample sizes on every bar:
  accuracy_by_scaffold_ambig.png     - readout LEVELS stacked as subfigure rows
  accuracy_by_scaffold_disambig.png    (non-thinking / 2-option / greedy-thinking /
                                        thinking), the SELECTED scaffold starred.
  two_vs_three_option__<category>.png- per category, 2-OPTION over 3-OPTION accuracy
                                        on disambiguated items, stacked.

Usage:
  uv run python sesgo/selection/visualize_selection_samples.py \
      out/sesgo/selection/Qwen3-0.6B/response_samples.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import seaborn as sns  # noqa: E402

# Bootstrap repo root + this dir onto sys.path so src.* and the sibling helper
# modules both resolve regardless of cwd (parents[2] is the repo root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.common.logging import log, log_box, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402
from selection_metrics_helpers import (  # noqa: E402
    LEVELS,
    all_scaffolds,
    counts_by_scaffold,
)
from selection_figure_helpers import (  # noqa: E402
    category_has_data,
    figure_accuracy_by_scaffold,
    figure_two_vs_three_option,
)

# The thinking level is the SELECT target (the model's free-form behavior); fall
# back to non-thinking when no thinking draw survived in this run.
_AMBIG_LEVELS = ["non_thinking", "greedy_thinking", "thinking"]
_DISAMBIG_LEVELS = list(LEVELS)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for selection visualization."""
    parser = argparse.ArgumentParser(
        description="Compute, plot, and SELECT the best scaffold for a SesgoDataset",
    )
    parser.add_argument(
        "samples", type=Path,
        help="Path to response_samples.json (a SesgoDataset) from collect_selection_samples.py",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"),
        help="Base output dir; plots land at <out-dir>/sesgo/selection/<MODEL>/plots/",
    )
    return parser.parse_args()


def _select_level(dataset: SesgoDataset, context: str, preferred: list[str]) -> str:
    """First level in ``preferred`` that has data for this context (else its head)."""
    for level in preferred:
        counts = counts_by_scaffold(dataset, level, context)
        if any(c.total > 0 for c in counts.values()):
            return level
    return preferred[0]


def _log_ranking(dataset: SesgoDataset, scaffolds, context, level, selected) -> None:
    """Log the per-scaffold accuracy table for one context at the SELECT level."""
    counts = counts_by_scaffold(dataset, level, context)
    log_section(f"PER-SCAFFOLD ({context}, select level = {level})")
    for sc in scaffolds:
        c = counts.get(sc)
        txt = "n/a" if not c or c.total == 0 else f"{c.rate:.1%} ({c.correct}/{c.total})"
        star = "  <- SELECTED" if sc == selected else ""
        log(f"  {sc:<36} {txt:>16}{star}")


def main() -> None:
    """Load the SesgoDataset, plot the stacked figures, SELECT, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO SELECTION")

    dataset = SesgoDataset.from_json(args.samples)
    scaffolds = all_scaffolds(dataset)
    by_cond = Counter(s.context_condition for s in dataset.samples)
    log(f"[viz] {len(dataset.samples)} samples · {len(scaffolds)} scaffold(s) · "
        f"conditions {dict(by_cond)} · model={dataset.model_name}")

    plots_dir = args.out_dir / "sesgo" / "selection" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    written: list[Path] = []

    # ----- per-scaffold accuracy, readout levels stacked, one fig per condition.
    context_levels = {"ambig": _AMBIG_LEVELS, "disambig": _DISAMBIG_LEVELS}
    for context, levels in context_levels.items():
        if by_cond.get(context, 0) == 0:
            continue
        sel_level = _select_level(dataset, context, levels)
        path, ranking = figure_accuracy_by_scaffold(
            dataset, scaffolds, context, levels, sel_level, dataset.model_name,
            plots_dir / f"accuracy_by_scaffold_{context}.png",
        )
        written.append(path)
        _log_ranking(dataset, scaffolds, context, sel_level, ranking.selected)
        log_box(f"SELECTED ({context}): {ranking.selected or 'n/a'}", gap=1)

    # ----- 2-option vs 3-option stacked, one fig per bias_category (disambig only).
    if by_cond.get("disambig", 0):
        categories = sorted(
            {s.bias_category for s in dataset.samples if s.context_condition == "disambig"}
        )
        for cat in categories:
            if not category_has_data(dataset, cat):
                log(f"[viz] skip 2-vs-3 for '{cat}' (too few disambig samples)")
                continue
            written.append(
                figure_two_vs_three_option(
                    dataset, scaffolds, cat, dataset.model_name,
                    plots_dir / f"two_vs_three_option__{cat}.png",
                )
            )

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
