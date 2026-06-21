"""Plot the model's SYSTEM DEFAULT + per-condition accuracy for a SesgoDataset.

Run-by-path driver for the DIVERGENCE study. Loads a response_samples.json from
collect_divergence_samples.py (a SesgoDataset, NO scaffold) over Spanish ORIGINAL
SESGO items in BOTH context conditions (ambiguous + disambiguated) and renders a
publication-quality plot suite (concept from arXiv:2601.06116 — "system default").

Two complementary views:
  * SYSTEM DEFAULT (ambiguous items): the [target, other, unknown] mix the model
    falls into over N sampled thinking draws. The ambiguous gold is one-hot
    UNKNOWN ("safe default"), so distance from it = how far the default deviates
    from correct abstention. Every mean carries a bootstrap 95% CI; n on every panel.
  * PER-CONDITION ACCURACY: the four readouts — non-thinking 3-option, 2-option
    forced choice (no UNKNOWN), greedy-thinking baseline, sampled thinking —
    scored against the per-condition gold (ambiguous: abstain; disambiguated: the
    ground-truth role), with Wilson score CIs and n on every bar.

PLOTS (land at out/sesgo/divergence/<MODEL>/plots/):
  accuracy_by_readout.png  Stacked ambig/disambig subfigures; 4 readouts; Wilson CIs.
  role_mix.png             HERO — mean system-default role mix + bootstrap CIs.
  default_per_item.png     Per-item UNKNOWN-fraction strip + density.
  default_uncertainty.png  Per-item Shannon entropy + bootstrap-CI mean.
  default_deviation.png    Per-item JS-divergence lollipop + bootstrap-CI mean.
  default_dispersion.png   Per-item across-draw std per role + bootstrap-CI means.
  <metric>_by_<axis>.png   uncertainty + deviation by category/polarity/language.

Usage:
  uv run python sesgo/divergence/visualize_divergence_samples.py \
      out/sesgo/divergence/Qwen3-0.6B/response_samples.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402
from sesgo.divergence.divergence_accuracy_panels import plot_accuracy_by_readout  # noqa: E402
from sesgo.divergence.divergence_breakdown_panels import plot_breakdown  # noqa: E402
from sesgo.divergence.divergence_default_panels import (  # noqa: E402
    plot_default_per_item,
    plot_role_mix,
)
from sesgo.divergence.divergence_spread_panels import (  # noqa: E402
    plot_default_deviation,
    plot_default_uncertainty,
    plot_dispersion,
)
from sesgo.divergence.divergence_item_metrics import (  # noqa: E402
    ambig_scored,
    group_values,
    item_deviation,
    item_entropy,
    scored_samples,
    split_by_condition,
)
from sesgo.divergence.divergence_plot_styles import (  # noqa: E402
    BREAKDOWN_AXES,
    LN2,
    LN3,
    ROLES,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for divergence visualization."""
    parser = argparse.ArgumentParser(
        description="Plot the SESGO system default + per-condition accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "samples", type=Path,
        help="Path to response_samples.json from collect_divergence_samples.py",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"),
        help="Base output dir; plots land at <out-dir>/sesgo/divergence/<MODEL>/plots/",
    )
    return parser.parse_args()


def _fmt(v: float | None) -> str:
    """Render a metric as a float, or n/a when undefined."""
    return f"{v:.3f}" if v is not None else "n/a"


def _log_stats(ambig, by_cond, ents, dev, mean_mix, std_by_role) -> None:
    """Emit the full divergence stats table to the log."""
    log_section("SYSTEM-DEFAULT STATS (thinking, ambiguous items)")
    log(f"  ambiguous scored items:                      {len(ambig)}")
    log(f"  default-uncertainty (mean entropy, nats):    {_fmt(float(np.mean(ents)) if ents else None)}  (max ln3={LN3:.3f})")
    log(f"  mean default role mix [t,o,u]:               "
        f"[{mean_mix[0]:.3f}, {mean_mix[1]:.3f}, {mean_mix[2]:.3f}]")
    log(f"  default-deviation (mean JS from UNKNOWN):    {_fmt(float(np.mean(dev)) if dev else None)}  (max ln2={LN2:.3f})")
    log("  mean across-draw instability (per role):")
    for r in ROLES:
        vals = std_by_role[r]
        log(f"    {r:<8}: {_fmt(float(np.mean(vals)) if vals else None)}")
    log_section("PER-CONDITION ACCURACY (all scored items)")
    for cond, samples in by_cond.items():
        log(f"  {cond} (n={len(samples)} items):")
        for readout, fn in (("non_thinking", lambda s: s.correct_non_thinking),
                            ("2opt", lambda s: s.correct_2opt),
                            ("greedy_thinking", lambda s: s.correct_greedy_thinking),
                            ("thinking", lambda s: s.correct_thinking)):
            scored = [fn(s) for s in samples if fn(s) is not None]
            acc = sum(bool(x) for x in scored) / len(scored) if scored else None
            log(f"    {readout:<16}: {_fmt(acc)} (n={len(scored)})")


def main() -> None:
    """Load the SesgoDataset, compute every statistic, plot, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO DIVERGENCE")

    dataset = SesgoDataset.from_json(args.samples)
    scored = scored_samples(dataset)
    ambig = ambig_scored(scored)
    by_cond = split_by_condition(scored)
    log(f"[viz] {len(dataset.samples)} samples; {len(scored)} with >=1 parsed draw "
        f"({len(ambig)} ambiguous, {len(scored) - len(ambig)} disambiguated)")

    ents = [item_entropy(s) for s in ambig]
    dev = [item_deviation(s) for s in ambig]
    mean_mix = (np.array([s.thinking.mean for s in ambig]).mean(axis=0).tolist()
                if ambig else [0.0, 0.0, 0.0])
    std_by_role: dict[str, list[float]] = {r: [] for r in ROLES}
    for s in ambig:
        for r, v in zip(ROLES, s.thinking.std):
            std_by_role[r].append(float(v))
    _log_stats(ambig, by_cond, ents, dev, mean_mix, std_by_role)

    plots = args.out_dir / "sesgo" / "divergence" / dataset.model_name / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    m = dataset.model_name

    written = [
        plot_accuracy_by_readout(by_cond, m, plots / "accuracy_by_readout.png"),
        plot_role_mix(mean_mix, len(ambig), ambig, m, plots / "role_mix.png"),
        plot_default_per_item(by_cond, m, plots / "default_per_item.png"),
        plot_default_uncertainty(ents, m, plots / "default_uncertainty.png"),
        plot_default_deviation(ambig, m, plots / "default_deviation.png"),
        plot_dispersion(std_by_role, m, plots / "default_dispersion.png"),
    ]
    for axis in BREAKDOWN_AXES:
        ent_groups = group_values(ambig, axis, item_entropy)
        dev_groups = group_values(ambig, axis, item_deviation)
        if len(ent_groups) < 2:  # degenerate single-group axis (e.g. one language)
            log(f"[viz] skipping {axis} breakdown — single group "
                f"({', '.join(f'{k}(n={len(v)})' for k, v in ent_groups.items())})")
            continue
        written.append(plot_breakdown("default-uncertainty", axis, ent_groups, m,
                       plots / f"uncertainty_by_{axis}.png", vmax=LN3 + 0.06))
        written.append(plot_breakdown("default-deviation", axis, dev_groups, m,
                       plots / f"deviation_by_{axis}.png", vmax=LN2 + 0.04))

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
