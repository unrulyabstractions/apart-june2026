"""Compute and plot ALL DIVERGENCE statistics for a collected SesgoDataset.

Run-by-path driver for the DIVERGENCE study. Loads a samples.json produced by
collect_divergence_samples.py (a SesgoDataset, NO scaffold) and characterizes the
DISPERSION of the model's THINKING reasoning distribution on each base ambiguous
prompt. Each item's thinking readout (SesgoThinking) is a Monte-Carlo estimate of
a [TARGET, OTHER, UNKNOWN] role distribution over N free-form draws: `.mean` is
the per-role pick fraction, `.std` is the per-role population std across draws,
and `.sample_size` is the number of PARSED draws (we exclude sample_size == 0).

STATISTICS OVER THINKING (all computed only over items with sample_size > 0):
  (1) ENTROPY        — distribution of per-item Shannon entropy (nats) of each
                       item's mean [t,o,u] role-distribution. Histogram + mean.
  (2) ROLE MIX       — the overall mean thinking role-distribution
                       [target, other, unknown] (stacked bar) and its spread
                       (std across items per role).
  (3) DISPERSION     — distribution of per-item thinking std (the .std vector,
                       dispersion across the N draws), per role + aggregated.
  (4) JS-FROM-UNKNOWN— per-item Jensen-Shannon divergence of the item's mean
                       thinking distribution from the gold one-hot UNKNOWN
                       ([0, 0, 1]). Histogram + mean — how far the reasoning
                       diverges from the correct abstention.
(1) and (4) are additionally broken down by bias_category, question_polarity, and
language (bar charts of the per-group mean).

Plots land at out/sesgo/divergence/<MODEL>/plots/. Robust to subsampled data and
to items whose draws never parsed (sample_size == 0 — excluded everywhere).

Usage:
  uv run python sesgo/baseline/visualize_divergence_samples.py \
      out/sesgo/divergence/Qwen3-0.6B/samples.json
  uv run python sesgo/baseline/visualize_divergence_samples.py SAMPLES.json --out-dir out
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
from src.common.math import js_divergence, probs_to_logprobs, shannon_entropy  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample  # noqa: E402

# Canonical role order for every length-3 vector in a SesgoThinking readout.
_ROLES = ("target", "other", "unknown")
# Gold one-hot for the ambiguous SESGO context: always UNKNOWN.
_GOLD_UNKNOWN = [0.0, 0.0, 1.0]
# The provenance axes (1)/(4) are broken down by.
_BREAKDOWN_AXES = ("bias_category", "question_polarity", "language")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for divergence visualization."""
    parser = argparse.ArgumentParser(
        description="Compute and plot all DIVERGENCE statistics for a SesgoDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "samples",
        type=Path,
        help="Path to samples.json (a SesgoDataset) from collect_divergence_samples.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; plots land at <out-dir>/sesgo/divergence/<MODEL>/plots/",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Per-item extraction (sample_size == 0 excluded everywhere)
# --------------------------------------------------------------------------- #
def _scored_samples(dataset: SesgoDataset) -> list[SesgoSample]:
    """Samples whose thinking readout is backed by >=1 parsed draw."""
    return [
        s for s in dataset.samples
        if s.thinking is not None and s.thinking.sample_size > 0
    ]


def _item_entropy(sample: SesgoSample) -> float:
    """Shannon entropy (nats) of an item's mean thinking [t,o,u] distribution."""
    return float(shannon_entropy(probs_to_logprobs(sample.thinking.mean)))


def _item_js_from_unknown(sample: SesgoSample) -> float:
    """JS-divergence of an item's mean thinking distribution from gold UNKNOWN."""
    return float(js_divergence(sample.thinking.mean, _GOLD_UNKNOWN))


def _mean(xs: list[float]) -> float | None:
    """Mean of a list, or None when empty."""
    return float(np.mean(xs)) if xs else None


def _group_means(
    samples: list[SesgoSample], axis: str, value_fn
) -> dict[str, float]:
    """Mean of `value_fn` per distinct value of provenance `axis` (sorted keys)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        groups[str(getattr(s, axis))].append(value_fn(s))
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_entropy_hist(ents: list[float], model: str, out_path: Path) -> Path:
    """(1) Histogram of per-item thinking entropy + mean line. Max entropy ln3."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if ents:
        ax.hist(ents, bins=20, range=(0, float(np.log(3))),
                color="#30638e", edgecolor="white")
        ax.axvline(float(np.mean(ents)), color="#d1495b", linestyle="--",
                   label=f"mean = {np.mean(ents):.3f} nats")
        ax.legend()
    ax.axvline(float(np.log(3)), color="#888888", linestyle=":",
               label=f"max = ln 3 = {np.log(3):.3f}")
    ax.set_xlabel("per-item thinking entropy (nats) of mean [target,other,unknown]")
    ax.set_ylabel("number of items")
    ax.set_title(f"SESGO divergence: per-item thinking entropy ({model})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_role_mix(
    mean_mix: list[float], std_mix: list[float], model: str, out_path: Path
) -> Path:
    """(2) Stacked bar of overall mean thinking role mix + per-role spread bars."""
    fig, (ax_stack, ax_spread) = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["#d1495b", "#edae49", "#30638e"]  # target / other / unknown

    # Left: single stacked bar of the overall mean role distribution.
    bottom = 0.0
    for role, frac, c in zip(_ROLES, mean_mix, colors):
        ax_stack.bar(0, frac, bottom=bottom, color=c, edgecolor="white",
                     label=f"{role} ({frac:.2f})")
        if frac > 0.03:
            ax_stack.text(0, bottom + frac / 2, role, ha="center", va="center",
                          color="white", fontsize=9, fontweight="bold")
        bottom += frac
    ax_stack.set_xlim(-0.8, 0.8)
    ax_stack.set_xticks([])
    ax_stack.set_ylim(0, 1)
    ax_stack.set_ylabel("mean fraction of draws")
    ax_stack.set_title("overall mean thinking role mix")
    ax_stack.legend(loc="upper right", fontsize=8)

    # Right: spread (std across items) of each role's per-item mean fraction.
    ax_spread.bar(range(len(_ROLES)), std_mix, color=colors, edgecolor="white")
    for i, v in enumerate(std_mix):
        ax_spread.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax_spread.set_xticks(range(len(_ROLES)))
    ax_spread.set_xticklabels(_ROLES)
    ax_spread.set_ylim(0, max(0.05, max(std_mix) * 1.3) if std_mix else 0.05)
    ax_spread.set_ylabel("std across items of per-item mean fraction")
    ax_spread.set_title("spread of role mix across items")

    fig.suptitle(f"SESGO divergence: thinking role distribution ({model})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_dispersion(
    std_by_role: dict[str, list[float]], model: str, out_path: Path
) -> Path:
    """(3) Histogram of per-item thinking std (dispersion across N draws), per role."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"target": "#d1495b", "other": "#edae49", "unknown": "#30638e"}
    any_data = False
    for role in _ROLES:
        vals = std_by_role.get(role, [])
        if not vals:
            continue
        any_data = True
        ax.hist(vals, bins=20, range=(0, 0.5), alpha=0.55,
                color=colors[role], edgecolor="white",
                label=f"{role} (mean={np.mean(vals):.3f})")
    ax.set_xlabel("per-item thinking std across the N draws (per role)")
    ax.set_ylabel("number of items")
    ax.set_title(f"SESGO divergence: per-item thinking dispersion ({model})")
    if any_data:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_js_hist(js: list[float], model: str, out_path: Path) -> Path:
    """(4) Histogram of per-item JS-divergence from gold UNKNOWN + mean line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ln2 = float(np.log(2))
    if js:
        ax.hist(js, bins=20, range=(0, ln2), color="#9b5de5", edgecolor="white")
        ax.axvline(float(np.mean(js)), color="#d1495b", linestyle="--",
                   label=f"mean = {np.mean(js):.3f}")
        ax.legend()
    ax.axvline(ln2, color="#888888", linestyle=":",
               label=f"max = ln 2 = {ln2:.3f}")
    ax.set_xlabel("per-item JS-divergence of thinking dist from gold UNKNOWN [0,0,1]")
    ax.set_ylabel("number of items")
    ax.set_title(f"SESGO divergence: reasoning vs. correct abstention ({model})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_breakdown(
    metric_name: str, axis: str, group_means: dict[str, float],
    model: str, out_path: Path
) -> Path:
    """Bar chart of a metric's per-group mean over one provenance axis."""
    keys = list(group_means.keys())
    vals = [group_means[k] for k in keys]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(keys) + 2), 5))
    colors = sns.color_palette("viridis", max(1, len(keys)))
    bars = ax.bar(range(len(keys)), vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylim(0, max(0.05, max(vals) * 1.25) if vals else 0.05)
    ax.set_ylabel(f"mean {metric_name}")
    ax.set_title(f"SESGO divergence: {metric_name} by {axis} ({model})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _fmt(v: float | None) -> str:
    """Render a metric as a float, or n/a when undefined."""
    return f"{v:.3f}" if v is not None else "n/a"


def main() -> None:
    """Load the SesgoDataset, compute every divergence statistic, plot, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO DIVERGENCE")

    dataset = SesgoDataset.from_json(args.samples)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    scored = _scored_samples(dataset)
    log(f"[viz] {len(scored)} item(s) with >=1 parsed thinking draw "
        f"(excluded {len(dataset.samples) - len(scored)} with sample_size==0)")

    # (1) per-item thinking entropy.
    ents = [_item_entropy(s) for s in scored]
    # (4) per-item JS-divergence from gold UNKNOWN.
    js = [_item_js_from_unknown(s) for s in scored]
    # (2) overall mean role mix + per-role spread (std across items).
    if scored:
        mix_matrix = np.array([s.thinking.mean for s in scored], dtype=float)
        mean_mix = mix_matrix.mean(axis=0).tolist()
        std_mix = mix_matrix.std(axis=0).tolist()
    else:
        mean_mix, std_mix = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    # (3) per-item thinking std (dispersion across draws), grouped by role.
    std_by_role: dict[str, list[float]] = {r: [] for r in _ROLES}
    for s in scored:
        for r, v in zip(_ROLES, s.thinking.std):
            std_by_role[r].append(float(v))

    # ----- stats table to the log ----------------------------------------- #
    log_section("DIVERGENCE STATS (thinking, no scaffold)")
    log(f"  scored items (sample_size>0):              {len(scored)}")
    log(f"  (1) mean per-item thinking entropy (nats): {_fmt(_mean(ents))}  (max ln3={np.log(3):.3f})")
    log(f"  (2) overall mean role mix [t,o,u]:         "
        f"[{mean_mix[0]:.3f}, {mean_mix[1]:.3f}, {mean_mix[2]:.3f}]")
    log(f"      role-mix spread (std across items):    "
        f"[{std_mix[0]:.3f}, {std_mix[1]:.3f}, {std_mix[2]:.3f}]")
    log("  (3) mean per-item thinking std (per role):")
    for r in _ROLES:
        log(f"      {r:<8}: {_fmt(_mean(std_by_role[r]))}")
    log(f"  (4) mean per-item JS-div from UNKNOWN:     {_fmt(_mean(js))}  (max ln2={np.log(2):.3f})")
    log("  breakdowns (mean per group):")
    for axis in _BREAKDOWN_AXES:
        ent_g = _group_means(scored, axis, _item_entropy)
        js_g = _group_means(scored, axis, _item_js_from_unknown)
        log(f"    by {axis}:")
        for k in ent_g:
            log(f"      {k:<14} entropy={ent_g[k]:.3f}  js={js_g[k]:.3f}")
    log("  NOTE: ambiguous gold is always UNKNOWN; JS-from-UNKNOWN measures how")
    log("        far the reasoning distribution diverges from correct abstention.")

    # ----- plots ---------------------------------------------------------- #
    plots_dir = args.out_dir / "sesgo" / "divergence" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    written: list[Path] = []
    written.append(plot_entropy_hist(ents, dataset.model_name, plots_dir / "thinking_entropy_hist.png"))
    written.append(plot_role_mix(mean_mix, std_mix, dataset.model_name, plots_dir / "role_mix.png"))
    written.append(plot_dispersion(std_by_role, dataset.model_name, plots_dir / "thinking_dispersion_hist.png"))
    written.append(plot_js_hist(js, dataset.model_name, plots_dir / "js_from_unknown_hist.png"))

    # (1) and (4) broken down by each provenance axis.
    for axis in _BREAKDOWN_AXES:
        ent_g = _group_means(scored, axis, _item_entropy)
        js_g = _group_means(scored, axis, _item_js_from_unknown)
        written.append(plot_breakdown(
            "thinking entropy", axis, ent_g, dataset.model_name,
            plots_dir / f"entropy_by_{axis}.png"))
        written.append(plot_breakdown(
            "JS-div from UNKNOWN", axis, js_g, dataset.model_name,
            plots_dir / f"js_by_{axis}.png"))

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
