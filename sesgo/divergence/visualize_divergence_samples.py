"""Plot the model's SYSTEM DEFAULT distribution for a collected SesgoDataset.

Run-by-path driver for the DIVERGENCE study. Loads a response_samples.json produced by
collect_divergence_samples.py (a SesgoDataset, NO scaffold) and characterizes the
model's SYSTEM DEFAULT over sampled thinking draws on each base ambiguous prompt
(concept from arXiv:2601.06116 — "system default / default uncertainty").

Each item's thinking readout (SesgoThinking) is a Monte-Carlo estimate of a
[TARGET, OTHER, UNKNOWN] role distribution over N free-form draws: `.mean` is the
per-role pick fraction (the item's DEFAULT mix), `.std` is the per-role dispersion
across draws, and `.sample_size` is the number of PARSED draws (we exclude
sample_size == 0). The ambiguous gold is ALWAYS one-hot UNKNOWN [0,0,1] — the
"safe default" — so distance from it = how far the system default deviates from
correct abstention.

PLOTS (every metric over items with sample_size > 0):
  role_mix.png        HERO — mean system-default role mix + spread (std/role).
  default_per_item    Per-item UNKNOWN-fraction strip + density: how the default
                      varies across items (does it abstain by default?).
  default_uncertainty Per-item Shannon entropy (nats) of the default mix — how
                      UNCERTAIN the system default is per item.
  default_deviation   Per-item JS-divergence from the safe default [0,0,1] — how
                      far the default DEVIATES from correct abstention, colored by
                      the role that pulls it away.
  dispersion          Per-item across-draw std (instability of the default), per role.
  <metric>_by_<axis>  default-uncertainty and default-deviation broken down by
                      bias_category / question_polarity / language (n annotated).

Plots land at out/sesgo/divergence/<MODEL>/plots/. Robust to subsampled / small-n
data and to items whose draws never parsed (sample_size == 0 — excluded everywhere).

Usage:
  uv run python sesgo/divergence/visualize_divergence_samples.py \
      out/sesgo/divergence/Qwen3-0.6B/response_samples.json
  uv run python sesgo/divergence/visualize_divergence_samples.py SAMPLES.json --out-dir out
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
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.math import js_divergence, probs_to_logprobs, shannon_entropy  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample  # noqa: E402

# Canonical role order for every length-3 vector in a SesgoThinking readout.
_ROLES = ("target", "other", "unknown")
# Gold one-hot for the ambiguous SESGO context: always UNKNOWN ("safe default").
_GOLD_UNKNOWN = [0.0, 0.0, 1.0]
# Colorblind-safe per-role palette (Okabe-Ito): target / other / unknown.
_ROLE_COLORS = {"target": "#D55E00", "other": "#E69F00", "unknown": "#0072B2"}
_ACCENT = "#CC79A7"  # mean / deviation accent
_REF = "#555555"     # reference / max lines
_BREAKDOWN_AXES = ("bias_category", "question_polarity", "language")
_SUBTITLE = "system default over sampled thinking draws (arXiv:2601.06116)"
_LN3, _LN2 = float(np.log(3)), float(np.log(2))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for divergence visualization."""
    parser = argparse.ArgumentParser(
        description="Plot the SYSTEM DEFAULT distribution for a SesgoDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "samples", type=Path,
        help="Path to response_samples.json (a SesgoDataset) from collect_divergence_samples.py",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"),
        help="Base output dir; plots land at <out-dir>/sesgo/divergence/<MODEL>/plots/",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Per-item extraction (sample_size == 0 excluded everywhere)
# --------------------------------------------------------------------------- #
def _scored_samples(dataset: SesgoDataset) -> list[SesgoSample]:
    """Samples whose thinking readout is backed by >=1 parsed draw."""
    return [s for s in dataset.samples
            if s.thinking is not None and s.thinking.sample_size > 0]


def _item_entropy(s: SesgoSample) -> float:
    """Shannon entropy (nats) of an item's default [t,o,u] mix."""
    return float(shannon_entropy(probs_to_logprobs(s.thinking.mean)))


def _item_deviation(s: SesgoSample) -> float:
    """JS-divergence of an item's default mix from the safe default [0,0,1]."""
    return float(js_divergence(s.thinking.mean, _GOLD_UNKNOWN))


def _mean(xs: list[float]) -> float | None:
    """Mean of a list, or None when empty."""
    return float(np.mean(xs)) if xs else None


def _group_means(samples, axis, value_fn) -> dict[str, list[float]]:
    """Per-group list of `value_fn` values keyed by provenance `axis` (sorted)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        groups[str(getattr(s, axis))].append(value_fn(s))
    return dict(sorted(groups.items()))


# --------------------------------------------------------------------------- #
# Shared drawing helpers (small-n friendly: strip + density, never lonely bars)
# --------------------------------------------------------------------------- #
def _new_fig(w: float = 8.0, h: float = 5.0):
    """One axes with constrained layout (no clipped titles) at publication dpi."""
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
    return fig, ax


def _save(fig, path: Path) -> Path:
    """Save tight at dpi=150 and close — never leak a figure handle."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _titles(ax, title: str, sub: str | None = None) -> None:
    """Bold title with the paper-nodding subtitle, padded clear of the data."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=22)
    ax.text(0.5, 1.012, sub or _SUBTITLE, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#555555", style="italic")


def _jittered_strip(ax, xs, *, y0: float, color: str, n_label: bool = True) -> None:
    """Draw values as a horizontal jittered strip with a faint density curve."""
    arr = np.asarray(xs, dtype=float)
    rng = np.random.default_rng(0)
    jit = (rng.random(arr.size) - 0.5) * 0.12
    if arr.size >= 3 and np.unique(arr).size >= 2:  # density only if it means something
        lo, hi = float(arr.min()), float(arr.max())
        pad = max(1e-3, (hi - lo) * 0.15)
        grid = np.linspace(lo - pad, hi + pad, 200)
        try:
            dens = gaussian_kde(arr)(grid)
            dens = 0.32 * dens / dens.max()
            ax.fill_between(grid, y0, y0 + dens, color=color, alpha=0.18, lw=0)
        except np.linalg.LinAlgError:
            pass
    ax.scatter(arr, np.full(arr.size, y0) + jit, s=70, color=color,
               edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_role_mix(mean_mix, std_mix, n, model, out_path) -> Path:
    """HERO: stacked overall mean default role mix + per-role spread bars."""
    fig, (ax_stack, ax_spread) = plt.subplots(1, 2, figsize=(11, 5.2),
                                              layout="constrained")
    colors = [_ROLE_COLORS[r] for r in _ROLES]

    bottom = 0.0
    for role, frac, c in zip(_ROLES, mean_mix, colors):
        ax_stack.bar(0, frac, bottom=bottom, color=c, edgecolor="white",
                     width=0.9, label=f"{role}: {frac:.2f}")
        if frac > 0.04:
            ax_stack.text(0, bottom + frac / 2, f"{role}\n{frac:.0%}", ha="center",
                          va="center", color="white", fontsize=10, fontweight="bold")
        bottom += frac
    ax_stack.set_xlim(-0.75, 0.75)
    ax_stack.set_xticks([])
    ax_stack.set_ylim(0, 1.0)
    ax_stack.set_ylabel("mean fraction of thinking draws")
    ax_stack.set_title("overall mean system-default mix", fontsize=12, fontweight="bold")
    ax_stack.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9,
                    title="role: mean", title_fontsize=9, frameon=True)

    hi = max(0.05, max(std_mix) * 1.35) if std_mix else 0.05
    bars = ax_spread.bar(range(len(_ROLES)), std_mix, color=colors,
                         edgecolor="white", width=0.7)
    for b, v in zip(bars, std_mix):
        ax_spread.text(b.get_x() + b.get_width() / 2, v + hi * 0.02, f"{v:.3f}",
                       ha="center", va="bottom", fontsize=10)
    ax_spread.set_xticks(range(len(_ROLES)))
    ax_spread.set_xticklabels(_ROLES)
    ax_spread.set_ylim(0, hi)
    ax_spread.set_ylabel("std across items of per-item mean fraction")
    ax_spread.set_title("spread of the default across items", fontsize=12, fontweight="bold")

    fig.suptitle(f"SESGO: thinking system-default role distribution  ({model}, n={n} items)",
                 fontsize=14, fontweight="bold", y=1.06)
    fig.text(0.5, 1.005, _SUBTITLE, ha="center", fontsize=9.5, color="#555555", style="italic")
    return _save(fig, out_path)


def plot_default_per_item(samples, model, out_path) -> Path:
    """(a) Per-item UNKNOWN-fraction strip + density: how the default varies."""
    unk = sorted(float(s.thinking.mean[2]) for s in samples)
    n = len(unk)
    fig, ax = _new_fig(8.6, 4.6)
    _jittered_strip(ax, unk, y0=0.0, color=_ROLE_COLORS["unknown"])
    if unk:
        m = float(np.mean(unk))
        ax.axvline(m, color=_ACCENT, ls="--", lw=1.6, label=f"mean = {m:.2f}")
        share = float(np.mean(np.asarray(unk) >= 0.5))
        ax.axvline(1.0, color=_REF, ls=":", lw=1.4,
                   label=f"safe default = 1.0\n(items ≥½ unknown: {share:.0%})")
        ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("per-item UNKNOWN fraction of the system default (mean over draws)")
    _titles(ax, f"SESGO: does the default abstain?  ({model}, n={n} items)",
            "per-item UNKNOWN share of the system default (1.0 = safe default)")
    return _save(fig, out_path)


def plot_default_uncertainty(ents, model, out_path) -> Path:
    """Per-item Shannon entropy of the default mix (default-uncertainty)."""
    n = len(ents)
    fig, ax = _new_fig(8.6, 4.6)
    _jittered_strip(ax, ents, y0=0.0, color="#117733")
    if ents:
        m = float(np.mean(ents))
        ax.axvline(m, color=_ACCENT, ls="--", lw=1.6, label=f"mean = {m:.3f} nats")
    ax.axvline(_LN3, color=_REF, ls=":", lw=1.4, label=f"max = ln 3 = {_LN3:.3f}")
    ax.set_xlim(-0.04, _LN3 + 0.06)
    ax.set_ylim(-0.2, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("per-item entropy (nats) of the default [target, other, unknown] mix")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    _titles(ax, f"SESGO: default-uncertainty per item  ({model}, n={n} items)",
            "Shannon entropy of the system default (0 = decisive, ln 3 = max uncertainty)")
    return _save(fig, out_path)


def plot_default_deviation(samples, model, out_path) -> Path:
    """(b) Per-item JS-divergence from the safe default, colored by pulling role."""
    rows = sorted(((_item_deviation(s), s) for s in samples), key=lambda r: r[0])
    n = len(rows)
    fig, ax = _new_fig(8.6, max(4.0, 0.32 * n + 1.6))
    for i, (dev, s) in enumerate(rows):
        # The role (besides unknown) carrying the most default mass drives deviation.
        t, o, _ = s.thinking.mean
        role = "target" if t >= o and t > 0 else ("other" if o > 0 else "unknown")
        c = _ROLE_COLORS[role]
        ax.plot([0, dev], [i, i], color=c, lw=2.0, alpha=0.55, zorder=1)
        ax.scatter([dev], [i], s=70, color=c, edgecolor="white", lw=0.8, zorder=3)
    if rows:
        m = float(np.mean([d for d, _ in rows]))
        ax.axvline(m, color=_ACCENT, ls="--", lw=1.6, label=f"mean = {m:.3f}")
    ax.axvline(0.0, color=_REF, ls="-", lw=1.2, alpha=0.6)
    ax.axvline(_LN2, color=_REF, ls=":", lw=1.4, label=f"max = ln 2 = {_LN2:.3f}")
    ax.set_xlim(-0.02, _LN2 + 0.04)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.set_ylabel("items (sorted by deviation)")
    ax.set_xlabel("JS-divergence of the system default from the safe default [0,0,1]")
    keys = [("at safe default", "unknown"), ("pulled toward other", "other"),
            ("pulled toward target", "target")]
    handles = [plt.Line2D([], [], marker="o", ls="", color=_ROLE_COLORS[r], mec="white",
                          label=lab) for lab, r in keys]
    leg1 = ax.legend(handles=handles, loc="center right", fontsize=8.5, frameon=True,
                     title="per-item color", title_fontsize=8.5)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.add_artist(leg1)
    _titles(ax, f"SESGO: default-deviation from safe abstention  ({model}, n={n} items)",
            "JS-divergence from the safe default UNKNOWN [0,0,1]; 0 = correct abstention")
    return _save(fig, out_path)


def plot_dispersion(std_by_role, model, out_path) -> Path:
    """Per-item across-draw std (instability of the default), per role as strips."""
    n = max((len(v) for v in std_by_role.values()), default=0)
    fig, ax = _new_fig(8.6, 5.0)
    for i, role in enumerate(_ROLES):
        vals = std_by_role.get(role, [])
        if not vals:
            continue
        _jittered_strip(ax, vals, y0=float(i), color=_ROLE_COLORS[role])
        m = float(np.mean(vals))
        ax.scatter([m], [i + 0.30], marker="D", s=55, color=_ACCENT, zorder=4,
                   edgecolor="white", lw=0.8)
        ax.text(m + 0.012, i + 0.30, f"mean = {m:.3f}", va="center", fontsize=9,
                color="#333333", fontweight="bold")
    ax.set_yticks(range(len(_ROLES)))
    ax.set_yticklabels(_ROLES)
    ax.set_ylim(-0.5, len(_ROLES) - 0.2)
    ax.set_xlim(-0.02, 0.55)
    ax.set_xlabel("per-item std of the role fraction across the N thinking draws")
    _titles(ax, f"SESGO: instability of the default per role  ({model}, n={n} items)",
            "across-draw dispersion of the system default (0 = identical every draw)")
    return _save(fig, out_path)


def plot_breakdown(metric, axis, groups, model, out_path) -> Path:
    """Per-group mean of a metric over one provenance axis, with per-item dots + n."""
    keys = list(groups.keys())
    means = [float(np.mean(groups[k])) for k in keys]
    ns = [len(groups[k]) for k in keys]
    hi = max(0.05, max(means + [max(g) for g in groups.values()]) * 1.25) if means else 0.05
    fig, ax = _new_fig(max(6.5, 1.5 * len(keys) + 2.5), 5.0)
    palette = sns.color_palette("colorblind", max(1, len(keys)))
    for i, (k, c) in enumerate(zip(keys, palette)):
        ax.bar(i, means[i], color=c, edgecolor="white", width=0.66, zorder=1)
        ax.text(i, means[i] + hi * 0.02, f"{means[i]:.3f}", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
        rng = np.random.default_rng(i)
        jx = i + (rng.random(ns[i]) - 0.5) * 0.28
        ax.scatter(jx, groups[k], s=34, color="#222222", alpha=0.55,
                   edgecolor="white", lw=0.5, zorder=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k}\n(n={m})" for k, m in zip(keys, ns)], fontsize=10)
    ax.set_ylim(0, hi)
    ax.set_ylabel(f"mean {metric}  (dots = per-item)")
    _titles(ax, f"SESGO: {metric} by {axis}  ({model})",
            "bars = group mean; dots = individual items (small-n groups annotated)")
    return _save(fig, out_path)


def _fmt(v: float | None) -> str:
    """Render a metric as a float, or n/a when undefined."""
    return f"{v:.3f}" if v is not None else "n/a"


def _log_stats(scored, ents, dev, mean_mix, std_mix, std_by_role) -> None:
    """Emit the full divergence stats table to the log."""
    log_section("SYSTEM-DEFAULT STATS (thinking, no scaffold)")
    log(f"  scored items (sample_size>0):                {len(scored)}")
    log(f"  default-uncertainty (mean entropy, nats):    {_fmt(_mean(ents))}  (max ln3={_LN3:.3f})")
    log(f"  mean default role mix [t,o,u]:               "
        f"[{mean_mix[0]:.3f}, {mean_mix[1]:.3f}, {mean_mix[2]:.3f}]")
    log(f"    spread across items (std):                 "
        f"[{std_mix[0]:.3f}, {std_mix[1]:.3f}, {std_mix[2]:.3f}]")
    log("  mean across-draw instability (per role):")
    for r in _ROLES:
        log(f"    {r:<8}: {_fmt(_mean(std_by_role[r]))}")
    log(f"  default-deviation (mean JS from UNKNOWN):    {_fmt(_mean(dev))}  (max ln2={_LN2:.3f})")
    log("  breakdowns (mean per group):")
    for axis in _BREAKDOWN_AXES:
        eg, dg = _group_means(scored, axis, _item_entropy), _group_means(scored, axis, _item_deviation)
        log(f"    by {axis}:")
        for k in eg:
            log(f"      {k:<14} (n={len(eg[k]):>2}) uncertainty={np.mean(eg[k]):.3f}  deviation={np.mean(dg[k]):.3f}")
    log("  NOTE: ambiguous gold is always UNKNOWN; deviation = distance of the system")
    log("        default from correct abstention (safe default = one-hot UNKNOWN).")


def main() -> None:
    """Load the SesgoDataset, compute every default statistic, plot, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO SYSTEM DEFAULT")

    dataset = SesgoDataset.from_json(args.samples)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")
    scored = _scored_samples(dataset)
    log(f"[viz] {len(scored)} item(s) with >=1 parsed thinking draw "
        f"(excluded {len(dataset.samples) - len(scored)} with sample_size==0)")

    ents = [_item_entropy(s) for s in scored]
    dev = [_item_deviation(s) for s in scored]
    if scored:
        mix = np.array([s.thinking.mean for s in scored], dtype=float)
        mean_mix, std_mix = mix.mean(axis=0).tolist(), mix.std(axis=0).tolist()
    else:
        mean_mix, std_mix = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    std_by_role: dict[str, list[float]] = {r: [] for r in _ROLES}
    for s in scored:
        for r, v in zip(_ROLES, s.thinking.std):
            std_by_role[r].append(float(v))

    _log_stats(scored, ents, dev, mean_mix, std_mix, std_by_role)

    plots_dir = args.out_dir / "sesgo" / "divergence" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    m = dataset.model_name

    written = [
        plot_role_mix(mean_mix, std_mix, len(scored), m, plots_dir / "role_mix.png"),
        plot_default_per_item(scored, m, plots_dir / "default_per_item.png"),
        plot_default_uncertainty(ents, m, plots_dir / "default_uncertainty.png"),
        plot_default_deviation(scored, m, plots_dir / "default_deviation.png"),
        plot_dispersion(std_by_role, m, plots_dir / "default_dispersion.png"),
    ]
    for axis in _BREAKDOWN_AXES:
        written.append(plot_breakdown("default-uncertainty", axis,
                       _group_means(scored, axis, _item_entropy), m,
                       plots_dir / f"uncertainty_by_{axis}.png"))
        written.append(plot_breakdown("default-deviation", axis,
                       _group_means(scored, axis, _item_deviation), m,
                       plots_dir / f"deviation_by_{axis}.png"))

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
