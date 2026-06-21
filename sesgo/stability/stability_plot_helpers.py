"""Publication-quality stability plots: stacked readouts/conditions with CIs + n.

Every figure is ONE file. Comparisons that the study cares about are stacked as
SUBPLOTS so they read together: 2-OPTION vs 3-OPTION consistency stack vertically;
ambiguous vs disambiguated are drawn side-by-side within each panel. Every bar /
mean carries an honest uncertainty band (Wilson for proportions, SEM for means,
bootstrap for the p(unknown) spread) and its sample size is annotated on or beside
it, never hidden.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from src.common.math import bootstrap_ci, sem, wilson_err, wilson_interval
from stability_metrics_helpers import AccuracyCount, ConsistencySet, FlipRate

# Colorblind-safe (Okabe-Ito): one hue per CONTEXT CONDITION, reused everywhere.
_AMBIG = "#0072B2"      # ambiguous (gold = UNKNOWN / abstain)
_DISAMBIG = "#E69F00"   # disambiguated (gold = ground-truth role)
_MEAN_RED = "#D55E00"   # mean reference line
_ZONE_GREY = "#BBBBBB"  # de-emphasized "no robust items" / empty zone
_COND_COLOR = {"ambig": _AMBIG, "disambig": _DISAMBIG}
_COND_LABEL = {"ambig": "ambiguous (gold=UNKNOWN)", "disambig": "disambiguated (gold=role)"}


def _titled(ax, title: str, takeaway: str) -> None:
    """Bold title stacked over an italic plain-language takeaway, anchored to axes."""
    ax.set_title(f"{title}\n", fontsize=12, fontweight="bold", pad=22)
    ax.text(0.5, 1.012, takeaway, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color="#444444", style="italic")


def _save(fig, out_path: Path) -> Path:
    """Persist a figure publication-clean: tight bbox, 150 dpi, then close it."""
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _consistency_panel(ax, sets: dict[str, ConsistencySet], readout: str) -> None:
    """One stacked panel: ambig vs disambig consistency histograms + mean+-SEM.

    The mean of each condition is drawn as a dashed line carrying a short
    horizontal SEM whisker (a capped error bar near the top of the panel), so the
    mean's own uncertainty is visible without a full-height band swamping the bars.
    """
    bins = np.linspace(0, 1, 11)
    # Neutral marker for the high-consistency / "robust" region (>0.8). With both
    # conditions present some items DO land here, so it is shaded, not labelled empty.
    ax.axvspan(0.8, 1.0, color=_ZONE_GREY, alpha=0.14, zorder=0)
    max_h = 0.0
    for cond in ("ambig", "disambig"):
        cs = sets[cond]
        if not cs.consistency:
            continue
        counts, _, _ = ax.hist(
            cs.consistency, bins=bins, alpha=0.60, color=_COND_COLOR[cond],
            edgecolor="white", label=f"{_COND_LABEL[cond]}  (n={len(cs.consistency)})")
        max_h = max(max_h, counts.max())
    top = max(1.0, max_h) * 1.40
    ax.set_ylim(0, top)
    # Mean +- SEM as a capped errorbar marker, one row per condition near the top.
    # The text label is placed on the side with whitespace and offset OFF the marker
    # (with a white bbox) so it never overprints the dashed mean line or the dot.
    for j, cond in enumerate(("ambig", "disambig")):
        cs = sets[cond]
        if not cs.consistency:
            continue
        m, e = float(np.mean(cs.consistency)), sem(cs.consistency)
        y = top * (0.95 - 0.085 * j)
        ax.axvline(m, color=_COND_COLOR[cond], linestyle="--", linewidth=1.8, zorder=2)
        ax.errorbar(m, y, xerr=e, fmt="o", color=_COND_COLOR[cond], capsize=4,
                    markersize=5, elinewidth=1.6, zorder=3)
        # Label left of the marker if the mean is in the right half, else right.
        ha, dx = ("right", -e - 0.02) if m > 0.5 else ("left", e + 0.02)
        ax.annotate(f"mean {m:.0%} +-{e:.0%}", (m, y), xytext=(m + dx, y),
                    ha=ha, va="center", fontsize=8, color=_COND_COLOR[cond],
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.text(0.9, top * 0.40, "robust zone\n(>0.8)", ha="center", va="center",
            fontsize=8, color="#888888")
    ax.set_xlim(0, 1)
    ax.set_ylabel(f"{readout}\nitems", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)


def plot_consistency(
    three_opt: dict[str, ConsistencySet], two_opt: dict[str, ConsistencySet],
    model: str, out_path: Path,
) -> Path:
    """Stacked 3-OPTION (top) vs 2-OPTION (bottom) per-item consistency by condition."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.2), layout="constrained", sharex=True)
    _consistency_panel(axes[0], three_opt, "3-option")
    _consistency_panel(axes[1], two_opt, "2-option")
    _titled(axes[0], f"SESGO stability: per-item prediction consistency ({model})",
            "fraction of an item's 18 format variations that keep its modal answer")
    axes[1].set_xlabel("per-item consistency (modal-answer fraction across format variations)")
    return _save(fig, out_path)


def plot_format_sensitivity(
    flips_by_cond: dict[str, list[FlipRate]], model: str, out_path: Path,
) -> Path:
    """Grouped bars: mean flip rate per axis, ambig vs disambig, with bootstrap CIs."""
    axes_names = [f.axis for f in next(iter(flips_by_cond.values()))]
    fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
    width, x = 0.36, np.arange(len(axes_names))
    for i, cond in enumerate(("ambig", "disambig")):
        rates, errs, ns = [], [[], []], []
        for fr in flips_by_cond[cond]:
            _, lo, hi = bootstrap_ci(fr.flips) if fr.flips else (fr.rate, fr.rate, fr.rate)
            r = 0.0 if np.isnan(fr.rate) else fr.rate
            rates.append(r)
            errs[0].append(max(0.0, r - (lo if not np.isnan(lo) else r)))
            errs[1].append(max(0.0, (hi if not np.isnan(hi) else r) - r))
            ns.append(fr.n_groups)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, rates, width, yerr=errs, capsize=4,
                      color=_COND_COLOR[cond], label=f"{_COND_LABEL[cond]}",
                      error_kw={"elinewidth": 1.4, "ecolor": "#333333"})
        # Annotate above the CI upper cap so the label never overlaps the whisker.
        for bar, r, n, above in zip(bars, rates, ns, errs[1]):
            ax.text(bar.get_x() + bar.get_width() / 2, r + above + 0.012,
                    f"{r:.0%}\nn={n}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", " ") for a in axes_names], fontsize=11)
    ax.set_ylim(0, max(0.05, ax.get_ylim()[1] * 1.12))
    ax.set_ylabel("mean flip rate  (1 - within-bucket consistency)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    _titled(ax, f"SESGO stability: format sensitivity per axis ({model})",
            "how often perturbing one format axis alone flips the 3-option prediction")
    return _save(fig, out_path)


def _spread_panel(ax, spreads: list[float], cond: str) -> None:
    """One p(unknown)-spread panel: KDE + rug + mean with bootstrap-CI band."""
    if spreads:
        sns.kdeplot(x=spreads, ax=ax, color=_COND_COLOR[cond], fill=True, alpha=0.20,
                    linewidth=2, cut=0, bw_adjust=1.2)
        sns.rugplot(x=spreads, ax=ax, color=_COND_COLOR[cond], height=0.07,
                    linewidth=2, alpha=0.9)
        m, lo, hi = bootstrap_ci(spreads)
        ax.axvspan(lo, hi, color=_MEAN_RED, alpha=0.12, zorder=0)
        ax.axvline(m, color=_MEAN_RED, linestyle="--", linewidth=2,
                   label=f"mean={m:.3f}  [{lo:.3f}, {hi:.3f}]  (n={len(spreads)} items)")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
        # Headroom so the legend never collides with the KDE peak.
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="#999999")
        ax.set_ylim(bottom=0)
    ax.set_ylabel(f"{_COND_LABEL[cond]}\ndensity", fontsize=9.5)


def plot_p_unknown_spread(
    spreads_by_cond: dict[str, list[float]], model: str, out_path: Path,
) -> Path:
    """Stacked ambig (top) vs disambig (bottom) per-item p(unknown) spread + CI."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.6), layout="constrained", sharex=True)
    _spread_panel(axes[0], spreads_by_cond["ambig"], "ambig")
    _spread_panel(axes[1], spreads_by_cond["disambig"], "disambig")
    _titled(axes[0], f"SESGO stability: p(unknown) calibration spread per item ({model})",
            "std of non-thinking p(unknown) across an item's format variations (ticks=items)")
    axes[1].set_xlabel("per-item std of non-thinking p(unknown) across variations")
    return _save(fig, out_path)


def _is_ambig(label: str) -> bool:
    """Whether an accuracy label is the AMBIGUOUS condition (exact, not substring).

    "disambig" contains "ambig" as a substring, so a naive ``in`` test miscolours
    every bar; split the trailing condition token and compare it exactly.
    """
    return label.split()[-1] == "ambig"


def plot_accuracy(counts: list[AccuracyCount], model: str, out_path: Path) -> Path:
    """Per-condition accuracy bars (3-opt + 2-opt) with Wilson 95% CIs and n.

    Hue encodes the CONTEXT CONDITION (blue=ambiguous, orange=disambiguated);
    hatching encodes the READOUT (solid=3-option, hatched=2-option) so the two
    crossed factors read at a glance without a four-colour palette.
    """
    fig, ax = plt.subplots(figsize=(9, 5.4), layout="constrained")
    xs = list(range(len(counts)))
    for i, c in enumerate(counts):
        p, _, _ = wilson_interval(c.correct, c.total)
        below, above = wilson_err(c.correct, c.total)
        h = 0.0 if np.isnan(p) else p
        color = _AMBIG if _is_ambig(c.label) else _DISAMBIG
        hatch = "" if c.label.startswith("3-opt") else "///"
        ax.bar(i, h, yerr=[[below], [above]], capsize=5, width=0.6, color=color,
               hatch=hatch, edgecolor="white", linewidth=0,
               error_kw={"elinewidth": 1.5, "ecolor": "#333333"})
        # Anchor the annotation ABOVE the CI cap so text never sits on a whisker.
        y = (h + above + 0.03) if c.total else 0.04
        txt = "N/A\n(no UNKNOWN\noption to score)" if c.total == 0 else f"{h:.0%}\n{c.correct}/{c.total}"
        ax.text(i, y, txt, ha="center", va="bottom", fontsize=9)
    ax.axhline(1 / 3, color=_ZONE_GREY, linestyle=":", linewidth=1.4)
    ax.text(-0.45, 1 / 3 + 0.012, "chance = 1/3 (3-opt)", fontsize=8,
            color="#777777", ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([c.label.replace(" ", "\n") for c in counts], fontsize=10)
    # Custom legend keying both encodings (condition hue, readout hatch).
    handles = [
        Patch(facecolor=_AMBIG, label="ambiguous (gold=UNKNOWN)"),
        Patch(facecolor=_DISAMBIG, label="disambiguated (gold=role)"),
        Patch(facecolor="#FFFFFF", edgecolor="#333333", label="3-option (solid)"),
        Patch(facecolor="#FFFFFF", edgecolor="#333333", hatch="///", label="2-option (hatched)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.95, ncol=2)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy vs per-condition gold  (Wilson 95% CI)")
    _titled(ax, f"SESGO stability: per-condition accuracy ({model})",
            "ambiguous gold=abstain (UNKNOWN); disambiguated gold=the ground-truth role")
    return _save(fig, out_path)
