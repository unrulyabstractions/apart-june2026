"""Central-tendency panels: the HERO role mix and the abstention strip.

Two plots about WHERE the AMBIGUOUS system default sits on average: the overall
mean [target, other, unknown] mix (the HERO), and the per-item UNKNOWN-fraction
strip stacked by context condition (does the default abstain when it should, and
NOT abstain when it shouldn't?). Both carry bootstrap 95% CIs and annotate n.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.common.math import bootstrap_ci
from .divergence_plot_styles import (
    ACCENT,
    REF,
    ROLE_COLORS,
    ROLES,
    boot_band,
    jittered_strip,
    role_tick,
    save_fig,
    titled,
)


def plot_role_mix(mean_mix, n, samples, model, out_path):
    """HERO: overall mean default role mix with per-role bootstrap CIs + n."""
    mix = np.array([s.thinking.mean for s in samples], dtype=float) if samples \
        else np.zeros((0, 3))
    fig, ax = plt.subplots(figsize=(8.6, 5.2), layout="constrained")
    xs = np.arange(len(ROLES))
    for i, role in enumerate(ROLES):
        col = mix[:, i].tolist() if mix.size else []
        m = mean_mix[i]
        _, lo, hi = bootstrap_ci(col)
        lo = m if np.isnan(lo) else lo
        hi = m if np.isnan(hi) else hi
        ax.bar(i, m, color=ROLE_COLORS[role], edgecolor="white", width=0.66, zorder=2)
        ax.errorbar(i, m, yerr=[[max(0, m - lo)], [max(0, hi - m)]], fmt="none",
                    ecolor="#222222", elinewidth=1.7, capsize=6, capthick=1.7, zorder=4)
        ax.text(i, m + max(0, hi - m) + 0.02, f"{m:.0%}", ha="center",
                va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([role_tick(r) for r in ROLES], fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Share of the model's reasoning tries\nlanding on this answer")
    titled(ax,
           f"On ambiguous questions, what does {model} usually answer?  "
           f"(n={n} questions)",
           "Bar height = how often each answer was chosen across reasoning tries; "
           "tall 'Abstains' bar is the desired behaviour. Whiskers = 95% range.")
    return save_fig(fig, out_path)


def _unk_strip(ax, samples, title: str) -> None:
    """One UNKNOWN-fraction strip + density + bootstrap-CI mean for a subset."""
    unk = sorted(float(s.thinking.mean[2]) for s in samples)
    n = len(unk)
    jittered_strip(ax, unk, y0=0.0, color=ROLE_COLORS["unknown"])
    if unk:
        m, _ = boot_band(ax, unk, vertical=True)
        ax.axvline(m, color=ACCENT, ls="--", lw=1.6,
                   label=f"average = {m:.0%} (95% range shaded)")
        share = float(np.mean(np.asarray(unk) >= 0.5))
        ax.axvline(1.0, color=REF, ls=":", lw=1.4,
                   label=f"always abstains = 100%\n(questions abstained on > half "
                         f"the time: {share:.0%})")
        ax.legend(loc="center left", fontsize=8, frameon=True)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 0.55)
    ax.set_yticks([])
    ax.set_title(f"{title}  (n={n} questions)", fontsize=11, fontweight="bold", loc="left")


def plot_default_per_item(by_cond, model, out_path):
    """Stacked abstention-rate strips: ambiguous (top) vs clear (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), layout="constrained",
                             sharex=True)
    _unk_strip(axes[0], by_cond.get("ambig", []),
               "Ambiguous questions — abstaining is correct, so right = 100%")
    _unk_strip(axes[1], by_cond.get("disambig", []),
               "Clear questions — the answer is stated, so abstaining is wrong")
    axes[1].set_xlabel("How often the model abstained ('unknown') on each question, "
                       "across its reasoning tries")
    fig.suptitle(f"Does {model} abstain when (and only when) it should?",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.text(0.5, 1.0,
             "Each dot is one question. Right = abstains; left = commits to a group. "
             "Want dots far right on top, far left on bottom.",
             ha="center", fontsize=8.5, color=REF, style="italic")
    return save_fig(fig, out_path)
