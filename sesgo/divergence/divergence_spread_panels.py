"""The spread suite: per-item entropy, JS-deviation lollipop, per-role dispersion.

These three plots quantify how the AMBIGUOUS system default spreads — how
UNCERTAIN it is (Shannon entropy), how far it DEVIATES from correct abstention
(JS-divergence from the safe default), and how UNSTABLE it is across draws (per-
role std). Each mean carries a bootstrap 95% CI band/whisker; n on every panel.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.common.math import bootstrap_ci
from .divergence_item_metrics import item_deviation
from .divergence_plot_styles import (
    ACCENT,
    LN2,
    LN3,
    REF,
    ROLE_COLORS,
    ROLES,
    boot_band,
    jittered_strip,
    role_tick,
    save_fig,
    titled,
)


def plot_default_uncertainty(ents, model, out_path):
    """Per-question indecision (entropy) of the answer across reasoning tries."""
    n = len(ents)
    fig, ax = plt.subplots(figsize=(8.6, 4.6), layout="constrained")
    jittered_strip(ax, ents, y0=0.0, color="#117733")
    if ents:
        m, _ = boot_band(ax, ents, vertical=True)
        ax.axvline(m, color=ACCENT, ls="--", lw=1.6,
                   label=f"average = {m:.2f} (95% range shaded)")
    ax.axvline(LN3, color=REF, ls=":", lw=1.4, label="most indecisive possible")
    ax.set_xlim(-0.04, LN3 + 0.06)
    ax.set_ylim(-0.2, 0.55)
    ax.set_yticks([0.0, LN3])
    ax.set_yticklabels([])
    ax.set_xticks([0.0, LN3])
    ax.set_xticklabels(["0\nalways same answer", f"{LN3:.2f}\nsplit evenly 3 ways"])
    ax.set_xlabel("How much the answer wavered across the model's reasoning tries")
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    titled(ax,
           f"How indecisive is {model} on ambiguous questions?  (n={n} questions)",
           "Each dot is one question. Left = same answer every try; "
           "right = answer flips around. Lower is steadier.")
    return save_fig(fig, out_path)


def plot_default_deviation(samples, model, out_path):
    """Per-question drift away from abstaining, colored by which group pulls it."""
    rows = sorted(((item_deviation(s), s) for s in samples), key=lambda r: r[0])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.6, max(4.0, 0.32 * n + 1.6)),
                           layout="constrained")
    for i, (dev, s) in enumerate(rows):
        t, o, _ = s.thinking.mean
        role = "target" if t >= o and t > 0 else ("other" if o > 0 else "unknown")
        c = ROLE_COLORS[role]
        ax.plot([0, dev], [i, i], color=c, lw=2.0, alpha=0.55, zorder=1)
        ax.scatter([dev], [i], s=66, color=c, edgecolor="white", lw=0.8, zorder=3)
    if rows:
        m, _ = boot_band(ax, [d for d, _ in rows], vertical=True)
        ax.axvline(m, color=ACCENT, ls="--", lw=1.6,
                   label=f"average = {m:.2f} (95% range shaded)")
    ax.axvline(0.0, color=REF, ls="-", lw=1.2, alpha=0.6)
    ax.axvline(LN2, color=REF, ls=":", lw=1.4, label="farthest possible")
    ax.set_xlim(-0.02, LN2 + 0.04)
    # Headroom above the top (largest) lollipop so the legends never sit on a dot.
    ax.set_ylim(-0.7, max(n - 0.3, 0.7) + max(2.5, 0.18 * n))
    ax.set_yticks([])
    ax.set_ylabel("Each line is one question")
    ax.set_xticks([0.0, LN2])
    ax.set_xticklabels(["0\nabstains correctly", f"{LN2:.2f}\nfully commits to a group"])
    ax.set_xlabel("How far the model drifted from correctly abstaining")
    keys = [("Correctly abstains", "unknown"),
            ("Leans toward the other group", "other"),
            ("Leans toward the stereotyped group", "target")]
    handles = [plt.Line2D([], [], marker="o", ls="", color=ROLE_COLORS[r], mec="white",
                          label=lab) for lab, r in keys]
    leg1 = ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=True,
                     title="Where each question landed", title_fontsize=8.5)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax.add_artist(leg1)
    titled(ax,
           f"How far does {model} stray from abstaining on ambiguous questions?  "
           f"(n={n} questions)",
           "Each line is one question; longer = drifts further from the correct "
           "'abstain'. Colour shows which group it leans toward.")
    return save_fig(fig, out_path)


def plot_dispersion(std_by_role, model, out_path):
    """Per-question wobble in each answer's frequency across reasoning tries."""
    n = max((len(v) for v in std_by_role.values()), default=0)
    fig, ax = plt.subplots(figsize=(8.6, 5.0), layout="constrained")
    for i, role in enumerate(ROLES):
        vals = std_by_role.get(role, [])
        if not vals:
            continue
        jittered_strip(ax, vals, y0=float(i), color=ROLE_COLORS[role])
        m, lo, hi = bootstrap_ci(list(vals))
        ax.errorbar(m, i + 0.30, xerr=[[max(0, m - lo)], [max(0, hi - m)]], fmt="D",
                    ms=7, color=ACCENT, ecolor=ACCENT, elinewidth=1.5, capsize=4,
                    mec="white", zorder=4)
        ax.text(hi + 0.012, i + 0.30, f"average = {m:.2f}", va="center", fontsize=9,
                color="#333333", fontweight="bold")
    ax.set_yticks(range(len(ROLES)))
    ax.set_yticklabels([role_tick(r) for r in ROLES])
    ax.set_ylim(-0.5, len(ROLES) - 0.2)
    ax.set_xlim(-0.02, 0.62)
    ax.set_xlabel("How much each answer's frequency wobbled "
                  "from one reasoning try to the next")
    titled(ax,
           f"How stable is each answer across {model}'s reasoning tries?  "
           f"(n={n} questions)",
           "Each dot is one question. Left = the model gave that answer just as "
           "often every try; right = it came and went.")
    return save_fig(fig, out_path)
