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
    save_fig,
    titled,
)


def plot_default_uncertainty(ents, model, out_path):
    """Per-item Shannon entropy of the default mix (default-uncertainty)."""
    n = len(ents)
    fig, ax = plt.subplots(figsize=(8.6, 4.6), layout="constrained")
    jittered_strip(ax, ents, y0=0.0, color="#117733")
    if ents:
        m, _ = boot_band(ax, ents, vertical=True)
        ax.axvline(m, color=ACCENT, ls="--", lw=1.6,
                   label=f"mean = {m:.3f} nats (boot 95% CI shaded)")
    ax.axvline(LN3, color=REF, ls=":", lw=1.4, label=f"max = ln 3 = {LN3:.3f}")
    ax.set_xlim(-0.04, LN3 + 0.06)
    ax.set_ylim(-0.2, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("per-item entropy (nats) of the default [target, other, unknown] mix")
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    titled(ax, f"SESGO: default-uncertainty per item  ({model}, n={n} items)",
           "Shannon entropy of the system default (0 = decisive, ln 3 = max uncertainty)")
    return save_fig(fig, out_path)


def plot_default_deviation(samples, model, out_path):
    """Per-item JS-divergence from the safe default, colored by pulling role."""
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
                   label=f"mean = {m:.3f} (boot 95% CI shaded)")
    ax.axvline(0.0, color=REF, ls="-", lw=1.2, alpha=0.6)
    ax.axvline(LN2, color=REF, ls=":", lw=1.4, label=f"max = ln 2 = {LN2:.3f}")
    ax.set_xlim(-0.02, LN2 + 0.04)
    # Headroom above the top (largest) lollipop so the legends never sit on a dot.
    ax.set_ylim(-0.7, max(n - 0.3, 0.7) + max(2.5, 0.18 * n))
    ax.set_yticks([])
    ax.set_ylabel("items (sorted by deviation)")
    ax.set_xlabel("JS-divergence of the system default from the safe default [0,0,1]")
    keys = [("at safe default", "unknown"), ("pulled toward other", "other"),
            ("pulled toward target", "target")]
    handles = [plt.Line2D([], [], marker="o", ls="", color=ROLE_COLORS[r], mec="white",
                          label=lab) for lab, r in keys]
    leg1 = ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=True,
                     title="per-item color", title_fontsize=8.5)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax.add_artist(leg1)
    titled(ax, f"SESGO: default-deviation from safe abstention  ({model}, n={n} items)",
           "JS-divergence from the safe default UNKNOWN [0,0,1]; 0 = correct abstention")
    return save_fig(fig, out_path)


def plot_dispersion(std_by_role, model, out_path):
    """Per-item across-draw std (instability of the default), per role as strips."""
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
        ax.text(hi + 0.012, i + 0.30, f"mean = {m:.3f}", va="center", fontsize=9,
                color="#333333", fontweight="bold")
    ax.set_yticks(range(len(ROLES)))
    ax.set_yticklabels(ROLES)
    ax.set_ylim(-0.5, len(ROLES) - 0.2)
    ax.set_xlim(-0.02, 0.62)
    ax.set_xlabel("per-item std of the role fraction across the N thinking draws")
    titled(ax, f"SESGO: instability of the default per role  ({model}, n={n} items)",
           "across-draw dispersion of the system default (0 = identical every draw)")
    return save_fig(fig, out_path)
