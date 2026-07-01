"""Render the SESGO bias-alignment-vs-accuracy figure (arXiv:2509.03329), broken out by family.

Layout: one ROW per model family (Gemma / Qwen / Mistral / Llama) x two COLUMNS
(Ambiguous | Disambiguated). In every cell: y = accuracy, x = bias alignment
= F(Target) - F(Other) (0 = unbiased). Each model is a horizontal SEGMENT spanning its
neutral-wording to negative-wording alignment at its accuracy, so a model that spreads its
errors across the two named groups (a wide, off-centre span) is visible at a glance, with a
filled dot at each wording endpoint, a diamond at the pooled bias, a faint vertical Wilson
95% CI on accuracy, and a short size tag. Circle = direct, triangle = thinking / reasoning.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from experiment.bias.bias_model_style import ModelStyle  # noqa: E402
from experiment.bias.bias_segments import BiasSegment  # noqa: E402
from experiment.bias.segment_label_layout import spread_labels  # noqa: E402

_PANELS = (("ambig", "Ambiguous  (correct $=$ abstain)"),
           ("disambig", "Disambiguated  (correct $=$ right group)"))
_LABEL_GAP = 0.07


def _thinking_tint(hex_color: str, frac: float = 0.5) -> str:
    """A lighter tint of the family colour, used so thinking/reasoning slices read as a
    distinct (paler) shade in addition to the triangle marker and the ``T`` label tag."""
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac))


def _model_color(st: ModelStyle) -> str:
    return _thinking_tint(st.color) if st.is_thinking else st.color


def _xlim(segments: list[BiasSegment]) -> float:
    m = max((max(abs(s.align_neutral), abs(s.align_negative), abs(s.align_pooled))
             for s in segments), default=0.5)
    return min(max(m * 1.15, 0.4), 1.0)


def _draw_model(ax, seg: BiasSegment, st: ModelStyle) -> None:
    y = seg.accuracy
    _, ci_lo, ci_hi = seg.wilson
    lo, hi = seg.span
    c = _model_color(st)
    marker, ls = ("^", "--") if st.is_thinking else ("o", "-")
    ax.plot([seg.align_pooled, seg.align_pooled], [ci_lo, ci_hi], color=c, lw=1.0,
            alpha=0.25, zorder=2, solid_capstyle="round")
    ax.plot([lo, hi], [y, y], color=c, lw=3.0, ls=ls, alpha=0.9, zorder=4, solid_capstyle="round")
    ax.plot([lo, hi], [y, y], marker, color=c, ms=6.0, zorder=5)
    ax.plot([seg.align_pooled], [y], "D", color=c, mec="white", mew=0.7, ms=6.0, zorder=6)


def _draw_labels(ax, segs: list[BiasSegment], styles: dict, xlim: float) -> None:
    """Size tag at the right edge, de-collided in y. Thinking slices get a paler colour and an
    explicit ``T`` tag so the direct/thinking split is unambiguous in both colour and label."""
    ranked = sorted(segs, key=lambda s: s.accuracy)
    slots = spread_labels([s.accuracy for s in ranked], _LABEL_GAP, hi=1.0)
    for slot in slots:
        seg = ranked[slot.index]; st = styles[seg.group_key]
        c = _model_color(st); right = max(seg.span)
        ax.plot([right, xlim - 0.02], [seg.accuracy, slot.y_label], color=c, lw=0.5,
                alpha=0.4, zorder=3)
        ax.text(xlim - 0.02, slot.y_label, st.size_label + (r"$\,$T" if st.is_thinking else ""),
                color=c, fontsize=7.5, va="center", ha="left", fontweight="bold", zorder=6)


def _style_cell(ax, xlim: float, is_top: bool, title: str, is_bottom: bool) -> None:
    ax.axvspan(-0.05, 0.05, color="#000000", alpha=0.05, zorder=0)
    ax.axvline(0.0, color="#333333", lw=0.9, zorder=1)
    ax.set_xlim(-xlim, xlim * 1.25); ax.set_ylim(-0.03, 1.06)
    ax.set_xticks([t for t in (-1, -0.5, 0, 0.5, 1) if abs(t) <= xlim])
    ax.grid(True, axis="y", ls=":", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if is_top:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if is_bottom:
        ax.set_xlabel("bias alignment   $\\leftarrow$ Other   $\\cdot$   Target $\\rightarrow$",
                      fontsize=9.5)


def plot_bias_alignment(segments: list[BiasSegment], styles: dict, family_order: list[str],
                        family_color: dict, suptitle: str, out_path) -> None:
    fams = [f for f in family_order if any(styles[s.group_key].family == f for s in segments)]
    xlim = _xlim(segments)
    fig, axes = plt.subplots(len(fams), 2, figsize=(11, 2.35 * len(fams) + 0.8),
                             sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.11, right=0.985, wspace=0.10, hspace=0.28, top=0.90, bottom=0.07)
    for r, fam in enumerate(fams):
        for c, (panel, title) in enumerate(_PANELS):
            ax = axes[r][c]
            cell = [s for s in segments
                    if s.panel == panel and styles[s.group_key].family == fam]
            for s in cell:
                _draw_model(ax, s, styles[s.group_key])
            if cell:
                _draw_labels(ax, cell, styles, xlim)
            _style_cell(ax, xlim, is_top=(r == 0), title=title, is_bottom=(r == len(fams) - 1))
        axes[r][0].set_ylabel(fam, fontsize=12, fontweight="bold", color=family_color.get(fam, "#333"))
    handles = [Line2D([], [], color="#444", lw=3, ls="-", marker="o", ms=6,
                      label="direct  (circle, solid, full colour)"),
               Line2D([], [], color="#9c9c9c", lw=3, ls="--", marker="^", ms=7,
                      label="thinking / reasoning  (triangle, dashed, pale, +T)"),
               Line2D([], [], color="#444", lw=0, marker="D", ms=6, mec="white", label="pooled bias")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8.6, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(suptitle, fontsize=13.5, fontweight="bold", y=0.995)
    fig.text(0.028, 0.5, "Accuracy  (per family row)", va="center", ha="center",
             rotation=90, fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
