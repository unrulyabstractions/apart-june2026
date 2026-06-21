"""Plot SESGO geometry: behavioural readouts + representational geometry.

Run-by-path driver. Renders publication-quality PNGs into
out/sesgo/geometry/<MODEL>/plots/. Every plot carries uncertainty (Wilson score
CIs on proportion bars, bootstrap CIs on centroid shifts / silhouette) and the
sample size n is annotated on every bar, panel, and title.

BEHAVIOURAL (read directly from response_samples.json):
  accuracy_by_condition.png : per-condition (ambiguous vs disambiguated) accuracy
    with the 3-OPTION readout (top) and the 2-OPTION forced choice (bottom)
    STACKED in one figure. Gold is per-condition (ambiguous -> UNKNOWN abstention;
    disambiguated -> the ground-truth role); the 2-option has no abstention so its
    ambiguous cell is N/A.
  accuracy_by_readout.png   : non-thinking vs greedy-thinking vs sampled-thinking
    accuracy STACKED as subfigures, each split by context condition.

GEOMETRY (from analysis/projections.json; run analyze_geometry.py first):
  pca_scatter_<position>.png : PC1-PC2 scatter coloured by scaffold, centroids +
    baseline->interpretive shift arrow, at every structural position, rendered at
    the ADAPTIVE feature layer (the captured mid->last layer that maximises the
    scaffold silhouette at the answer-label position — where the structure is
    clearest). The title carries that layer + its model-relative depth (e.g.
    "layer 31/36 (0.86 depth)") so the chosen depth is comparable across sizes.
  pca_by_<axis>.png / pca_axes_grid.png : the SAME feature-layer answer projection
    recoloured by EVERY axis in the shared geometry_color_axes registry —
    CATEGORICAL axes (scaffold, origin, language, bias category, question polarity,
    context condition, accuracy, selected/gold role, readout, identities, gold,
    label style) get a discrete legend; CONTINUOUS answer-distribution signals
    (top-choice prob/logit, entropy, diversity, inverse perplexity) get a
    sequential colormap + colorbar.
  silhouette_by_layer_axis.png : (layer x axis) silhouette HEATMAP — at what depth
    each colour-by axis becomes separable (the per-layer differentiation, visible).
  silhouette_layer_sweep.png : silhouette-vs-layer line sweep for the KEY axes
    (accuracy, context_condition, selected_role, scaffold).
  centroid_shift_by_position.png : centroid-shift magnitude per (layer, position)
    with bootstrap 95% CIs.
  silhouette_separability.png : per-axis silhouette separability at the
    representative layer/position, with bootstrap CI on the scaffold axis.
  explained_variance.png : EV%(PC1-3) per structural position.

Usage:
  uv run python sesgo/geometry/analyze_geometry.py \
      out/sesgo/geometry/Qwen3-0.6B/response_samples.json   # writes projections.json
  uv run python sesgo/geometry/visualize_geometry_samples.py \
      out/sesgo/geometry/Qwen3-0.6B/response_samples.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import load_json  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import GeometryDataset  # noqa: E402

from sesgo.geometry.geometry_color_axes import (  # noqa: E402
    COLOR_AXES,
    KEY_SWEEP_AXIS_KEYS,
    SCAFFOLD_AXIS_KEY,
)
from sesgo.geometry.geometry_plain_labels import (  # noqa: E402
    axis_title,
    axis_value_label,
    position_label,
)
from sesgo.geometry.geometry_feature_layer import (  # noqa: E402
    FeatureLayer,
    select_feature_layer,
)
from sesgo.geometry.geometry_layer_plot_helpers import (  # noqa: E402
    draw_continuous_scatter,
    draw_layer_sweep,
    draw_silhouette_heatmap,
)
from sesgo.geometry.geometry_plot_helpers import (  # noqa: E402
    AXIS_PALETTE,
    BASELINE,
    PALETTE,
    apply_style,
    axis_caption,
    draw_axis_scatter,
    finish,
    plot_accuracy_by_condition,
    plot_accuracy_by_readout,
    robust_limits,
    wrapped,
)

_POSITIONS = (
    "im_end",
    "newline",
    "im_start",
    "assistant",
    "think_open",
    "think_close",
    "answer_prefix",
    "label",
)
_AXIS_PANEL_POSITION = "label"  # the projection the per-axis scatters recolour
# The colour-by axes come straight from the single shared registry (DRY): adding
# one ColorAxis there makes it appear in every per-axis panel + the grid here.
_AXES = COLOR_AXES


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for geometry visualization."""
    parser = argparse.ArgumentParser(
        description="Plot SESGO geometry (behavioural + representational geometry)"
    )
    parser.add_argument("samples", type=Path, help="response_samples.json (a GeometryDataset)")
    parser.add_argument("--out-dir", type=Path, default=Path("out"), help="Base output dir")
    return parser.parse_args()


def _colour(label: str, ordered: list[str]) -> str:
    """Stable scaffold colour; the baseline always takes the first palette entry."""
    if label == BASELINE:
        return PALETTE[0]
    rest = [l for l in ordered if l != BASELINE]
    return PALETTE[(rest.index(label) + 1) % len(PALETTE)]


def _layer_phrase(feat: FeatureLayer) -> str:
    """Plain "layer 14 of 28 (halfway through)" tag, no pipeline jargon."""
    if feat.total_layers <= 0:
        return "averaged over the prompt"
    return (f"layer {feat.layer_index} of {feat.total_layers} "
            f"({feat.relative_depth:.0%} of the way through the network)")


def _wrap_tick(text: str, width: int = 14) -> str:
    """Wrap a multi-word x-tick label across lines so dense axes stay readable."""
    return "\n".join(textwrap.wrap(text, width=width)) or text


def _separation_note(sil: float | None, lo=None, hi=None) -> str:
    """Compact, plain separation-score phrase (empty when no score is available)."""
    if sil is None:
        return ""
    note = f"Separation score {sil:.2f} (1 = fully apart, 0 = overlapping, below 0 = intermixed)"
    return note + (f" [{lo:.2f}, {hi:.2f}]" if lo is not None else "")


def _scatter_subtitle(ptype: str, feat: FeatureLayer, n: int, sep: str) -> str:
    """One wrapped 'how to read it' subtitle line shared by the PCA scatters."""
    base = f"Read at {position_label(ptype)}; {_layer_phrase(feat)}, n={n}"
    if sep:
        base += f". {sep}"
    return wrapped(base, 90)


def _ordered_scaffolds(labels: set[str]) -> list[str]:
    """Scaffold labels with the baseline first, the rest sorted after it."""
    rest = sorted(labels - {BASELINE})
    return ([BASELINE] if BASELINE in labels else []) + rest


def _resolve_feature_layer(results: dict) -> FeatureLayer:
    """Adaptively pick the scatter layer = where the scaffold structure is clearest.

    Delegates to the shared ``select_feature_layer`` helper (DRY), which reads the
    precomputed per-(layer, position) silhouettes at the answer-``_AXIS_PANEL_POSITION``
    and returns the captured layer maximising scaffold separability (deeper on a
    near-tie; deepest captured layer if separations are missing). When no integer
    layer exists at all, degrades to the "mean" cloud (then the first key) so the
    scatters still render with a sensible (non-depth) title.
    """
    feat = select_feature_layer(results, _AXIS_PANEL_POSITION)
    if feat is not None:
        return feat
    fallback_key = "mean" if "mean" in results else next(iter(results))
    return FeatureLayer(fallback_key, -1, 0, 0.0, None, by_silhouette=False)


# ── PCA scatter (coloured by scaffold, with shift arrow) ──────────────────────


def _draw_centroid_shift(ax, stats: dict, ordered: list[str]) -> None:
    """Plot each centroid as a diamond and draw the baseline->intervention arrow."""
    cents = stats["centroids"]
    for lab in ordered:
        if lab not in cents:
            continue
        cx, cy = cents[lab]["coord2d"]
        ax.scatter(cx, cy, s=320, marker="D", facecolor=_colour(lab, ordered),
                   edgecolor="black", linewidth=1.6, zorder=6)
    base = cents.get(BASELINE)
    for lab, sh in stats.get("shifts", {}).items():
        if base is None or lab not in cents:
            continue
        bx, by = base["coord2d"]
        tx, ty = cents[lab]["coord2d"]
        ax.annotate("", xy=(tx, ty), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#333333",
                                    shrinkA=10, shrinkB=10), zorder=5)
        ci = ""
        if "shift_ci_low" in sh:
            ci = f"\n[{sh['shift_ci_low']:.1f}, {sh['shift_ci_high']:.1f}]"
        # Anchor the shift label in the empty bottom-centre gap between the two
        # scaffold clusters (they separate along PC1), with a thin connector to the
        # arrow midpoint, so the box never occludes a centroid or its point cloud.
        mx, my = (bx + tx) / 2, (by + ty) / 2
        ax.annotate(f"how far the scaffold\nmoved the state = {sh['shift_magnitude']:.1f}{ci}",
                    xy=(mx, my), xytext=(0.5, 0.03), textcoords="axes fraction",
                    fontsize=8.5, fontweight="bold", color="#333333", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.92),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#999999", alpha=0.6), zorder=7)


def plot_pca_scatter(block: dict, ptype: str, feat: FeatureLayer, model: str, out_path: Path) -> Path:
    """PC1-PC2 scatter coloured by scaffold, with centroids + shift arrow."""
    coords = np.asarray([s["coord2d"] for s in block["samples"]], dtype=float)
    labels = [s["scaffold_id"] or BASELINE for s in block["samples"]]
    ordered = _ordered_scaffolds(set(labels))
    evr = block["explained_variance_ratio"]
    stats = block["scaffold_stats"]
    cent_xy = np.asarray([c["coord2d"] for c in stats["centroids"].values()], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for lab in ordered:
        idx = [i for i, l in enumerate(labels) if l == lab]
        pretty = "No scaffold" if lab == BASELINE else axis_value_label(SCAFFOLD_AXIS_KEY, lab)
        ax.scatter(coords[idx, 0], coords[idx, 1], s=70, alpha=0.8,
                   color=_colour(lab, ordered), edgecolor="white", linewidth=0.6,
                   label=f"{pretty} (n={len(idx)})", zorder=4)
    _draw_centroid_shift(ax, stats, ordered)

    # Frame to the cluster (no forced equal aspect: it would expand the empty
    # y-range and collapse both clouds into a single dot, hiding the separation).
    xlim, ylim = robust_limits(coords, cent_xy)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    inside = ((coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1])
              & (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1]))
    n_off = int((~inside).sum())
    if n_off:
        ax.text(0.99, 0.01, f"{n_off} outlier point(s) off-view",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="#777777", style="italic")
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1  ({evr[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2  ({evr[1]:.0%} of variance)" if len(evr) > 1 else "PC2")

    sep = _separation_note(stats.get("silhouette"), stats.get("silhouette_ci_low"),
                           stats.get("silhouette_ci_high"))
    ax.set_title(
        wrapped(f"The model's internal state, with vs without a debiasing scaffold "
                f"({model})", 62)
        + "\nFar-apart clouds mean the scaffold changed how the model reads the prompt."
        + f"\n{_scatter_subtitle(ptype, feat, block['n_samples'], sep)}", fontsize=11)
    ax.legend(loc="best", frameon=True, framealpha=0.92, title="Coloured by scaffold")
    return finish(fig, out_path)


# ── Per-axis PCA panels ───────────────────────────────────────────────────────


def _cat_value(sample: dict, key: str) -> str:
    """Read one CATEGORICAL axis as a display string (scaffold None -> baseline)."""
    if key == SCAFFOLD_AXIS_KEY:
        return sample.get("scaffold_id") or BASELINE
    return str(sample.get(key, "(missing)"))


def _cont_value(sample: dict, key: str) -> float:
    """Read one CONTINUOUS axis scalar (NaN when missing so it greys out)."""
    v = sample.get(key)
    return float(v) if isinstance(v, (int, float)) else float("nan")


def _axis_silhouette(block: dict, key: str) -> float | None:
    """The stored silhouette for a categorical axis out of one PCA cell."""
    if key == SCAFFOLD_AXIS_KEY:
        return block["scaffold_stats"].get("silhouette")
    return block.get("axis_separation", {}).get(key, {}).get("silhouette")


def _draw_one_axis(ax, block: dict, axis, evr: list[float]):
    """Draw one cell coloured by ``axis`` (continuous colormap or discrete legend).

    Continuous answer-distribution scalars get a sequential colormap (mappable
    returned for the colorbar); categorical axes get the discrete-legend scatter.
    Returns (mappable_or_None, off_view_count).
    """
    samples = block["samples"]
    if axis.continuous:
        vals = [_cont_value(s, axis.key) for s in samples]
        coords = np.asarray([s["coord2d"] for s in samples], dtype=float)
        return draw_continuous_scatter(ax, coords, vals, evr), 0
    vals = [_cat_value(s, axis.key) for s in samples]
    coords = np.asarray([s["coord2d"] for s in samples], dtype=float)
    return None, draw_axis_scatter(ax, coords, vals, evr, axis.key)


def plot_pca_by_axis(block: dict, axis, feat: FeatureLayer, model: str, out_path: Path) -> Path:
    """Standalone PC1-PC2 scatter of the answer projection coloured by one axis."""
    evr = block["explained_variance_ratio"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mappable, n_off = _draw_one_axis(ax, block, axis, evr)
    if n_off:
        ax.text(0.99, 0.01, f"{n_off} outlier point(s) off-view",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="#777777", style="italic")
    pretty = axis_title(axis.key, axis.pretty)
    sep, cap, n_distinct = "", None, None
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02, label=pretty)
    else:
        sep = _separation_note(_axis_silhouette(block, axis.key))
        cat_values = [_cat_value(s, axis.key) for s in block["samples"]]
        n_distinct = len({v for v in cat_values if v is not None})
        cap = axis_caption(cat_values)
        ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=8,
                  title=f"Coloured by {pretty}", title_fontsize=9)
    subtitle = _scatter_subtitle(_AXIS_PANEL_POSITION, feat, block["n_samples"], sep)
    if cap:
        subtitle += f"\n{cap}"
    # The "separate clusters = encodes this" gloss only makes sense with >=2 colours to
    # contrast; gate it so a single-value axis or a continuous colourbar reads honestly.
    if mappable is not None:
        howto = "\nColour = the value; a smooth gradient across the cloud means the state tracks it."
    elif n_distinct is not None and n_distinct <= 1:
        howto = "\nOnly one value is present in this run — no contrast to show."
    else:
        howto = "\nSeparate clusters mean the model encodes this; overlap means it does not."
    ax.set_title(
        wrapped(f"The model's internal state, coloured by {pretty}  ({model})", 62)
        + howto + f"\n{subtitle}", fontsize=11)
    return finish(fig, out_path)


def plot_axes_grid(block: dict, feat: FeatureLayer, model: str, out_path: Path) -> Path:
    """Small-multiples grid: the answer projection recoloured by EVERY axis."""
    evr = block["explained_variance_ratio"]
    n = len(_AXES)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows))
    flat = axes.ravel()
    for ax, axis in zip(flat, _AXES):
        mappable, _ = _draw_one_axis(ax, block, axis, evr)
        title = axis_title(axis.key, axis.pretty)
        if axis.continuous:
            fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
        else:
            cap = axis_caption([_cat_value(s, axis.key) for s in block["samples"]])
            title += f"\n({cap})" if cap else ""
            ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=6.5,
                      markerscale=0.8, handletextpad=0.3, borderpad=0.3)
        ax.set_title(title, fontsize=10)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle(
        wrapped(f"The same internal state, coloured many ways  ({model})", 78)
        + "\nEach panel recolours the model's state at the answer token. Separate "
          "clusters = the model encodes that property; overlap = it does not."
        + f"\n{_layer_phrase(feat)}, n={block['n_samples']}",
        fontsize=12, fontweight="bold")
    return finish(fig, out_path)


# ── Centroid shift / silhouette / EV bars ─────────────────────────────────────


def plot_centroid_shift_bars(results: dict, model: str, out_path: Path) -> Path:
    """Grouped bars: centroid-shift magnitude per (layer, position) + bootstrap CI."""
    layers = list(results.keys())
    rows = []  # (position, layer, magnitude, ci_lo, ci_hi)
    for layer in layers:
        for ptype in _POSITIONS:
            block = results[layer].get(ptype)
            if block is None:
                continue
            for sh in block["scaffold_stats"].get("shifts", {}).values():
                rows.append((ptype, layer, sh["shift_magnitude"],
                             sh.get("shift_ci_low", sh["shift_magnitude"]),
                             sh.get("shift_ci_high", sh["shift_magnitude"])))
    positions = [p for p in _POSITIONS if any(r[0] == p for r in rows)]
    by_cell = {(p, L): (m, lo, hi) for p, L, m, lo, hi in rows}
    n_by_pos = {p: max((results[L][p]["n_samples"] for L in layers if p in results[L]),
                       default=0) for p in positions}

    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    width = 0.86 / max(len(layers), 1)
    x = np.arange(len(positions))
    # Colour each layer along a depth gradient (shallow -> deep) so the 15 series
    # read as one ordered ramp instead of clashing repeated hues. 120 bars would
    # bury per-bar numbers, so we let the bars + CI whiskers carry the values.
    cmap = plt.get_cmap("viridis")
    for j, layer in enumerate(layers):
        vals = [by_cell.get((p, layer), (0.0, 0.0, 0.0)) for p in positions]
        mags = [v[0] for v in vals]
        elo = [max(0.0, v[0] - v[1]) for v in vals]
        ehi = [max(0.0, v[2] - v[0]) for v in vals]
        xpos = x + j * width - 0.43 + width / 2
        ax.bar(xpos, mags, width, color=cmap(j / max(len(layers) - 1, 1)),
               label=f"Layer {layer}", zorder=3)
        ax.errorbar(xpos, mags, yerr=[elo, ehi], fmt="none", ecolor="#555555",
                    elinewidth=0.9, capsize=2, capthick=0.9, zorder=4)
    top = max((r[4] for r in rows), default=1.0)
    ax.set_ylim(0, top * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_wrap_tick(position_label(p), 12)}\n(n={n_by_pos[p]})"
                        for p in positions], fontsize=8)
    ax.set_xlabel("Where in the prompt we read the model's state")
    ax.set_ylabel("How far the scaffold moved the state\n(distance between cloud centres)")
    ax.set_title(
        wrapped(f"How much the debiasing scaffold shifts the model's internal state "
                f"({model})", 64)
        + "\nTaller bars = a bigger shift. Bars run shallow (dark) to deep (yellow); "
          "whiskers show 95% confidence intervals.")
    ax.legend(title="Read at layer", frameon=True, ncol=2, fontsize=8,
              title_fontsize=9, loc="upper left")
    return finish(fig, out_path)


def plot_silhouette_separability(results: dict, feat: FeatureLayer, model: str, out_path: Path) -> Path:
    """Per-axis silhouette separability at the feature layer / answer position."""
    layer = feat.layer_key
    block = results[layer].get(_AXIS_PANEL_POSITION) or next(
        (results[layer][p] for p in _POSITIONS if p in results[layer]), None)
    if block is None:
        return out_path
    pairs = [("scaffold", block["scaffold_stats"].get("silhouette"),
              block["scaffold_stats"].get("silhouette_ci_low"),
              block["scaffold_stats"].get("silhouette_ci_high"))]
    # Skip the duplicate "scaffold" entry already shown from scaffold_stats above.
    for axis, sep in block.get("axis_separation", {}).items():
        if axis != "scaffold":
            pairs.append((axis, sep.get("silhouette"), None, None))
    pairs = [(a, s, lo, hi) for a, s, lo, hi in pairs if s is not None]
    pairs.sort(key=lambda t: t[1], reverse=True)

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    x = np.arange(len(pairs))
    vals = [p[1] for p in pairs]
    colours = [PALETTE[0] if p[0] == "scaffold" else "#999999" for p in pairs]
    ax.bar(x, vals, color=colours, zorder=3)
    for xi, (_a, s, lo, hi) in zip(x, pairs):
        if lo is not None:
            ax.errorbar(xi, s, yerr=[[max(0.0, s - lo)], [max(0.0, hi - s)]],
                        fmt="none", ecolor="#333333", elinewidth=1.6, capsize=5,
                        capthick=1.6, zorder=4)
        ax.text(xi, max(s, hi or s) + 0.005, f"{s:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    ax.axhline(0, color="#888888", lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap_tick(axis_title(p[0], p[0]), 12) for p in pairs],
                       fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("How cleanly the groups separate\n(0 = overlapping, 1 = fully apart)")
    ax.text(0.0, -0.42, "Bars below 0 mean those groups land closer together than "
            "random, i.e. the model does not separate them.",
            transform=ax.transAxes, fontsize=8, color="#777777", style="italic",
            ha="left", va="top")
    ax.set_title(
        wrapped(f"Which prompt or answer properties the model encodes in its state "
                f"({model}, n={block['n_samples']})", 70)
        + "\nTaller bars = the model separates those groups more strongly. Read at "
        + f"{position_label(_AXIS_PANEL_POSITION)}, {_layer_phrase(feat)}."
        + "\nBlue bar (scaffold) shows a 95% confidence interval.", fontsize=11)
    return finish(fig, out_path)


def plot_explained_variance(results: dict, feat: FeatureLayer, model: str, out_path: Path) -> Path:
    """Grouped bars of EV%(PC1..3) per position at the feature layer."""
    layer = feat.layer_key
    cells = results[layer]
    positions = [p for p in _POSITIONS if p in cells]
    n_pc = 3
    direction_labels = ("1st direction", "2nd direction", "3rd direction")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(positions))
    width = 0.8 / n_pc
    for c in range(n_pc):
        vals = [cells[p]["explained_variance_ratio"][c] for p in positions]
        bars = ax.bar(x + c * width - 0.4 + width / 2, vals, width,
                      color=PALETTE[(c + 1) % len(PALETTE)], label=direction_labels[c])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.0%}", ha="center", va="bottom", fontsize=8.5)
    cum3 = [sum(cells[p]["explained_variance_ratio"][:3]) for p in positions]
    tallest = max(cells[p]["explained_variance_ratio"][0] for p in positions)
    ax.set_ylim(0, min(1.0, tallest + 0.12))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_wrap_tick(position_label(p))}\n(n={cells[p]['n_samples']})"
                        for p in positions], fontsize=8.5)
    ax.set_xlabel("Where in the prompt we read the model's state")
    ax.set_ylabel("Share of the spread captured")
    cum_note = ",  ".join(f"{position_label(p)} {c:.0%}" for p, c in zip(positions, cum3))
    ax.set_title(
        wrapped(f"How much of the model's variation the 2D view captures  ({model})", 64)
        + "\nThe scatter plots show the first two bars; higher means the 2D picture "
          "is more faithful."
        + "\n" + wrapped(f"First three together: {cum_note}", 78)
        + f"\n{_layer_phrase(feat)}", fontsize=10.5)
    ax.legend(title="Spread direction", frameon=True)
    return finish(fig, out_path)


# ── Layer-aware separability (heatmap + key-axis sweep) ───────────────────────


def plot_silhouette_heatmap(table: dict, model: str, out_path: Path) -> Path:
    """(layer x axis) silhouette heatmap: at what DEPTH each axis separates."""
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(table["layers"]),
                                    0.45 * len(table["axes"]) + 2.0))
    draw_silhouette_heatmap(ax, table)
    ax.set_title(
        wrapped(f"At which depth the model starts to encode each property  ({model})", 56)
        + f"\nRead at {position_label(table['position'])}. "
          "Red = cleanly separated, white = overlapping."
        + "\nRead each row left to right to see how deep the separation appears.",
        fontsize=10.5)
    return finish(fig, out_path)


def plot_silhouette_layer_sweep(table: dict, model: str, out_path: Path) -> Path:
    """Silhouette-vs-layer sweep for the KEY axes (accuracy / context / role / scaffold)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    draw_layer_sweep(ax, table, KEY_SWEEP_AXIS_KEYS, AXIS_PALETTE)
    ax.set_title(
        wrapped(f"How deep in the network each property becomes readable  ({model})", 62)
        + f"\nRead at {position_label(table['position'])}. A line rising above 0 marks "
          "the depth where that property separates.", fontsize=11)
    return finish(fig, out_path)


# ── Orchestration ─────────────────────────────────────────────────────────────


def _behavioral_plots(dataset: GeometryDataset, plots_dir: Path) -> list[Path]:
    """The per-condition 2-opt/3-opt figure + the readout-comparison figure."""
    return [
        plot_accuracy_by_condition(dataset, plots_dir / "accuracy_by_condition.png"),
        plot_accuracy_by_readout(dataset, plots_dir / "accuracy_by_readout.png"),
    ]


def _per_axis_plots(cells: dict, feat: FeatureLayer, model: str, plots_dir: Path) -> list[Path]:
    """Recolour the feature-layer answer projection by EVERY registry axis."""
    panel = cells.get(_AXIS_PANEL_POSITION) or next(
        (cells[p] for p in _POSITIONS if p in cells), None)
    if panel is None:
        log("[viz] no projection cell available; skipping per-axis panels")
        return []
    written = [plot_pca_by_axis(panel, axis, feat, model,
                                plots_dir / f"pca_by_{axis.key}.png")
               for axis in _AXES]
    written.append(plot_axes_grid(panel, feat, model, plots_dir / "pca_axes_grid.png"))
    return written


def _layer_separability_plots(table: dict, model: str, plots_dir: Path) -> list[Path]:
    """The (layer x axis) silhouette heatmap + the key-axis depth sweep."""
    if not table or not table.get("layers"):
        log("[viz] no layer_axis_silhouette table; skipping the depth heatmap/sweep")
        return []
    return [
        plot_silhouette_heatmap(table, model, plots_dir / "silhouette_by_layer_axis.png"),
        plot_silhouette_layer_sweep(table, model, plots_dir / "silhouette_layer_sweep.png"),
    ]


def _geometry_plots(payload: dict, model: str, plots_dir: Path) -> list[Path]:
    """PCA scatters, per-axis panels, centroid-shift, silhouette, EV%, depth views."""
    results = payload["results"]
    written: list[Path] = []
    # Adaptive scatter depth: the captured layer where the scaffold structure is
    # clearest (comparable RELATIVE depth across model sizes), not a fixed index.
    feat = _resolve_feature_layer(results)
    log(f"[viz] feature layer (scatters) = {feat.title_suffix()} "
        f"silhouette={feat.silhouette}")
    cells = results[feat.layer_key]
    for ptype in _POSITIONS:
        block = cells.get(ptype)
        if block is None:
            log(f"[viz] no projection for position '{ptype}'; skipping its scatter")
            continue
        written.append(plot_pca_scatter(block, ptype, feat, model,
                                        plots_dir / f"pca_scatter_{ptype}.png"))
    written += _per_axis_plots(cells, feat, model, plots_dir)
    written += _layer_separability_plots(payload.get("layer_axis_silhouette", {}),
                                         model, plots_dir)
    written.append(plot_centroid_shift_bars(results, model,
                                            plots_dir / "centroid_shift_by_position.png"))
    written.append(plot_silhouette_separability(results, feat, model,
                                                plots_dir / "silhouette_separability.png"))
    written.append(plot_explained_variance(results, feat, model,
                                            plots_dir / "explained_variance.png"))
    return written


def main() -> None:
    """Load samples + the PCA projections and render every plot."""
    args = parse_args()
    log_header("VISUALIZE SESGO GEOMETRY")
    apply_style()

    dataset = GeometryDataset.from_json(args.samples)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    model_dir = args.out_dir / "sesgo" / "geometry" / dataset.model_name
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    written = _behavioral_plots(dataset, plots_dir)

    proj_path = model_dir / "analysis" / "projections.json"
    if proj_path.exists():
        payload = load_json(proj_path)
        written += _geometry_plots(payload, dataset.model_name, plots_dir)
    else:
        log_section("missing projections.json")
        log(f"[viz] {proj_path} not found — run analyze_geometry.py first for the "
            "PCA geometry plots (only the behavioural plots were written)")

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
