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
    baseline->interpretive shift arrow, at every structural position.
  pca_by_<axis>.png / pca_axes_grid.png : the SAME answer-position projection
    recoloured by EVERY per-sample axis (scaffold, origin, language, bias category,
    question polarity, context condition, accuracy, identities, gold, label style).
  centroid_shift_by_position.png : centroid-shift magnitude per (layer, position)
    with bootstrap 95% CIs.
  silhouette_separability.png : per-axis silhouette separability with bootstrap CI
    on the scaffold axis.
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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import load_json  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import GeometryDataset  # noqa: E402

from sesgo.geometry.geometry_plot_helpers import (  # noqa: E402
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

_SCATTER_LAYER = "mean"  # representative layer for the PCA scatter panels
_POSITIONS = ("turn", "think_open", "think_close", "answer")
_AXIS_PANEL_POSITION = "answer"  # the projection the per-axis scatters recolour
_AXES: tuple[tuple[str, str], ...] = (
    ("scaffold_id", "scaffold"),
    ("origin", "origin (BBQ-adapted vs original)"),
    ("language", "language"),
    ("bias_category", "bias category"),
    ("question_polarity", "question polarity"),
    ("context_condition", "context condition (ambig vs disambig)"),
    ("accuracy", "accuracy (correct vs incorrect)"),
    ("target_identity", "target identity"),
    ("other_identity", "other identity"),
    ("gold_label", "gold label"),
    ("label_style", "label style"),
)


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


def _ordered_scaffolds(labels: set[str]) -> list[str]:
    """Scaffold labels with the baseline first, the rest sorted after it."""
    rest = sorted(labels - {BASELINE})
    return ([BASELINE] if BASELINE in labels else []) + rest


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
        ax.annotate(f"shift = {sh['shift_magnitude']:.1f}{ci}",
                    xy=((bx + tx) / 2, (by + ty) / 2), fontsize=8.5,
                    fontweight="bold", color="#333333", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999",
                              alpha=0.85), zorder=7)


def plot_pca_scatter(block: dict, ptype: str, model: str, out_path: Path) -> Path:
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
        ax.scatter(coords[idx, 0], coords[idx, 1], s=70, alpha=0.8,
                   color=_colour(lab, ordered), edgecolor="white", linewidth=0.6,
                   label=f"{lab} (n={len(idx)})", zorder=4)
    _draw_centroid_shift(ax, stats, ordered)

    xlim, ylim = robust_limits(coords, cent_xy)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="datalim")
    fig.canvas.draw()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    inside = ((coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1])
              & (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1]))
    n_off = int((~inside).sum())
    if n_off:
        ax.text(0.99, 0.01, f"{n_off} outlier point(s) off-view",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="#777777", style="italic")
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1  ({evr[0]:.0%} explained variance)")
    ax.set_ylabel(f"PC2  ({evr[1]:.0%} explained variance)" if len(evr) > 1 else "PC2")

    sil = stats.get("silhouette")
    subtitle = f"layer={_SCATTER_LAYER}, n={block['n_samples']}"
    if sil is not None:
        lo, hi = stats.get("silhouette_ci_low"), stats.get("silhouette_ci_high")
        subtitle += f", scaffold silhouette={sil:.2f}"
        if lo is not None:
            subtitle += f" [{lo:.2f}, {hi:.2f}]"
    ax.set_title(wrapped(f"SESGO representational geometry @ {ptype}  ({model})")
                 + f"\n{subtitle}", fontsize=12)
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    return finish(fig, out_path)


# ── Per-axis PCA panels ───────────────────────────────────────────────────────


def _axis_value(sample: dict, axis: str) -> str:
    """Read one per-sample axis as a display string (scaffold None -> baseline)."""
    if axis == "scaffold_id":
        return sample.get("scaffold_id") or BASELINE
    return str(sample.get(axis, "(missing)"))


def plot_pca_by_axis(block: dict, axis: str, pretty: str, model: str, out_path: Path) -> Path:
    """Standalone PC1-PC2 scatter of the answer projection coloured by one axis."""
    coords = np.asarray([s["coord2d"] for s in block["samples"]], dtype=float)
    values = [_axis_value(s, axis) for s in block["samples"]]
    evr = block["explained_variance_ratio"]
    sep = block.get("axis_separation", {}).get(axis, {}) if axis != "scaffold_id" else {}
    sil = (block["scaffold_stats"].get("silhouette") if axis == "scaffold_id"
           else sep.get("silhouette"))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    n_off = draw_axis_scatter(ax, coords, values, evr)
    if n_off:
        ax.text(0.99, 0.01, f"{n_off} outlier point(s) off-view",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="#777777", style="italic")
    subtitle = f"@ {_AXIS_PANEL_POSITION}, layer={_SCATTER_LAYER}, n={block['n_samples']}"
    if sil is not None:
        subtitle += f", silhouette={sil:.2f}"
    cap = axis_caption(values)
    if cap:
        subtitle += f"\n{cap}"
    ax.set_title(wrapped(f"SESGO geometry coloured by {pretty}  ({model})")
                 + f"\n{subtitle}", fontsize=12)
    ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=8, title=pretty,
              title_fontsize=9)
    return finish(fig, out_path)


def plot_axes_grid(block: dict, model: str, out_path: Path) -> Path:
    """Small-multiples grid: the answer projection recoloured by EVERY axis."""
    coords = np.asarray([s["coord2d"] for s in block["samples"]], dtype=float)
    evr = block["explained_variance_ratio"]
    n = len(_AXES)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows))
    flat = axes.ravel()
    for ax, (axis, pretty) in zip(flat, _AXES):
        values = [_axis_value(s, axis) for s in block["samples"]]
        draw_axis_scatter(ax, coords, values, evr)
        cap = axis_caption(values)
        ax.set_title(pretty + (f"\n({cap})" if cap else ""), fontsize=10)
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=6.5,
                  markerscale=0.8, handletextpad=0.3, borderpad=0.3)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle(wrapped(f"SESGO answer-position geometry by every axis  ({model}, "
                         f"layer={_SCATTER_LAYER}, n={block['n_samples']})", 78),
                 fontsize=13, fontweight="bold")
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

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    width = 0.8 / max(len(layers), 1)
    x = np.arange(len(positions))
    for j, layer in enumerate(layers):
        vals = [by_cell.get((p, layer), (0.0, 0.0, 0.0)) for p in positions]
        mags = [v[0] for v in vals]
        elo = [max(0.0, v[0] - v[1]) for v in vals]
        ehi = [max(0.0, v[2] - v[0]) for v in vals]
        xpos = x + j * width - 0.4 + width / 2
        ax.bar(xpos, mags, width, color=PALETTE[(j + 1) % len(PALETTE)],
               label=f"layer={layer}", zorder=3)
        ax.errorbar(xpos, mags, yerr=[elo, ehi], fmt="none", ecolor="#333333",
                    elinewidth=1.4, capsize=4, capthick=1.4, zorder=4)
        for xi, (m, _lo, hi) in zip(xpos, vals):
            if m > 0:
                ax.text(xi, hi + 0.02 * max(r[2] for r in rows), f"{m:.1f}",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    top = max((r[4] for r in rows), default=1.0)
    ax.set_ylim(0, top * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(n={n_by_pos[p]})".replace("_", " ") for p in positions])
    ax.set_xlabel("structural position")
    ax.set_ylabel("centroid shift magnitude\n(full-dim L2, baseline -> interpretive)")
    ax.set_title(wrapped("How far the interpretive scaffold moves the representation "
                         f"(bootstrap 95% CI)  ({model})", 64))
    ax.legend(title="layer reduction", frameon=True)
    return finish(fig, out_path)


def plot_silhouette_separability(results: dict, model: str, out_path: Path) -> Path:
    """Per-axis silhouette separability at the representative layer/position."""
    layer = _SCATTER_LAYER if _SCATTER_LAYER in results else next(iter(results))
    block = results[layer].get(_AXIS_PANEL_POSITION) or next(
        (results[layer][p] for p in _POSITIONS if p in results[layer]), None)
    if block is None:
        return out_path
    pairs = [("scaffold", block["scaffold_stats"].get("silhouette"),
              block["scaffold_stats"].get("silhouette_ci_low"),
              block["scaffold_stats"].get("silhouette_ci_high"))]
    for axis, sep in block.get("axis_separation", {}).items():
        pairs.append((axis, sep.get("silhouette"), None, None))
    pairs = [(a, s, lo, hi) for a, s, lo, hi in pairs if s is not None]
    pairs.sort(key=lambda t: t[1], reverse=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
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
    ax.set_xticklabels([p[0].replace("_", "\n") for p in pairs], fontsize=8.5)
    ax.set_ylabel("silhouette separability\n(full PCA space)")
    ax.set_title(wrapped(f"Which axis separates the representation @ "
                         f"{_AXIS_PANEL_POSITION}, layer={layer}  "
                         f"({model}, n={block['n_samples']})", 64)
                 + "\nscaffold axis shows bootstrap 95% CI")
    return finish(fig, out_path)


def plot_explained_variance(results: dict, model: str, out_path: Path) -> Path:
    """Grouped bars of EV%(PC1..3) per position at the representative layer."""
    layer = _SCATTER_LAYER if _SCATTER_LAYER in results else next(iter(results))
    cells = results[layer]
    positions = [p for p in _POSITIONS if p in cells]
    n_pc = 3
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(positions))
    width = 0.8 / n_pc
    for c in range(n_pc):
        vals = [cells[p]["explained_variance_ratio"][c] for p in positions]
        bars = ax.bar(x + c * width - 0.4 + width / 2, vals, width,
                      color=PALETTE[(c + 1) % len(PALETTE)], label=f"PC{c + 1}")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.0%}", ha="center", va="bottom", fontsize=8.5)
    cum3 = [sum(cells[p]["explained_variance_ratio"][:3]) for p in positions]
    tallest = max(cells[p]["explained_variance_ratio"][0] for p in positions)
    ax.set_ylim(0, min(1.0, tallest + 0.12))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(n={cells[p]['n_samples']})".replace("_", " ")
                        for p in positions])
    ax.set_xlabel("structural position")
    ax.set_ylabel("explained variance ratio")
    cum_note = ",  ".join(f"{p} {c:.0%}" for p, c in zip(positions, cum3))
    ax.set_title(wrapped(f"PCA explained variance (PC1-3) @ layer={layer}  ({model})", 64)
                 + "\n" + wrapped(f"cumulative PC1-3:  {cum_note}", 70), fontsize=11.5)
    ax.legend(title="component", frameon=True)
    return finish(fig, out_path)


# ── Orchestration ─────────────────────────────────────────────────────────────


def _behavioral_plots(dataset: GeometryDataset, plots_dir: Path) -> list[Path]:
    """The per-condition 2-opt/3-opt figure + the readout-comparison figure."""
    return [
        plot_accuracy_by_condition(dataset, plots_dir / "accuracy_by_condition.png"),
        plot_accuracy_by_readout(dataset, plots_dir / "accuracy_by_readout.png"),
    ]


def _per_axis_plots(cells: dict, model: str, plots_dir: Path) -> list[Path]:
    """Recolour the answer-position projection by EVERY per-sample axis."""
    panel = cells.get(_AXIS_PANEL_POSITION) or next(
        (cells[p] for p in _POSITIONS if p in cells), None)
    if panel is None:
        log("[viz] no projection cell available; skipping per-axis panels")
        return []
    written = [plot_pca_by_axis(panel, axis, pretty, model, plots_dir / f"pca_by_{axis}.png")
               for axis, pretty in _AXES]
    written.append(plot_axes_grid(panel, model, plots_dir / "pca_axes_grid.png"))
    return written


def _geometry_plots(results: dict, model: str, plots_dir: Path) -> list[Path]:
    """PCA scatters, per-axis panels, centroid-shift, silhouette, EV%."""
    written: list[Path] = []
    scatter_layer = _SCATTER_LAYER if _SCATTER_LAYER in results else next(iter(results))
    cells = results[scatter_layer]
    for ptype in _POSITIONS:
        block = cells.get(ptype)
        if block is None:
            log(f"[viz] no projection for position '{ptype}'; skipping its scatter")
            continue
        written.append(plot_pca_scatter(block, ptype, model,
                                        plots_dir / f"pca_scatter_{ptype}.png"))
    written += _per_axis_plots(cells, model, plots_dir)
    written.append(plot_centroid_shift_bars(results, model,
                                            plots_dir / "centroid_shift_by_position.png"))
    written.append(plot_silhouette_separability(results, model,
                                                plots_dir / "silhouette_separability.png"))
    written.append(plot_explained_variance(results, model,
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
        results = load_json(proj_path)["results"]
        written += _geometry_plots(results, dataset.model_name, plots_dir)
    else:
        log_section("missing projections.json")
        log(f"[viz] {proj_path} not found — run analyze_geometry.py first for the "
            "PCA geometry plots (only the behavioural plots were written)")

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
