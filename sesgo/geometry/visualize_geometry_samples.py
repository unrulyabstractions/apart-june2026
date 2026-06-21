"""Plot SESGO geometry: behavioral scaffold effect + representational geometry.

Run-by-path driver. Renders into out/sesgo/geometry/<MODEL>/plots/:

  BEHAVIORAL - non-thinking (and thinking) abstention accuracy by scaffold. On
    ambiguous SESGO items the gold is UNKNOWN, so accuracy = fraction predicted
    UNKNOWN; the no-scaffold baseline anchors the with/without comparison.
  GEOMETRY   - the actual representational-geometry plots, read from the PCA
    analysis in analysis/projections.json (run analyze_geometry.py first):
      * pca_scatter_<position>.png : PC1-PC2 scatter at a representative layer,
        points coloured by scaffold ((baseline) vs interpretive_direction),
        per-class centroids marked, and an arrow drawn for the centroid SHIFT.
      * pca_by_<axis>.png : the SAME PCA projection (answer position, representative
        layer) recoloured by EVERY per-sample axis — scaffold, origin, language,
        bias_category, question_polarity, context_condition (ambig vs disambig),
        accuracy (correct vs incorrect), target_identity, other_identity,
        gold_label, label_style — each with a compact legend. High-cardinality
        identity axes are capped at the top-K values + an "(other)" bucket.
      * pca_axes_grid.png : all those axis panels in one small-multiples grid.
      * centroid_shift_by_position.png : centroid-shift magnitude across every
        (layer, position) cell.
      * explained_variance.png : EV%(PC1..3) context per position.

Robust to missing data: positions absent from projections.json are skipped, and
the behavioural plots fall back to the samples.json dataset directly.

Usage:
  uv run python sesgo/geometry/analyze_geometry.py \
      out/sesgo/geometry/Qwen3-0.6B/samples.json   # writes projections.json
  uv run python sesgo/geometry/visualize_geometry_samples.py \
      out/sesgo/geometry/Qwen3-0.6B/samples.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import load_json  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import GeometryDataset  # noqa: E402

_BASELINE = "(baseline)"
_SCATTER_LAYER = "mean"  # representative layer for the PCA scatter panels
_POSITIONS = ("turn", "think_open", "think_close", "answer")
# The structural position whose PCA panel the per-axis scatters colour. "answer"
# is the most behaviourally meaningful (the residual over the chosen token).
_AXIS_PANEL_POSITION = "answer"
# Every per-sample axis we render a coloured PCA panel for, in display order. The
# pretty names are the panel titles / filename stems.
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
# High-cardinality axes: cap the legend at the top-K most frequent values, fold
# the rest into a single "(other)" bucket so the legend stays legible.
_TOP_K = 8
_OTHER_BUCKET = "(other)"
# Colourblind-safe (Okabe-Ito) extended for higher-cardinality axes; first entry
# anchors the scaffold baseline, last is reserved for the "(other)" bucket.
_PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
_AXIS_PALETTE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#F0E442", "#999999", "#7E2F8E", "#4DBEEE",
)
_OTHER_COLOUR = "#bbbbbb"  # neutral grey for the folded-in "(other)" bucket
_SAVE = dict(dpi=150, bbox_inches="tight")  # consistent crisp export


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for geometry visualization."""
    parser = argparse.ArgumentParser(
        description="Plot SESGO geometry (behavioral + representational geometry)"
    )
    parser.add_argument("samples", type=Path, help="samples.json (a GeometryDataset)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"), help="Base output directory"
    )
    return parser.parse_args()


def _apply_style() -> None:
    """A clean, legible, seaborn-ish global style for every figure."""
    plt.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.edgecolor": "#555555",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
        }
    )


def _colour(label: str, ordered: list[str]) -> str:
    """Stable colour per label; baseline always the first palette entry."""
    if label == _BASELINE:
        return _PALETTE[0]
    rest = [l for l in ordered if l != _BASELINE]
    return _PALETTE[(rest.index(label) + 1) % len(_PALETTE)]


def _wrapped_title(text: str, width: int = 58) -> str:
    """Wrap long titles so the model name never clips off the right edge."""
    return "\n".join(textwrap.wrap(text, width=width))


def _finish(fig: plt.Figure, out_path: Path) -> Path:
    """Save with crisp, tightly-cropped settings and close the figure."""
    fig.savefig(out_path, **_SAVE)
    plt.close(fig)
    return out_path


# ── Behavioral ───────────────────────────────────────────────────────────────


def _ordered_scaffolds(labels: set[str]) -> list[str]:
    """Scaffold labels with the baseline first, the rest sorted after it."""
    rest = sorted(labels - {_BASELINE})
    return ([_BASELINE] if _BASELINE in labels else []) + rest


def plot_accuracy_by_scaffold(
    dataset: GeometryDataset, level: str, out_path: Path
) -> tuple[Path, dict[str, float]]:
    """Bar chart of abstention accuracy by scaffold; also returns the rates."""
    flags: dict[str, list[bool]] = defaultdict(list)
    for s in dataset.samples:
        label = s.scaffold_id or _BASELINE
        if level == "non_thinking" and s.predicted_non_thinking is not None:
            flags[label].append(s.correct_non_thinking)
        elif level == "thinking" and s.predicted_thinking is not None:
            flags[label].append(s.predicted_thinking.value == "unknown")
    scaffolds = _ordered_scaffolds(set(flags))
    accs = {sc: (sum(f) / len(f) if f else 0.0) for sc, f in flags.items()}
    ns = [len(flags[sc]) for sc in scaffolds]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colours = [_colour(sc, scaffolds) for sc in scaffolds]
    bars = ax.bar(range(len(scaffolds)), [accs[sc] for sc in scaffolds], color=colours)
    for bar, sc, n in zip(bars, scaffolds, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{accs[sc]:.0%}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_xticks(range(len(scaffolds)))
    ax.set_xticklabels([w.replace("_", "\n") for w in scaffolds], fontsize=10)
    ax.set_ylim(0, 1.18)  # headroom so the value labels never collide with title
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_ylabel("abstention accuracy\n(fraction predicted UNKNOWN)")
    ax.set_axisbelow(True)
    pretty = level.replace("_", "-")
    n_total = sum(ns)
    ax.set_title(
        _wrapped_title(
            f"SESGO {pretty} abstention accuracy by scaffold  "
            f"({dataset.model_name}, n={n_total})"
        )
    )
    return _finish(fig, out_path), accs


# ── Representational geometry (from projections.json) ─────────────────────────


def _scatter_arrays(block: dict) -> tuple[np.ndarray, list[str]]:
    """Return the [n,2] PC1-PC2 coords and parallel scaffold labels for a cell."""
    coords, labels = [], []
    for s in block["samples"]:
        coords.append(s["coord2d"])
        labels.append(s["scaffold_id"] or _BASELINE)
    return np.asarray(coords, dtype=float), labels


def _robust_limits(
    coords: np.ndarray, centroids: np.ndarray, pad: float = 0.18
) -> tuple[tuple, tuple]:
    """Tukey-fence x/y limits that keep the cluster + both centroids in frame.

    A handful of PCA outliers can stretch the raw range by an order of magnitude
    (n=25), so we clip to median ± 2.5·IQR per axis, then widen to guarantee every
    centroid (and thus the shift arrow) sits inside.
    """
    q1, med, q3 = np.percentile(coords, [25, 50, 75], axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    lo = med - 2.5 * iqr
    hi = med + 2.5 * iqr
    if centroids.size:  # never crop a centroid out of the frame
        lo = np.minimum(lo, centroids.min(axis=0))
        hi = np.maximum(hi, centroids.max(axis=0))
    span = np.maximum(hi - lo, 1e-6)
    return (lo[0] - pad * span[0], hi[0] + pad * span[0]), (
        lo[1] - pad * span[1],
        hi[1] + pad * span[1],
    )


def _draw_centroid_shift(ax, stats: dict, ordered: list[str]) -> None:
    """Plot each centroid as a diamond and draw the baseline→intervention arrow."""
    cents = stats["centroids"]
    for lab in ordered:
        if lab not in cents:
            continue
        cx, cy = cents[lab]["coord2d"]
        ax.scatter(
            cx, cy, s=320, marker="D", facecolor=_colour(lab, ordered),
            edgecolor="black", linewidth=1.6, zorder=6,
        )
    base = cents.get(_BASELINE)
    for lab, sh in stats.get("shifts", {}).items():
        if base is None or lab not in cents:
            continue
        bx, by = base["coord2d"]
        tx, ty = cents[lab]["coord2d"]
        ax.annotate(
            "", xy=(tx, ty), xytext=(bx, by),
            arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#333333",
                            shrinkA=10, shrinkB=10), zorder=5,
        )
        mx, my = (bx + tx) / 2, (by + ty) / 2
        ax.annotate(
            f"shift = {sh['shift_magnitude']:.1f}", xy=(mx, my),
            fontsize=9, fontweight="bold", color="#333333",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.85),
            zorder=7,
        )


def plot_pca_scatter(block: dict, ptype: str, model: str, out_path: Path) -> Path:
    """PC1-PC2 scatter coloured by scaffold, with centroids + shift arrow."""
    coords, labels = _scatter_arrays(block)
    ordered = _ordered_scaffolds(set(labels))
    evr = block["explained_variance_ratio"]
    stats = block["scaffold_stats"]
    cent_xy = np.asarray(
        [c["coord2d"] for c in stats["centroids"].values()], dtype=float
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for lab in ordered:
        idx = [i for i, l in enumerate(labels) if l == lab]
        ax.scatter(
            coords[idx, 0], coords[idx, 1], s=70, alpha=0.8,
            color=_colour(lab, ordered), edgecolor="white", linewidth=0.6,
            label=f"{lab} (n={len(idx)})", zorder=4,
        )
    _draw_centroid_shift(ax, stats, ordered)

    xlim, ylim = _robust_limits(coords, cent_xy)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # Equal aspect (adjustable='datalim') so PC1/PC2 distances and the shift arrow
    # are read truthfully, while matplotlib only *grows* the shorter axis to fill
    # the panel — no wasteful square padding when PC1 variance dominates.
    ax.set_aspect("equal", adjustable="datalim")
    fig.canvas.draw()  # realise the aspect-adjusted limits before reading them
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # Honest note: count points actually outside the final (equal-aspect) frame.
    inside = (
        (coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1])
        & (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1])
    )
    n_off = int((~inside).sum())
    if n_off:
        ax.text(
            0.99, 0.01, f"{n_off} outlier point(s) off-view",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#777777", style="italic",
        )
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1  ({evr[0]:.0%} explained variance)")
    ax.set_ylabel(f"PC2  ({evr[1]:.0%} explained variance)" if len(evr) > 1 else "PC2")

    ss = block["scaffold_stats"]
    sil = ss.get("silhouette")
    subtitle = f"layer={_SCATTER_LAYER}, n={block['n_samples']}"
    if sil is not None:
        subtitle += f", scaffold silhouette={sil:.2f}"
    ax.set_title(
        _wrapped_title(
            f"SESGO representational geometry @ {ptype}  ({model})"
        ) + f"\n{subtitle}",
        fontsize=12,
    )
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    return _finish(fig, out_path)


# ── Per-axis PCA scatters (colour the SAME projection by every label) ─────────


def _axis_value(sample: dict, axis: str) -> str:
    """Read one per-sample axis as a display string (scaffold None -> baseline)."""
    if axis == "scaffold_id":
        return sample.get("scaffold_id") or _BASELINE
    return str(sample.get(axis, "(missing)"))


def _legend_order(values: list[str]) -> list[str]:
    """Most-frequent-first ordering, capped at top-K + an "(other)" bucket.

    High-cardinality identity axes can have many distinct group strings; folding
    the long tail into "(other)" keeps the legend legible. The baseline scaffold
    label, when present, is pinned first for a stable reading.
    """
    counts = Counter(values)
    ranked = [v for v, _ in counts.most_common()]
    if _BASELINE in ranked:  # pin the scaffold baseline first
        ranked = [_BASELINE] + [v for v in ranked if v != _BASELINE]
    if len(ranked) <= _TOP_K:
        return ranked
    return ranked[:_TOP_K] + [_OTHER_BUCKET]


def _axis_colour(label: str, ordered: list[str]) -> str:
    """Stable colour per axis value; the folded "(other)" bucket is neutral grey."""
    if label == _OTHER_BUCKET:
        return _OTHER_COLOUR
    return _AXIS_PALETTE[ordered.index(label) % len(_AXIS_PALETTE)]


def _bucket(label: str, kept: set[str]) -> str:
    """Map a raw value to its legend bucket (itself, or "(other)" if folded)."""
    return label if label in kept else _OTHER_BUCKET


def _draw_axis_scatter(ax, coords: np.ndarray, values: list[str], evr: list[float]) -> int:
    """Scatter the PCA cloud coloured by one axis; return the off-view count.

    Shared by the standalone per-axis figures and the small-multiples grid so the
    capping / colouring / framing logic lives in exactly one place.
    """
    ordered = _legend_order(values)
    kept = {v for v in ordered if v != _OTHER_BUCKET}
    centroids: list[list[float]] = []
    for lab in ordered:
        idx = [i for i, v in enumerate(values) if _bucket(v, kept) == lab]
        if not idx:
            continue
        # Anchor each group's centroid so the Tukey-fence framing can never clip
        # an entire minority group out of view (e.g. the smaller language group).
        centroids.append(coords[idx].mean(axis=0).tolist())
        ax.scatter(
            coords[idx, 0], coords[idx, 1], s=60, alpha=0.8,
            color=_axis_colour(lab, ordered), edgecolor="white", linewidth=0.5,
            label=f"{lab} (n={len(idx)})", zorder=4,
        )
    xlim, ylim = _robust_limits(coords, np.asarray(centroids, dtype=float))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1  ({evr[0]:.0%} EV)")
    ax.set_ylabel(f"PC2  ({evr[1]:.0%} EV)" if len(evr) > 1 else "PC2")
    inside = (
        (coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1])
        & (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1])
    )
    return int((~inside).sum())


def _axis_caption(values: list[str]) -> str | None:
    """Note capping when a high-cardinality axis was folded to top-K + (other)."""
    n_distinct = len(set(values))
    if n_distinct <= _TOP_K:
        return None
    return f"{n_distinct} distinct values; legend shows top {_TOP_K} + (other)"


def plot_pca_by_axis(block: dict, axis: str, pretty: str, model: str, out_path: Path) -> Path:
    """Standalone PC1-PC2 scatter of the answer projection coloured by one axis."""
    coords = np.asarray([s["coord2d"] for s in block["samples"]], dtype=float)
    values = [_axis_value(s, axis) for s in block["samples"]]
    evr = block["explained_variance_ratio"]
    sep = block.get("axis_separation", {}).get(axis, {}) if axis != "scaffold_id" else {}
    sil = (block["scaffold_stats"].get("silhouette") if axis == "scaffold_id"
           else sep.get("silhouette"))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    n_off = _draw_axis_scatter(ax, coords, values, evr)
    if n_off:
        ax.text(0.99, 0.01, f"{n_off} outlier point(s) off-view",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="#777777", style="italic")
    subtitle = f"@ {_AXIS_PANEL_POSITION}, layer={_SCATTER_LAYER}, n={block['n_samples']}"
    if sil is not None:
        subtitle += f", silhouette={sil:.2f}"
    cap = _axis_caption(values)
    if cap:
        subtitle += f"\n{cap}"
    ax.set_title(
        _wrapped_title(f"SESGO geometry coloured by {pretty}  ({model})") + f"\n{subtitle}",
        fontsize=12,
    )
    ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=8, title=pretty,
              title_fontsize=9)
    return _finish(fig, out_path)


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
        _draw_axis_scatter(ax, coords, values, evr)
        cap = _axis_caption(values)
        ax.set_title(pretty + (f"\n({cap})" if cap else ""), fontsize=10)
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=6.5,
                  markerscale=0.8, handletextpad=0.3, borderpad=0.3)
    for ax in flat[n:]:  # blank any unused grid cells
        ax.axis("off")
    fig.suptitle(
        _wrapped_title(
            f"SESGO answer-position geometry by every axis  ({model}, "
            f"layer={_SCATTER_LAYER}, n={block['n_samples']})", 78
        ),
        fontsize=13, fontweight="bold",
    )
    return _finish(fig, out_path)


def plot_centroid_shift_bars(results: dict, model: str, out_path: Path) -> Path:
    """Grouped bars: centroid-shift magnitude per (layer, position) cell."""
    layers = list(results.keys())
    rows = []  # (position, layer, magnitude) for the only non-baseline scaffold
    for layer in layers:
        for ptype in _POSITIONS:
            block = results[layer].get(ptype)
            if block is None:
                continue
            for lab, sh in block["scaffold_stats"].get("shifts", {}).items():
                rows.append((ptype, layer, sh["shift_magnitude"]))
    positions = [p for p in _POSITIONS if any(r[0] == p for r in rows)]
    by_cell = {(p, L): m for p, L, m in rows}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.8 / max(len(layers), 1)
    x = np.arange(len(positions))
    for j, layer in enumerate(layers):
        vals = [by_cell.get((p, layer), 0.0) for p in positions]
        bars = ax.bar(
            x + j * width - 0.4 + width / 2, vals, width,
            color=_PALETTE[(j + 1) % len(_PALETTE)], label=f"layer={layer}",
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold",
                )
    top = max((m for *_, m in rows), default=1.0)
    ax.set_ylim(0, top * 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in positions])
    ax.set_xlabel("structural position")
    ax.set_ylabel("centroid shift magnitude\n(full-dim L2, baseline → interpretive)")
    ax.set_title(
        _wrapped_title(
            f"How far the interpretive scaffold moves the representation  ({model})"
        )
    )
    ax.legend(title="layer reduction", frameon=True)
    return _finish(fig, out_path)


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
        bars = ax.bar(
            x + c * width - 0.4 + width / 2, vals, width,
            color=_PALETTE[(c + 1) % len(_PALETTE)], label=f"PC{c + 1}",
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.0%}", ha="center", va="bottom", fontsize=8.5,
            )
    cum3 = [sum(cells[p]["explained_variance_ratio"][:3]) for p in positions]
    tallest = max(cells[p]["explained_variance_ratio"][0] for p in positions)
    ax.set_ylim(0, min(1.0, tallest + 0.12))  # headroom for the PC1 value labels
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in positions])
    ax.set_xlabel("structural position")
    ax.set_ylabel("explained variance ratio")
    cum_note = ",  ".join(f"{p} {c:.0%}" for p, c in zip(positions, cum3))
    ax.set_title(
        _wrapped_title(f"PCA explained variance (PC1-3) @ layer={layer}  ({model})", 64)
        + "\n" + _wrapped_title(f"cumulative PC1-3:  {cum_note}", 70),
        fontsize=11.5,
    )
    ax.legend(title="component", frameon=True)
    return _finish(fig, out_path)


# ── Orchestration ─────────────────────────────────────────────────────────────


def _behavioral_plots(dataset: GeometryDataset, plots_dir: Path) -> list[Path]:
    """Render the non-thinking (always) + thinking (when parsed) accuracy bars."""
    written = [
        plot_accuracy_by_scaffold(
            dataset, "non_thinking",
            plots_dir / "accuracy_by_scaffold_non_thinking.png",
        )[0]
    ]
    if any(s.predicted_thinking is not None for s in dataset.samples):
        written.append(
            plot_accuracy_by_scaffold(
                dataset, "thinking",
                plots_dir / "accuracy_by_scaffold_thinking.png",
            )[0]
        )
    else:
        log("[viz] no parsed thinking draws; skipping thinking-accuracy chart")
    return written


def _per_axis_plots(cells: dict, model: str, plots_dir: Path) -> list[Path]:
    """Recolour the answer-position PCA projection by EVERY per-sample axis.

    One ``pca_by_<axis>.png`` per axis plus a single small-multiples grid; all
    share the same projection so the cloud is identical and only the colouring
    changes. Falls back to any present position if "answer" is missing.
    """
    panel = cells.get(_AXIS_PANEL_POSITION) or next(
        (cells[p] for p in _POSITIONS if p in cells), None
    )
    if panel is None:
        log("[viz] no projection cell available; skipping per-axis panels")
        return []
    written = [
        plot_pca_by_axis(panel, axis, pretty, model, plots_dir / f"pca_by_{axis}.png")
        for axis, pretty in _AXES
    ]
    written.append(plot_axes_grid(panel, model, plots_dir / "pca_axes_grid.png"))
    return written


def _geometry_plots(results: dict, model: str, plots_dir: Path) -> list[Path]:
    """Render the PCA scatters, per-axis panels, centroid-shift bars, EV% context."""
    written: list[Path] = []
    scatter_layer = _SCATTER_LAYER if _SCATTER_LAYER in results else next(iter(results))
    cells = results[scatter_layer]
    for ptype in _POSITIONS:
        block = cells.get(ptype)
        if block is None:
            log(f"[viz] no projection for position '{ptype}'; skipping its scatter")
            continue
        written.append(
            plot_pca_scatter(block, ptype, model, plots_dir / f"pca_scatter_{ptype}.png")
        )
    written += _per_axis_plots(cells, model, plots_dir)
    written.append(
        plot_centroid_shift_bars(results, model, plots_dir / "centroid_shift_by_position.png")
    )
    written.append(
        plot_explained_variance(results, model, plots_dir / "explained_variance.png")
    )
    return written


def main() -> None:
    """Load samples + the PCA projections and render every plot."""
    args = parse_args()
    log_header("VISUALIZE SESGO GEOMETRY")
    _apply_style()

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
            "PCA geometry plots (only the behavioural bars were written)")

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
