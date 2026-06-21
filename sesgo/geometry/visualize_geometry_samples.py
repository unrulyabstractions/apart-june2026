"""Plot SESGO geometry: behavioral scaffold effect + representational shift.

Run-by-path driver. Loads a GeometryDataset (samples.json from
collect_geometry_samples.py) and the per-position residual tensors it points at,
then renders into out/sesgo/geometry/<MODEL>/plots/:

  BEHAVIORAL - non-thinking (and thinking) abstention accuracy by scaffold. On
    ambiguous SESGO items the gold is UNKNOWN, so accuracy = fraction predicted
    UNKNOWN; the no-scaffold baseline anchors the with/without comparison.
  GEOMETRY  - per question_id, the L2 distance between the no-scaffold residual
    and each scaffold's residual (per layer, then mean over layers) at the
    "answer" and "think_close" positions: "how far does each scaffold move the
    representation". Plotted as mean shift by scaffold, and by layer.

Robust to missing positions / subsampled data: questions without a baseline (or
without a given position) are simply skipped in the geometry aggregation.

Usage:
  uv run python sesgo/geometry/visualize_geometry_samples.py \
      out/sesgo/geometry/Qwen3-0.6B/samples.json
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
import torch  # noqa: E402

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.math import l2_distance  # noqa: E402
from src.datasets.sesgo_eval import GeometryDataset, GeometrySample  # noqa: E402

_BASELINE = "(baseline)"
_GEOM_POSITIONS = ("answer", "think_close")  # positions we measure shift at


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for geometry visualization."""
    parser = argparse.ArgumentParser(
        description="Plot SESGO geometry (behavioral + representational shift)"
    )
    parser.add_argument("samples", type=Path, help="samples.json (a GeometryDataset)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"), help="Base output directory"
    )
    return parser.parse_args()


def _scaffold_label(scaffold_id: str | None) -> str:
    """Human label for a scaffold condition; None == the no-scaffold baseline."""
    return scaffold_id or _BASELINE


def _ordered_scaffolds(dataset: GeometryDataset) -> list[str]:
    """Scaffold labels with the baseline first, the rest sorted after it."""
    labels = {_scaffold_label(s.scaffold_id) for s in dataset.samples}
    rest = sorted(labels - {_BASELINE})
    return ([_BASELINE] if _BASELINE in labels else []) + rest


def _accuracy(flags: list[bool]) -> float:
    """Fraction of True flags (== predicted UNKNOWN); 0.0 when empty."""
    return sum(flags) / len(flags) if flags else 0.0


# ── Behavioral ───────────────────────────────────────────────────────────────


def plot_accuracy_by_scaffold(
    dataset: GeometryDataset, level: str, out_path: Path
) -> tuple[Path, dict[str, float]]:
    """Bar chart of abstention accuracy by scaffold; also returns the rates."""
    scaffolds = _ordered_scaffolds(dataset)
    flags: dict[str, list[bool]] = defaultdict(list)
    for s in dataset.samples:
        if level == "non_thinking" and s.predicted_non_thinking is not None:
            flags[_scaffold_label(s.scaffold_id)].append(s.correct_non_thinking)
        elif level == "thinking" and s.predicted_thinking is not None:
            flags[_scaffold_label(s.scaffold_id)].append(
                s.predicted_thinking.value == "unknown"
            )
    accs = {sc: _accuracy(flags.get(sc, [])) for sc in scaffolds}
    ns = [len(flags.get(sc, [])) for sc in scaffolds]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(scaffolds)), [accs[sc] for sc in scaffolds], color="#30638e")
    for bar, sc, n in zip(bars, scaffolds, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{accs[sc]:.0%}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(scaffolds)))
    ax.set_xticklabels(scaffolds, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy (fraction predicted UNKNOWN)")
    pretty = level.replace("_", "-")
    ax.set_title(f"SESGO {pretty} abstention accuracy by scaffold ({dataset.model_name})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path, accs


# ── Geometry ─────────────────────────────────────────────────────────────────


def _load_tensor(root: Path, sample: GeometrySample, ptype: str) -> np.ndarray | None:
    """Load the [n_layers, d_model] residual for one sample's position type."""
    for a in sample.activations:
        if a.position_type == ptype:
            return torch.load(root / a.path, map_location="cpu").numpy()
    return None


def compute_shifts(
    dataset: GeometryDataset, root: Path, ptype: str
) -> tuple[dict[str, list[float]], dict[str, np.ndarray]]:
    """Per-(question, scaffold) L2 distance from the no-scaffold baseline.

    For each question_id we take its no-scaffold residual as the reference and
    measure each scaffold's residual against it: per-layer L2 distance, then the
    mean over layers is the scalar "shift". Returns mean-shift lists per scaffold
    plus a per-scaffold mean over the per-layer distance curve (for the by-layer
    plot). Questions lacking a baseline or the position are skipped.
    """
    # question_id -> scaffold_label -> [n_layers, d_model]
    by_q: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for s in dataset.samples:
        t = _load_tensor(root, s, ptype)
        if t is not None:
            by_q[s.question_id][_scaffold_label(s.scaffold_id)] = t

    scalar: dict[str, list[float]] = defaultdict(list)  # scaffold -> mean-shift list
    layer_curves: dict[str, list[np.ndarray]] = defaultdict(list)  # per-layer dists
    for scaffs in by_q.values():
        base = scaffs.get(_BASELINE)
        if base is None:
            continue  # no reference to measure against
        for label, vec in scaffs.items():
            if label == _BASELINE:
                continue
            # Per-layer L2 distance (reuse src.common.math.l2_distance).
            per_layer = np.array(
                [l2_distance(base[L].tolist(), vec[L].tolist()) for L in range(len(base))]
            )
            scalar[label].append(float(per_layer.mean()))
            layer_curves[label].append(per_layer)
    layer_mean = {
        label: np.mean(np.stack(curves), axis=0) for label, curves in layer_curves.items()
    }
    return scalar, layer_mean


def plot_shift_by_scaffold(
    scalar: dict[str, list[float]], ptype: str, model: str, out_path: Path
) -> tuple[Path, dict[str, float]]:
    """Bar chart of mean per-question activation-shift by scaffold."""
    labels = sorted(scalar)
    means = {lab: float(np.mean(scalar[lab])) if scalar[lab] else 0.0 for lab in labels}
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(labels)), [means[la] for la in labels], color="#d1495b")
    for bar, lab in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{means[lab]:.2f}\n(n={len(scalar[lab])})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(f"mean L2 shift from baseline ({ptype} residual)")
    ax.set_title(f"SESGO activation shift by scaffold @ {ptype} ({model})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path, means


def plot_shift_by_layer(
    layer_mean: dict[str, np.ndarray], ptype: str, model: str, out_path: Path
) -> Path:
    """Line plot of mean activation-shift per layer, one line per scaffold."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in sorted(layer_mean):
        curve = layer_mean[label]
        ax.plot(range(len(curve)), curve, marker="o", markersize=3, label=label)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"mean L2 shift from baseline ({ptype})")
    ax.set_title(f"SESGO activation shift by layer @ {ptype} ({model})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    """Load the GeometryDataset + tensors and render behavioral + geometry plots."""
    args = parse_args()
    log_header("VISUALIZE SESGO GEOMETRY")

    dataset = GeometryDataset.from_json(args.samples)
    root = args.samples.resolve().parent  # activation paths are relative to here
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    plots_dir = args.out_dir / "sesgo" / "geometry" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Behavioral: non-thinking always; thinking only when draws were parsed.
    nt_path, nt_acc = plot_accuracy_by_scaffold(
        dataset, "non_thinking", plots_dir / "accuracy_by_scaffold_non_thinking.png"
    )
    written.append(nt_path)
    th_acc: dict[str, float] = {}
    if any(s.predicted_thinking is not None for s in dataset.samples):
        th_path, th_acc = plot_accuracy_by_scaffold(
            dataset, "thinking", plots_dir / "accuracy_by_scaffold_thinking.png"
        )
        written.append(th_path)
    else:
        log("[viz] no parsed thinking draws; skipping thinking-accuracy chart")

    # Geometry: per position, shift-by-scaffold (+ by-layer when there's data).
    shift_means: dict[str, dict[str, float]] = {}
    for ptype in _GEOM_POSITIONS:
        scalar, layer_mean = compute_shifts(dataset, root, ptype)
        if not scalar:
            log(f"[viz] no baseline-paired '{ptype}' activations; skipping its shift plots")
            continue
        sp, means = plot_shift_by_scaffold(
            scalar, ptype, dataset.model_name, plots_dir / f"shift_by_scaffold_{ptype}.png"
        )
        written.append(sp)
        shift_means[ptype] = means
        written.append(
            plot_shift_by_layer(
                layer_mean, ptype, dataset.model_name,
                plots_dir / f"shift_by_layer_{ptype}.png",
            )
        )

    _log_stats_table(dataset, nt_acc, th_acc, shift_means)
    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


def _log_stats_table(
    dataset: GeometryDataset,
    nt_acc: dict[str, float],
    th_acc: dict[str, float],
    shift_means: dict[str, dict[str, float]],
) -> None:
    """Per-scaffold accuracy + mean activation-shift, one row per scaffold."""
    log_section("per-scaffold stats")
    header = f"  {'scaffold':<28} {'nt-acc':>7} {'th-acc':>7}"
    for ptype in shift_means:
        header += f" {ptype + '-shift':>16}"
    log(header)
    for sc in _ordered_scaffolds(dataset):
        nt = f"{nt_acc.get(sc, 0.0):.1%}"
        th = f"{th_acc.get(sc, 0.0):.1%}" if th_acc else "n/a"
        row = f"  {sc:<28} {nt:>7} {th:>7}"
        for ptype in shift_means:
            val = shift_means[ptype].get(sc)
            row += f" {('%.3f' % val) if val is not None else 'n/a':>16}"
        log(row)


if __name__ == "__main__":
    main()
