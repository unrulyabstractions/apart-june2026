"""Plot the GEOMETRY RiskDataset into PNGs under .../geometry/<MODEL>/plots/.

Run-by-path driver (risk analogue of sesgo/geometry/visualize_geometry_samples.py).
SESGO plotted abstention accuracy by scaffold and L2 activation shift by
scaffold/layer; risk plots mean predicted risk by FRAMING and the L2 residual
shift of each framing from the canonical anchor framing (per position, per layer).
Plots:

  risk_by_framing_non_thinking.png - mean non-thinking risk by framing.
  shift_by_framing_<pos>.png       - mean L2 residual shift from the anchor framing
                                     per framing, for the answer / think_close pos.
  shift_by_layer_<pos>.png         - per-layer L2 shift, one line per framing.

Usage:
  uv run python mental_risk/geometry/visualize_geometry_risk.py
  uv run python mental_risk/geometry/visualize_geometry_risk.py \
      out/mental_risk/geometry/Qwen3-0.6B/response_samples.json
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
import torch  # noqa: E402

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header  # noqa: E402
from src.datasets.risk_geometry import RiskGeometryDataset, RiskGeometrySample  # noqa: E402

_SHIFT_POSITIONS = ("answer", "think_close")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for geometry visualization."""
    parser = argparse.ArgumentParser(description="Plot a geometry RiskDataset")
    parser.add_argument(
        "samples", type=Path, nargs="?",
        default=Path("out/mental_risk/geometry/Qwen3-0.6B/response_samples.json"),
        help="Path to a geometry response_samples.json (a RiskGeometryDataset)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    return parser.parse_args()


def plot_risk_by_framing(dataset: RiskGeometryDataset, out_path: Path) -> Path:
    """Bar chart of mean non-thinking predicted risk by framing."""
    by_framing: dict[str, list[float]] = defaultdict(list)
    for s in dataset.samples:
        if s.predicted_risk_non_thinking is not None:
            by_framing[s.framing].append(s.predicted_risk_non_thinking)
    framings = sorted(by_framing)
    means = [float(np.mean(by_framing[f])) for f in framings]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(framings)), 5))
    if framings:
        ax.bar(framings, means, color="#30638e", edgecolor="white")
    else:
        ax.text(0.5, 0.5, "no non-thinking readouts", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean non-thinking risk")
    ax.set_title(f"MentalRiskES risk by framing ({dataset.model_name})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _tensor(root: Path, s: RiskGeometrySample, ptype: str):
    """Load the [n_layers, d_model] residual for one sample's position, or None."""
    for a in s.activations:
        if a.position_type == ptype:
            return torch.load(root / a.path, map_location="cpu").numpy()
    return None


def compute_shifts(dataset: RiskGeometryDataset, root: Path, ptype: str):
    """Per-(subject, framing) per-layer L2 distance from the anchor framing.

    For each subject the alphabetically-first framing present is the anchor; every
    other framing's residual is measured against it (per layer). Returns
    (by_framing_meanmag, by_framing_perlayer) parallel dicts.
    """
    by_subject: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for s in dataset.samples:
        t = _tensor(root, s, ptype)
        if t is not None:
            by_subject[s.subject_id][s.framing] = t
    perlayer: dict[str, list[np.ndarray]] = defaultdict(list)
    for framings in by_subject.values():
        if len(framings) < 2:
            continue
        anchor = sorted(framings)[0]
        ref = framings[anchor]
        for framing, t in framings.items():
            if framing == anchor:
                continue
            perlayer[framing].append(np.linalg.norm(t - ref, axis=1))  # [n_layers]
    meanmag = {f: float(np.mean([v.mean() for v in vs])) for f, vs in perlayer.items()}
    perlayer_mean = {f: np.mean(np.stack(vs), axis=0) for f, vs in perlayer.items()}
    return meanmag, perlayer_mean


def plot_shift_by_framing(meanmag: dict, ptype: str, model: str, out_path: Path) -> Path:
    """Bar chart of mean L2 residual shift from the anchor framing per framing."""
    framings = sorted(meanmag)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * max(1, len(framings))), 5))
    if framings:
        ax.bar(framings, [meanmag[f] for f in framings], color="#9b7dff", edgecolor="white")
    else:
        ax.text(0.5, 0.5, "no multi-framing subjects", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel(f"mean L2 shift at '{ptype}' from anchor framing")
    ax.set_title(f"MentalRiskES residual shift by framing ({model}, {ptype})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_shift_by_layer(perlayer: dict, ptype: str, model: str, out_path: Path) -> Path:
    """Line plot of per-layer L2 residual shift, one line per framing."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if perlayer:
        for framing in sorted(perlayer):
            vec = perlayer[framing]
            ax.plot(range(len(vec)), vec, marker="o", markersize=3, label=framing)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no multi-framing subjects", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"mean L2 shift at '{ptype}' from anchor")
    ax.set_title(f"MentalRiskES per-layer residual shift ({model}, {ptype})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    """Load the geometry dataset + tensors and render the behavioral + shift plots."""
    args = parse_args()
    log_header("VISUALIZE GEOMETRY RISK")
    dataset = RiskGeometryDataset.from_json(args.samples)
    root = args.samples.resolve().parent  # activation paths are relative to here
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    plots_dir = args.out_dir / "mental_risk" / "geometry" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    written = [plot_risk_by_framing(dataset, plots_dir / "risk_by_framing_non_thinking.png")]
    for ptype in _SHIFT_POSITIONS:
        meanmag, perlayer = compute_shifts(dataset, root, ptype)
        written.append(plot_shift_by_framing(
            meanmag, ptype, dataset.model_name, plots_dir / f"shift_by_framing_{ptype}.png"))
        written.append(plot_shift_by_layer(
            perlayer, ptype, dataset.model_name, plots_dir / f"shift_by_layer_{ptype}.png"))

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
