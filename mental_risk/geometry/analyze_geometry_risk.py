"""PCA / projection analysis of mental_risk geometry activations.

Run-by-path driver (risk analogue of sesgo/geometry/analyze_geometry.py). Loads a
RiskGeometryDataset (response_samples.json) and the per-position residual tensors it points
at, then runs a separate PCA per (layer x position), measuring how each FRAMING
moves the representation relative to the canonical anchor framing. All the heavy
lifting lives in src.datasets.risk_geometry.risk_geometry_analysis so this stays a
thin orchestrator. Writes one frontend-consumable JSON to
out/mental_risk/geometry/<MODEL>/analysis/projections.json.

Usage:
  uv run python mental_risk/geometry/analyze_geometry_risk.py \
      out/mental_risk/geometry/Qwen3-0.6B/response_samples.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import save_json  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.risk_geometry import (  # noqa: E402
    RiskGeometryDataset,
    analyze_position,
)

_ALL_POSITIONS = ("turn", "think_open", "think_close", "answer")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for geometry PCA analysis."""
    parser = argparse.ArgumentParser(
        description="PCA / projection analysis of mental_risk geometry activations"
    )
    parser.add_argument("samples", type=Path, help="response_samples.json (a RiskGeometryDataset)")
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument(
        "--layer", type=str, default="last,mean",
        help="comma list, each of last|mean|<int> (default: last,mean)",
    )
    parser.add_argument(
        "--position", type=str, default="all",
        help="'all' or comma list of position_types (turn|think_open|think_close|answer)",
    )
    parser.add_argument("--n-components", type=int, default=10, help="max PCA components")
    parser.add_argument("--seed", type=int, default=42, help="PCA random_state")
    return parser.parse_args()


def _resolve_positions(dataset: RiskGeometryDataset, requested: str) -> list[str]:
    """Resolve the --position arg ('all' or comma list) to an ordered list."""
    if requested == "all":
        present = {a.position_type for s in dataset.samples for a in s.activations}
        return [p for p in _ALL_POSITIONS if p in present] or sorted(present)
    return [p.strip() for p in requested.split(",") if p.strip()]


def _resolve_layers(requested: str) -> list:
    """Resolve the --layer arg (comma list of last|mean|int) to a list."""
    out: list = []
    for tok in requested.split(","):
        tok = tok.strip()
        if tok:
            out.append(tok if tok in ("last", "mean") else int(tok))
    return out


def _log_stats_table(results: dict, layers: list, positions: list[str]) -> None:
    """Per (layer, position): n, EV%, framing silhouette + per-framing shift."""
    log_section("PCA projection stats (per layer x position)")
    for layer in layers:
        lkey = str(layer)
        for ptype in positions:
            block = results.get(lkey, {}).get(ptype)
            if block is None:
                continue
            evr = block["explained_variance_ratio"]
            fs = block["framing_stats"]
            sil = fs["silhouette"]
            log(f"  layer={lkey:<5} pos={ptype:<11} n={block['n_samples']:>3} "
                f"EV%(PC1)={evr[0] if evr else 0.0:6.1%} "
                f"sil(framing)={('%.3f' % sil) if sil is not None else '  n/a':>6} "
                f"anchor={fs['anchor']}")
            for lab in sorted(fs["shifts"]):
                log(f"      {lab:<14} shift_magnitude={fs['shifts'][lab]['shift_magnitude']:.4f}")


def main() -> None:
    """Load the dataset + tensors, run per-cell PCA, write projections.json."""
    args = parse_args()
    log_header("ANALYZE MENTAL_RISK GEOMETRY (PCA / PROJECTION)")

    dataset = RiskGeometryDataset.from_json(args.samples)
    root = args.samples.resolve().parent  # activation paths are relative to here
    log(f"[analyze] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    layers = _resolve_layers(args.layer)
    positions = _resolve_positions(dataset, args.position)
    log(f"[analyze] layers={layers} positions={positions} "
        f"n_components={args.n_components} seed={args.seed}")

    results: dict[str, dict] = {}
    for layer in layers:
        per_pos = {}
        for ptype in positions:
            block = analyze_position(dataset, root, ptype, layer, args.n_components, args.seed)
            if block is not None:
                per_pos[ptype] = block
        if per_pos:
            results[str(layer)] = per_pos

    payload = {
        "model": dataset.model, "model_name": dataset.model_name,
        "prompt_dataset_id": dataset.prompt_dataset_id,
        "params": {"layers": [str(l) for l in layers], "positions": positions,
                   "n_components": args.n_components, "seed": args.seed},
        "results": results,
    }
    _log_stats_table(results, layers, positions)

    out_path = (args.out_dir / "mental_risk" / "geometry" / dataset.model_name
                / "analysis" / "projections.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, out_path)
    log(f"\n[analyze] wrote {out_path}")


if __name__ == "__main__":
    main()
