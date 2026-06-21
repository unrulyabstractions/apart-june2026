"""PCA / projection analysis of SESGO geometry activations.

Run-by-path driver. Loads a GeometryDataset (samples.json from
collect_geometry_samples.py) and the per-position residual tensors it points at,
then runs a separate PCA per (layer-choice x position_type), measuring how each
scaffold condition moves the representation away from the no-scaffold baseline in
the low-dimensional projection. Writes a single frontend-consumable JSON to
out/sesgo/geometry/<MODEL>/analysis/projections.json.

For each (layer, position) we build the [n_valid, d_model] matrix of that
position's residual (one row per sample, reduced over layers), fit a PCA, and
report:

  PROJECTIONS    - per-sample PC1-2 / PC1-3 coordinates (+ the flat color-by
    axes), so a frontend can scatter and color points.
  SCAFFOLD_STATS - in FULL PCA space: per-scaffold centroids, the shift vector
    (and full-dim L2 magnitude) of each scaffold from the no-scaffold baseline,
    the pairwise centroid distance matrix, a scaffold silhouette score, and the
    between/within scatter-trace ratio.
  AXIS_SEPARATION- the same silhouette + between/within ratio computed for EVERY
    other per-sample axis: origin (bbq), language, bias_category,
    question_polarity, context_condition (ambig vs disambig), accuracy (correct vs
    incorrect), target_identity, other_identity, gold_label, label_style.

Robust to missing positions / subsampled data: positions absent or with fewer
than MIN_SAMPLES valid rows are skipped (logged), degenerate (baseline-only)
groups simply emit empty shifts / a null silhouette, and the JSON always loads.

Usage:
  uv run python sesgo/geometry/analyze_geometry.py \
      out/sesgo/geometry/Qwen3-0.6B/samples.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import save_json  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.math import l2_distance, l2_norm  # noqa: E402
from src.datasets.sesgo import origin_label  # noqa: E402
from src.datasets.sesgo_eval import GeometryDataset, GeometrySample  # noqa: E402

_BASELINE = "(baseline)"
MIN_SAMPLES = 4  # a position with fewer valid rows is skipped (PCA degenerate)
_ALL_POSITIONS = ("turn", "think_open", "think_close", "answer")
# Every per-sample colour-by axis the rows carry. "origin" is derived from the
# bbq flag (original vs BBQ-adapted); the rest are stored verbatim on the sample.
_PER_SAMPLE_AXES = (
    "scaffold_id",
    "origin",
    "language",
    "bias_category",
    "question_polarity",
    "context_condition",
    "accuracy",
    "target_identity",
    "other_identity",
    "gold_label",
    "label_style",
)
# Flat axes (besides scaffold_id, handled separately) we report separation for.
_SEPARATION_AXES = tuple(a for a in _PER_SAMPLE_AXES if a != "scaffold_id")


# ── Loaders & matrix building ─────────────────────────────────────────────────


def _load_tensor(root: Path, sample: GeometrySample, ptype: str) -> np.ndarray | None:
    """Load the [n_layers, d_model] residual for one sample's position type."""
    for a in sample.activations:
        if a.position_type == ptype:
            return torch.load(root / a.path, map_location="cpu").numpy()
    return None


def _layer_reduce(t: np.ndarray, layer) -> np.ndarray:
    """Reduce a [n_layers, d_model] residual to a single [d_model] vector.

    ``layer`` is "last" (top layer), "mean" (mean over layers), or an int index.
    """
    if layer == "last":
        return t[-1]
    if layer == "mean":
        return t.mean(axis=0)
    return t[int(layer)]


def _scaffold_label(scaffold_id: str | None) -> str:
    """Human label for a scaffold condition; None == the no-scaffold baseline."""
    return scaffold_id or _BASELINE


def _accuracy_label(s: GeometrySample) -> str:
    """Per-sample correctness vs the per-condition gold, as a colour-by string.

    "correct"/"incorrect" by the non-thinking prediction matching the gold role
    (ambiguous -> UNKNOWN, disambiguated -> the ground-truth role); "(no readout)"
    when the sample has no non-thinking prediction to score.
    """
    if s.predicted_non_thinking is None:
        return "(no readout)"
    return "correct" if s.correct_non_thinking else "incorrect"


def _sample_row(s: GeometrySample) -> dict:
    """Flatten one sample's every colour-by axis into a parallel metadata row.

    ``scaffold_id`` stays raw (None == baseline; downstream maps it); ``origin``
    is derived from the bbq flag; ``accuracy`` from the non-thinking readout vs the
    per-condition gold. Every other axis is a stringified verbatim field so a
    frontend can scatter and colour by any of them.
    """
    return {
        "sample_idx": s.sample_idx,
        "question_id": s.question_id,
        "scaffold_id": s.scaffold_id,  # raw, may be None
        "origin": origin_label(getattr(s, "bbq", False)),
        "language": s.language,
        "bias_category": s.bias_category,
        "question_polarity": s.question_polarity,
        "context_condition": getattr(s, "context_condition", "") or "(unknown)",
        "accuracy": _accuracy_label(s),
        "target_identity": getattr(s, "target_identity", "") or "(unknown)",
        "other_identity": getattr(s, "other_identity", "") or "(unknown)",
        "gold_label": getattr(s.gold_label, "value", s.gold_label),
        "label_style": getattr(s, "label_style", "") or "(none)",
    }


def build_matrix(
    dataset: GeometryDataset, root: Path, ptype: str, layer
) -> tuple[np.ndarray, list[dict]]:
    """Stack the per-sample [d_model] residual at one position into a matrix.

    Returns ``(X, rows)`` where X is [n_valid, d_model] float32 and rows is a
    parallel list of metadata dicts (raw scaffold_id, may be None). Samples
    missing the position are skipped. Raises loudly if X has any NaN/Inf.
    """
    vecs: list[np.ndarray] = []
    rows: list[dict] = []
    for idx, s in enumerate(dataset.samples):
        t = _load_tensor(root, s, ptype)
        if t is None:
            continue
        vecs.append(_layer_reduce(t, layer).astype(np.float32))
        rows.append(_sample_row(s))
    if not vecs:
        return np.empty((0, 0), dtype=np.float32), rows
    X = np.stack(vecs).astype(np.float32)
    if not np.all(np.isfinite(X)):
        raise ValueError(
            f"non-finite activations at position '{ptype}' layer '{layer}' "
            f"({np.isnan(X).sum()} NaN, {np.isinf(X).sum()} Inf)"
        )
    return X, rows


# ── PCA & condition statistics ────────────────────────────────────────────────


def run_pca(
    X: np.ndarray, n_components: int, seed: int
) -> tuple[np.ndarray, list[float], list[float], int]:
    """Fit a (mean-centering) PCA on X and return the projected coordinates.

    k is clamped to min(n_components, n_samples, d_model) so sklearn never errors
    on n_samples << d_model. sklearn always mean-centers, which is what we want;
    we do NOT double-center. Returns (Z, explained_variance_ratio, singular_values, k).
    """
    n_samples, d_model = X.shape
    k = min(n_components, n_samples, d_model)
    pca = PCA(n_components=k, random_state=seed)
    Z = pca.fit_transform(X)
    return (
        Z,
        [float(v) for v in pca.explained_variance_ratio_],
        [float(v) for v in pca.singular_values_],
        k,
    )


def _coord2d(vec: np.ndarray) -> list[float]:
    """First two PCA coords as a length-2 list (pads with 0.0 if k < 2)."""
    return [float(vec[i]) if i < vec.shape[0] else 0.0 for i in range(2)]


def _coord3d(vec: np.ndarray) -> list[float]:
    """First three PCA coords as a length-3 list (pads with 0.0 if k < 3)."""
    return [float(vec[i]) if i < vec.shape[0] else 0.0 for i in range(3)]


def _group_indices(rows: list[dict], axis: str) -> dict[str, list[int]]:
    """Map each axis label to the row indices in that group.

    For scaffold_id the None value maps to the baseline label; other axes use
    their raw value (stringified) directly.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if axis == "scaffold_id":
            label = _scaffold_label(r["scaffold_id"])
        else:
            label = str(r[axis])
        groups[label].append(i)
    return dict(groups)


def _scatter_traces(Z: np.ndarray, labels: list[str]) -> tuple[float, float]:
    """Trace of the between-group and within-group scatter matrices.

    between = sum_g n_g ||mu_g - mu||^2, within = sum_g sum_i ||z_i - mu_g||^2
    (the squared-distance forms equal the traces of the scatter matrices).
    """
    grand = Z.mean(axis=0)
    between = 0.0
    within = 0.0
    for lab in set(labels):
        idx = [i for i, l in enumerate(labels) if l == lab]
        sub = Z[idx]
        mu = sub.mean(axis=0)
        between += len(idx) * float(np.sum((mu - grand) ** 2))
        within += float(np.sum((sub - mu) ** 2))
    return between, within


def _between_within_ratio(Z: np.ndarray, labels: list[str]) -> float | None:
    """trace(between scatter) / trace(within scatter); None if undefined."""
    if len(set(labels)) < 2:
        return None
    between, within = _scatter_traces(Z, labels)
    if within <= 0.0:
        return None
    return between / within


def _silhouette(Z: np.ndarray, labels: list[str]) -> float | None:
    """Silhouette score over the FULL PCA space; None when ill-defined.

    Needs >= 2 groups and at least 2 distinct labels among >= 2 samples each is
    not strictly required by sklearn, but we guard the n_groups>=2 and
    n_samples>n_groups conditions sklearn itself enforces.
    """
    uniq = set(labels)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return None
    try:
        return float(silhouette_score(Z, labels))
    except ValueError:
        return None


def _axis_separation(Z: np.ndarray, rows: list[dict]) -> dict:
    """Per-axis silhouette + between/within ratio for the non-scaffold axes."""
    block: dict[str, dict] = {}
    for axis in _SEPARATION_AXES:
        groups = _group_indices(rows, axis)
        labels = [None] * len(rows)
        for lab, idxs in groups.items():
            for i in idxs:
                labels[i] = lab
        block[axis] = {
            "silhouette": _silhouette(Z, labels),
            "between_within_ratio": _between_within_ratio(Z, labels),
        }
    return block


def condition_stats(Z: np.ndarray, rows: list[dict], axis: str = "scaffold_id") -> dict:
    """Centroids / shifts / pairwise distances / separation in FULL PCA space.

    All geometry (shift magnitude, pairwise distance) uses the FULL k-dim PCA
    coordinates via src.common.math (l2_norm / l2_distance); only coord2d/coord3d
    are truncated for plotting. The baseline group anchors the shift vectors.
    """
    groups = _group_indices(rows, axis)
    # Stable label order: baseline first, then the rest sorted.
    rest = sorted(lab for lab in groups if lab != _BASELINE)
    ordered = ([_BASELINE] if _BASELINE in groups else []) + rest

    centroids_full: dict[str, np.ndarray] = {}
    centroids: dict[str, dict] = {}
    for lab in ordered:
        sub = Z[groups[lab]]
        mu = sub.mean(axis=0)
        centroids_full[lab] = mu
        centroids[lab] = {
            "coord2d": _coord2d(mu),
            "coord3d": _coord3d(mu),
            "n": int(sub.shape[0]),
        }

    # Shift vectors of each non-baseline group from the baseline centroid.
    shifts: dict[str, dict] = {}
    base = centroids_full.get(_BASELINE)
    if base is not None:
        for lab in ordered:
            if lab == _BASELINE:
                continue
            delta = centroids_full[lab] - base
            shifts[lab] = {
                "vec2d": _coord2d(delta),
                "vec3d": _coord3d(delta),
                "shift_magnitude": l2_norm(delta.tolist()),
            }

    # Pairwise full-dim L2 distance matrix between every pair of centroids.
    labels_order = ordered
    matrix = [
        [l2_distance(centroids_full[a].tolist(), centroids_full[b].tolist()) for b in labels_order]
        for a in labels_order
    ]

    # Scaffold-axis labels for silhouette / separation.
    labels = [None] * len(rows)
    for lab, idxs in groups.items():
        for i in idxs:
            labels[i] = lab

    return {
        "axis": axis,
        "centroids": centroids,
        "shifts": shifts,
        "pairwise_distances": {"labels": labels_order, "matrix": matrix},
        "silhouette": _silhouette(Z, labels),
        "between_within_ratio": _between_within_ratio(Z, labels),
    }


# ── Orchestration ─────────────────────────────────────────────────────────────


def analyze_position(
    dataset: GeometryDataset,
    root: Path,
    ptype: str,
    layer,
    n_components: int,
    seed: int,
) -> dict | None:
    """Build the matrix, run PCA, and assemble the result block for one cell.

    Returns None (and logs the skip) if the position is absent or has fewer than
    MIN_SAMPLES valid rows. Never raises on degenerate groups.
    """
    X, rows = build_matrix(dataset, root, ptype, layer)
    n_valid = X.shape[0]
    if n_valid < MIN_SAMPLES:
        log(f"[analyze] skip layer={layer} pos={ptype}: only {n_valid} valid sample(s)")
        return None

    Z, evr, _singular, k = run_pca(X, n_components, seed)

    # Every projected sample carries ALL per-sample axes so the viz can colour
    # by any of them; coords are appended to the verbatim metadata row.
    samples = [
        {**r, "coord2d": _coord2d(z), "coord3d": _coord3d(z)}
        for r, z in zip(rows, Z)
    ]

    return {
        "n_samples": int(n_valid),
        "d_model": int(X.shape[1]),
        "n_components": int(k),
        "explained_variance_ratio": evr,
        "samples": samples,
        "scaffold_stats": condition_stats(Z, rows, axis="scaffold_id"),
        "axis_separation": _axis_separation(Z, rows),
    }


def _resolve_positions(dataset: GeometryDataset, requested: str) -> list[str]:
    """Resolve the --position arg ("all" or comma list) to an ordered list."""
    if requested == "all":
        present = {a.position_type for s in dataset.samples for a in s.activations}
        return [p for p in _ALL_POSITIONS if p in present] or sorted(present)
    return [p.strip() for p in requested.split(",") if p.strip()]


def _resolve_layers(requested: str) -> list:
    """Resolve the --layer arg (comma list of last|mean|int) to a list."""
    out: list = []
    for tok in requested.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(tok if tok in ("last", "mean") else int(tok))
    return out


# ── Stats table & main ────────────────────────────────────────────────────────


def _log_stats_table(results: dict, layers: list, positions: list[str]) -> None:
    """Per (layer, position): n, EV%, silhouette, btw/within, per-scaffold shift."""
    log_section("PCA projection stats (per layer x position)")
    for layer in layers:
        lkey = str(layer)
        if lkey not in results:
            continue
        for ptype in positions:
            block = results[lkey].get(ptype)
            if block is None:
                continue
            evr = block["explained_variance_ratio"]
            ev1 = evr[0] if evr else 0.0
            cum3 = sum(evr[:3])
            ss = block["scaffold_stats"]
            sil = ss["silhouette"]
            bw = ss["between_within_ratio"]
            log(
                f"  layer={lkey:<5} pos={ptype:<11} n={block['n_samples']:>3} "
                f"EV%(PC1)={ev1:6.1%} EV%(PC1-3)={cum3:6.1%} "
                f"sil(scaf)={('%.3f' % sil) if sil is not None else '  n/a':>6} "
                f"btw/within={('%.3f' % bw) if bw is not None else '  n/a':>6}"
            )
            shifts = ss["shifts"]
            for lab in sorted(shifts):
                log(f"      {lab:<32} shift_magnitude={shifts[lab]['shift_magnitude']:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for geometry PCA analysis."""
    parser = argparse.ArgumentParser(
        description="PCA / projection analysis of SESGO geometry activations"
    )
    parser.add_argument("samples", type=Path, help="samples.json (a GeometryDataset)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"), help="Base output directory"
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="last,mean",
        help="comma list, each of last|mean|<int> (default: last,mean)",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="all",
        help="'all' or comma list of position_types (turn|think_open|think_close|answer)",
    )
    parser.add_argument("--n-components", type=int, default=10, help="max PCA components")
    parser.add_argument("--seed", type=int, default=42, help="PCA random_state")
    return parser.parse_args()


def main() -> None:
    """Load the GeometryDataset + tensors, run per-cell PCA, write projections.json."""
    args = parse_args()
    log_header("ANALYZE SESGO GEOMETRY (PCA / PROJECTION)")

    dataset = GeometryDataset.from_json(args.samples)
    root = args.samples.resolve().parent  # activation paths are relative to here
    log(f"[analyze] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    layers = _resolve_layers(args.layer)
    positions = _resolve_positions(dataset, args.position)
    log(f"[analyze] layers={layers} positions={positions} "
        f"n_components={args.n_components} seed={args.seed}")

    results: dict[str, dict] = {}
    for layer in layers:
        per_pos: dict[str, dict] = {}
        for ptype in positions:
            block = analyze_position(
                dataset, root, ptype, layer, args.n_components, args.seed
            )
            if block is not None:
                per_pos[ptype] = block
        if per_pos:
            results[str(layer)] = per_pos

    payload = {
        "model": dataset.model,
        "model_name": dataset.model_name,
        "prompt_dataset_id": dataset.prompt_dataset_id,
        "params": {
            "layers": [str(l) for l in layers],
            "positions": positions,
            "n_components": args.n_components,
            "seed": args.seed,
        },
        "results": results,
    }

    _log_stats_table(results, layers, positions)

    out_path = (
        args.out_dir / "sesgo" / "geometry" / dataset.model_name / "analysis" / "projections.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, out_path)
    log(f"\n[analyze] wrote {out_path}")


if __name__ == "__main__":
    main()
