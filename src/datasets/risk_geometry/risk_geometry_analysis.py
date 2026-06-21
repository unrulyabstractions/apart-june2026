"""PCA / projection analysis primitives for risk-geometry activations.

The reusable engine behind analyze_geometry_risk.py (risk analogue of SESGO's
in-driver analyze_geometry helpers). For a given (layer, position) it stacks the
per-sample residual into a matrix, fits a PCA, and reports per-FRAMING centroids /
shifts / pairwise distances plus framing/disorder/language separation. The
condition axis is ``framing`` (not SESGO's scaffold_id); with no no-op baseline
framing, shift vectors are anchored on the FIRST framing encountered (the
canonical reference). All geometry uses src.common.math (l2_norm / l2_distance).

Returned blocks are plain JSON-able dicts (the same shape the SESGO frontend
expects), built once here so the driver stays a thin orchestrator.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.common.math import l2_distance, l2_norm
from .risk_geometry_dataset import RiskGeometryDataset
from .risk_geometry_sample import RiskGeometrySample

MIN_SAMPLES = 4  # a position with fewer valid rows is skipped (PCA degenerate)
SEPARATION_AXES = ("framing", "disorder", "language")
# Flat per-sample fields carried into the projection JSON (the color-by axes).
_ROW_FIELDS = ("framing", "disorder", "language")


def _load_tensor(root, sample: RiskGeometrySample, ptype: str):
    """Load the [n_layers, d_model] residual for one sample's position type."""
    for a in sample.activations:
        if a.position_type == ptype:
            return torch.load(root / a.path, map_location="cpu").numpy()
    return None


def _layer_reduce(t: np.ndarray, layer) -> np.ndarray:
    """Reduce a [n_layers, d_model] residual to one [d_model] vector."""
    if layer == "last":
        return t[-1]
    if layer == "mean":
        return t.mean(axis=0)
    return t[int(layer)]


def build_matrix(dataset: RiskGeometryDataset, root, ptype: str, layer):
    """Stack the per-sample [d_model] residual at one position into a matrix."""
    vecs, rows = [], []
    for s in dataset.samples:
        t = _load_tensor(root, s, ptype)
        if t is None:
            continue
        vecs.append(_layer_reduce(t, layer).astype(np.float32))
        rows.append({
            "sample_idx": s.sample_idx, "subject_id": s.subject_id,
            "framing": s.framing, "disorder": s.disorder, "language": s.language,
            "gold_risk": s.gold_risk,
        })
    if not vecs:
        return np.empty((0, 0), dtype=np.float32), rows
    X = np.stack(vecs).astype(np.float32)
    if not np.all(np.isfinite(X)):
        raise ValueError(f"non-finite activations at position '{ptype}' layer '{layer}'")
    return X, rows


def run_pca(X: np.ndarray, n_components: int, seed: int):
    """Fit a (mean-centering) PCA on X, k clamped to min(n_components, n, d)."""
    n_samples, d_model = X.shape
    k = min(n_components, n_samples, d_model)
    pca = PCA(n_components=k, random_state=seed)
    Z = pca.fit_transform(X)
    return Z, [float(v) for v in pca.explained_variance_ratio_], k


def _coord(vec: np.ndarray, dims: int) -> list[float]:
    """First ``dims`` PCA coords (pads with 0.0 if k < dims)."""
    return [float(vec[i]) if i < vec.shape[0] else 0.0 for i in range(dims)]


def _group_indices(rows, axis: str):
    """Map each axis label to the row indices in that group."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[str(r[axis])].append(i)
    return dict(groups)


def _scatter_traces(Z, labels):
    """Trace of the between- and within-group scatter matrices."""
    grand = Z.mean(axis=0)
    between = within = 0.0
    for lab in set(labels):
        idx = [i for i, l in enumerate(labels) if l == lab]
        sub = Z[idx]
        mu = sub.mean(axis=0)
        between += len(idx) * float(np.sum((mu - grand) ** 2))
        within += float(np.sum((sub - mu) ** 2))
    return between, within


def _between_within_ratio(Z, labels):
    """trace(between) / trace(within); None if undefined."""
    if len(set(labels)) < 2:
        return None
    between, within = _scatter_traces(Z, labels)
    return between / within if within > 0.0 else None


def _silhouette(Z, labels):
    """Silhouette over the FULL PCA space; None when ill-defined."""
    uniq = set(labels)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return None
    try:
        return float(silhouette_score(Z, labels))
    except ValueError:
        return None


def _labels_for(rows, axis: str) -> list[str]:
    """The axis label per row, in row order."""
    return [str(r[axis]) for r in rows]


def framing_stats(Z, rows):
    """Centroids / shifts / pairwise distances / separation along the framing axis.

    With no baseline framing, the FIRST framing label (sorted) anchors the shift
    vectors. All magnitudes use the FULL k-dim PCA coords via src.common.math.
    """
    groups = _group_indices(rows, "framing")
    ordered = sorted(groups)
    anchor = ordered[0] if ordered else None
    centroids_full, centroids = {}, {}
    for lab in ordered:
        mu = Z[groups[lab]].mean(axis=0)
        centroids_full[lab] = mu
        centroids[lab] = {"coord2d": _coord(mu, 2), "coord3d": _coord(mu, 3),
                          "n": int(len(groups[lab]))}
    shifts = {}
    if anchor is not None:
        base = centroids_full[anchor]
        for lab in ordered:
            if lab == anchor:
                continue
            delta = centroids_full[lab] - base
            shifts[lab] = {"vec2d": _coord(delta, 2), "vec3d": _coord(delta, 3),
                           "shift_magnitude": l2_norm(delta.tolist())}
    matrix = [[l2_distance(centroids_full[a].tolist(), centroids_full[b].tolist())
               for b in ordered] for a in ordered]
    labels = _labels_for(rows, "framing")
    return {
        "axis": "framing", "anchor": anchor, "centroids": centroids, "shifts": shifts,
        "pairwise_distances": {"labels": ordered, "matrix": matrix},
        "silhouette": _silhouette(Z, labels),
        "between_within_ratio": _between_within_ratio(Z, labels),
    }


def axis_separation(Z, rows):
    """Per-axis silhouette + between/within ratio for framing/disorder/language."""
    return {
        axis: {"silhouette": _silhouette(Z, _labels_for(rows, axis)),
               "between_within_ratio": _between_within_ratio(Z, _labels_for(rows, axis))}
        for axis in SEPARATION_AXES
    }


def analyze_position(dataset, root, ptype, layer, n_components, seed):
    """Build the matrix, run PCA, and assemble the result block for one cell."""
    X, rows = build_matrix(dataset, root, ptype, layer)
    if X.shape[0] < MIN_SAMPLES:
        return None
    Z, evr, k = run_pca(X, n_components, seed)
    samples = [
        {**{f: r[f] for f in _ROW_FIELDS}, "sample_idx": r["sample_idx"],
         "gold_risk": r["gold_risk"], "coord2d": _coord(z, 2), "coord3d": _coord(z, 3)}
        for r, z in zip(rows, Z)
    ]
    return {
        "n_samples": int(X.shape[0]), "d_model": int(X.shape[1]), "n_components": int(k),
        "explained_variance_ratio": evr, "samples": samples,
        "framing_stats": framing_stats(Z, rows), "axis_separation": axis_separation(Z, rows),
    }
