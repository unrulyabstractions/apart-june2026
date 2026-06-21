"""PCA / projection analysis of SESGO geometry activations.

Run-by-path driver. Loads a GeometryDataset (response_samples.json from
collect_geometry_samples.py) and the per-position residual tensors it points at,
then runs a separate PCA per (layer-choice x position_type), measuring how each
scaffold condition moves the representation away from the no-scaffold baseline in
the low-dimensional projection. Writes a single frontend-consumable JSON to
out/sesgo/geometry/<MODEL>/analysis/projections.json.

Activations are captured PER (position, LAYER) — one tensor per structural token
position x mid-to-last transformer layer — so every captured layer is its own
analysis unit. For each (layer-view, position) we build the [n_valid, d_model]
matrix of that position's residual at that layer view (one row per sample; the
layer view is a single captured layer, the highest captured "last", or the
"mean" over all captured layers), fit a PCA, and report:

  PROJECTIONS    - per-sample PC1-2 / PC1-3 coordinates (+ the flat color-by
    axes, both categorical and the continuous answer-distribution signals), so a
    frontend can scatter and color points.
  SCAFFOLD_STATS - in FULL PCA space: per-scaffold centroids, the shift vector
    (and full-dim L2 magnitude) of each scaffold from the no-scaffold baseline,
    the pairwise centroid distance matrix, a scaffold silhouette score, and the
    between/within scatter-trace ratio.
  AXIS_SEPARATION- the same silhouette + between/within ratio computed for EVERY
    other categorical per-sample axis in the shared geometry_color_axes registry:
    scaffold (has-scaffold), origin (bbq), language, bias_category,
    question_polarity, context_condition (ambig vs disambig), accuracy (correct
    vs incorrect), thinking_outcome (did reasoning flip the committed answer:
    unchanged/changed/unparsable), selected_role, gold_role, readout,
    target_identity, other_identity, gold_label, label_style.

By DEFAULT (--layer all) every captured mid->last layer is its OWN PCA cell (plus
a "mean" layer-averaged cell), so depth is differentiated rather than collapsed.
A top-level LAYER_AXIS_SILHOUETTE table re-keys the per-cell silhouettes into a
(layer x axis) grid at a representative late position, so a frontend can render a
heatmap / per-axis layer sweep showing AT WHAT DEPTH each axis becomes separable.

Robust to missing positions / subsampled data: positions absent or with fewer
than MIN_SAMPLES valid rows are skipped (logged), degenerate (baseline-only)
groups simply emit empty shifts / a null silhouette, and the JSON always loads.

Usage:
  uv run python sesgo/geometry/analyze_geometry.py \
      out/sesgo/geometry/Qwen3-0.6B/response_samples.json
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import save_json  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.math import (  # noqa: E402
    bootstrap_labelled_ci,
    l2_distance,
    l2_norm,
)
from src.datasets.sesgo import origin_label  # noqa: E402
from src.datasets.sesgo_eval import GeometryDataset, GeometrySample  # noqa: E402

from sesgo.geometry.geometry_color_axes import (  # noqa: E402
    SCAFFOLD_AXIS_KEY,
    SEPARATION_AXES,
)

_BASELINE = "(baseline)"
MIN_SAMPLES = 4  # a position with fewer valid rows is skipped (PCA degenerate)
# silhouette_score is O(n^2 * d); above this sample size we estimate it on a
# seeded random subset so each call is O(cap^2) — negligible accuracy loss for a
# silhouette estimate, but it turns ~1.5h runs into minutes (see module docstring).
SILHOUETTE_SAMPLE_CAP = 1500
SILHOUETTE_SUBSAMPLE_SEED = 0  # fixed so the silhouette estimate is reproducible
# Bootstrap draws for the scaffold silhouette CI. Each draw is a full (capped)
# silhouette, so we use far fewer than the shift-magnitude CI's default 1000 —
# the CI band only needs a coarse spread, not 1000 O(cap^2) silhouettes per cell.
SILHOUETTE_CI_N_BOOT = 200
_ALL_POSITIONS = (
    "im_end",
    "newline",
    "im_start",
    "assistant",
    "think_open",
    "think_close",
    "answer_prefix",
    "label",
)
# Flat CATEGORICAL axes (besides scaffold_id, handled separately) we report
# silhouette / between-within separation for — straight from the single registry.
_SEPARATION_AXIS_KEYS = tuple(a.key for a in SEPARATION_AXES)


# ── Loaders & matrix building ─────────────────────────────────────────────────


# Bounded so a whole-dataset run can't pin every tensor in RAM at once: a large
# model captures ~1M tensors of ~5k floats (tens of GB if all cached). This budget
# comfortably holds one position's full layer set across all samples (the working
# set that lets the "mean" view reuse the int-layer loads) while LRU-evicting the
# rest, so memory stays bounded even though I/O is a negligible fraction of cost.
_TENSOR_CACHE_MAXSIZE = 200_000


@functools.lru_cache(maxsize=_TENSOR_CACHE_MAXSIZE)
def _load_tensor_cached(path: str) -> np.ndarray:
    """Load + (LRU-)cache one saved [d_model] residual tensor by its absolute path.

    A given (position, layer) tensor is read by BOTH that int-layer cell and the
    "mean" layer-view (which averages all captured layers for the position), so
    caching on the path lets the "mean" cell reuse the already-loaded int-layer
    tensors instead of re-``torch.load``-ing every file a second time. The LRU is
    thread-safe, so the joblib threading pool shares one cache across cells.
    """
    return torch.load(path, map_location="cpu").numpy()


def _load_layer_vec(root: Path, act) -> np.ndarray:
    """Load one saved single-layer [d_model] residual tensor as a numpy vector."""
    return _load_tensor_cached(str((root / act.path).resolve()))


def _captured_layers(sample: GeometrySample, ptype: str) -> list[int]:
    """The captured layer indices for this sample's position, sorted ascending."""
    return sorted(a.layer for a in sample.activations if a.position_type == ptype)


def _load_position_vec(
    root: Path, sample: GeometrySample, ptype: str, layer
) -> np.ndarray | None:
    """The [d_model] residual at one (position, layer view) for one sample.

    Tensors are stored PER (position, LAYER). ``layer`` selects the view:
    "last" (the highest captured layer), "mean" (mean over ALL captured layers
    for this position), or an int layer index (that exact captured layer). Returns
    None when the sample lacks the position (or the requested int layer).
    """
    acts = [a for a in sample.activations if a.position_type == ptype]
    if not acts:
        return None
    if layer == "mean":
        return np.mean([_load_layer_vec(root, a) for a in acts], axis=0)
    if layer == "last":
        target = max(a.layer for a in acts)
    else:
        target = int(layer)
    for a in acts:
        if a.layer == target:
            return _load_layer_vec(root, a)
    return None


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

    ``scaffold_id`` stays raw (None == baseline; downstream maps it); ``scaffold``
    is the has-scaffold binary; ``origin`` is derived from the bbq flag;
    ``accuracy`` from the non-thinking readout vs the per-condition gold;
    ``thinking_outcome`` is whether reasoning flipped the committed answer
    (pre-think argmax vs post-``</think>`` parse);
    ``selected_role``/``gold_role``/``readout`` come from the flat schema fields;
    the five answer-distribution signals are continuous colormap axes. Every
    categorical axis is a stringified verbatim field so a frontend can scatter and
    colour by any of them.
    """
    sig = s.answer_signals
    return {
        "sample_idx": s.sample_idx,
        "question_id": s.question_id,
        "scaffold_id": s.scaffold_id,  # raw, may be None
        "scaffold": "scaffold" if s.has_scaffold else "no-scaffold",
        "origin": origin_label(getattr(s, "bbq", False)),
        "language": s.language,
        "bias_category": s.bias_category,
        "question_polarity": s.question_polarity,
        "context_condition": getattr(s, "context_condition", "") or "(unknown)",
        "accuracy": _accuracy_label(s),
        # Did reasoning flip the committed answer? (pre-think argmax vs post-</think>)
        "thinking_outcome": s.thinking_outcome,
        "selected_role": s.selected_role,
        "gold_role": s.gold_role,
        "readout": s.readout,
        "target_identity": getattr(s, "target_identity", "") or "(unknown)",
        "other_identity": getattr(s, "other_identity", "") or "(unknown)",
        "gold_label": getattr(s.gold_label, "value", s.gold_label),
        "label_style": getattr(s, "label_style", "") or "(none)",
        # Continuous answer-distribution signals (colormap axes).
        "top_choice_prob": sig.top_choice_prob,
        "top_choice_logit": sig.top_choice_logit,
        "vocab_entropy": sig.vocab_entropy,
        "answer_diversity": sig.answer_diversity,
        "inv_perplexity": sig.inv_perplexity,
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
        vec = _load_position_vec(root, s, ptype, layer)
        if vec is None:
            continue
        vecs.append(vec.astype(np.float32))
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

    For n above SILHOUETTE_SAMPLE_CAP we hand sklearn a seeded ``sample_size`` so
    it scores a random subset — the silhouette is an O(n^2) statistic, so the cap
    is what keeps a whole-dataset cell (n up to several thousand) from costing
    O(n^2) per call across the ~1000 bootstrap draws (see SILHOUETTE_SAMPLE_CAP).
    """
    uniq = set(labels)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return None
    sample_size = SILHOUETTE_SAMPLE_CAP if len(labels) > SILHOUETTE_SAMPLE_CAP else None
    try:
        return float(
            silhouette_score(
                Z,
                labels,
                sample_size=sample_size,
                random_state=SILHOUETTE_SUBSAMPLE_SEED,
            )
        )
    except ValueError:
        # A subsample can land < 2 groups; sklearn raises — treat as ill-defined.
        return None


def _shift_magnitude_stat(label: str):
    """A statistic(Z, labels) -> ||centroid(label) - centroid(baseline)|| in full dim.

    Returns a closure so ``bootstrap_labelled_ci`` can resample (Z, labels) rows
    and recompute the shift each time. NaN when either group vanishes in a
    resample (so that draw is dropped from the bootstrap band).
    """

    def stat(Z: np.ndarray, labels: np.ndarray) -> float:
        base = Z[labels == _BASELINE]
        grp = Z[labels == label]
        if base.shape[0] == 0 or grp.shape[0] == 0:
            return float("nan")
        return float(np.linalg.norm(grp.mean(axis=0) - base.mean(axis=0)))

    return stat


def _silhouette_stat(Z: np.ndarray, labels: np.ndarray) -> float:
    """silhouette(Z, labels) as a bootstrap-able statistic; NaN when ill-defined."""
    val = _silhouette(Z, list(labels))
    return float("nan") if val is None else val


def _axis_separation(Z: np.ndarray, rows: list[dict]) -> dict:
    """Per-axis silhouette + between/within ratio for the non-scaffold axes."""
    block: dict[str, dict] = {}
    for axis in _SEPARATION_AXIS_KEYS:
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

    # Scaffold-axis labels (parallel to Z rows) reused for shift / silhouette CIs.
    labels_full = [_scaffold_label(r["scaffold_id"]) if axis == "scaffold_id"
                   else str(r[axis]) for r in rows]

    # Shift vectors of each non-baseline group from the baseline centroid, each
    # with a percentile-bootstrap CI on its full-dim magnitude (resampling rows)
    # so the viz can draw an honest error bar on the small-n shift.
    shifts: dict[str, dict] = {}
    base = centroids_full.get(_BASELINE)
    if base is not None:
        for lab in ordered:
            if lab == _BASELINE:
                continue
            delta = centroids_full[lab] - base
            _pt, lo, hi = bootstrap_labelled_ci(
                Z, labels_full, _shift_magnitude_stat(lab)
            )
            shifts[lab] = {
                "vec2d": _coord2d(delta),
                "vec3d": _coord3d(delta),
                "shift_magnitude": l2_norm(delta.tolist()),
                "shift_ci_low": lo,
                "shift_ci_high": hi,
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

    # Bootstrap CI on the silhouette so the viz subtitle can show its spread.
    # Each draw is a full (capped) silhouette, so we use SILHOUETTE_CI_N_BOOT
    # draws rather than the shift CI's default 1000 — a coarse band is enough.
    _sil_pt, sil_lo, sil_hi = bootstrap_labelled_ci(
        Z, labels_full, _silhouette_stat, n_boot=SILHOUETTE_CI_N_BOOT
    )

    return {
        "axis": axis,
        "centroids": centroids,
        "shifts": shifts,
        "pairwise_distances": {"labels": labels_order, "matrix": matrix},
        "silhouette": _silhouette(Z, labels),
        "silhouette_ci_low": sil_lo,
        "silhouette_ci_high": sil_hi,
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


def _all_captured_layers(dataset: GeometryDataset) -> list[int]:
    """Every distinct captured layer index across the dataset, ascending.

    The capture keys tensors per (position, LAYER) over the MIDDLE->LAST band, so
    this is the mid-to-last layer set the geometry probe actually saved. Used to
    expand ``--layer all`` so EVERY captured layer is its own PCA / silhouette unit
    (never collapsed to only a mean view).
    """
    return sorted({a.layer for s in dataset.samples for a in s.activations if a.layer >= 0})


def _resolve_layers(requested: str, dataset: GeometryDataset) -> list:
    """Resolve --layer to a layer-view list (last|mean|int, or 'all'+'mean').

    "all" expands to EVERY captured layer index (mid->last) PLUS the "mean" view,
    so the per-(layer, position) grid differentiates every depth while still
    keeping the layer-averaged cloud as one comparison cell. A comma list resolves
    each token to last|mean|<captured-layer-int> verbatim.
    """
    if requested.strip() == "all":
        return [*_all_captured_layers(dataset), "mean"]
    out: list = []
    for tok in requested.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(tok if tok in ("last", "mean") else int(tok))
    return out


def _cell_silhouette(block: dict, axis_key: str) -> float | None:
    """The silhouette for one colour-by axis out of an already-computed cell.

    The scaffold axis is stored under ``scaffold_stats``; every other categorical
    axis under ``axis_separation``. Returns None when the cell never scored it.
    """
    if axis_key == SCAFFOLD_AXIS_KEY:
        return block["scaffold_stats"].get("silhouette")
    return block.get("axis_separation", {}).get(axis_key, {}).get("silhouette")


def layer_axis_silhouette(results: dict, positions: list[str]) -> dict:
    """Flat (layer x axis) silhouette table at a representative late position.

    Re-keys the per-cell silhouettes into ``{position, layers, axes, values}`` so
    the viz can draw a layer x axis heatmap (where, at what depth, each axis
    becomes separable) WITHOUT recomputing PCA. The position is the latest
    structural one present (``label`` preferred), and only INTEGER layers are
    swept (the depth axis), with the rows ordered ascending so the heatmap reads
    shallow->deep top-to-bottom.
    """
    int_layers = sorted(L for L in results if L.lstrip("-").isdigit())
    if not int_layers:
        return {}
    pos = next((p for p in reversed(positions)
                if any(p in results[L] for L in int_layers)), None)
    if pos is None:
        return {}
    axes = [a.key for a in SEPARATION_AXES] + [SCAFFOLD_AXIS_KEY]
    values = {
        axis: [
            _cell_silhouette(results[L][pos], axis) if pos in results[L] else None
            for L in int_layers
        ]
        for axis in axes
    }
    return {"position": pos, "layers": int_layers, "axes": axes, "values": values}


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
    parser.add_argument("samples", type=Path, help="response_samples.json (a GeometryDataset)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"), help="Base output directory"
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="all",
        help="'all' (EVERY captured mid->last layer + mean) or a comma list, each "
        "of last|mean|<captured-layer-int> (default: all)",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="all",
        help="'all' or comma list of position_types (im_end|newline|im_start|"
        "assistant|think_open|think_close|answer_prefix|label)",
    )
    parser.add_argument("--n-components", type=int, default=10, help="max PCA components")
    parser.add_argument("--seed", type=int, default=42, help="PCA random_state")
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="parallel worker threads over (layer x position) cells "
        "(-1 = all cores; 1 = serial). Threads share the tensor cache.",
    )
    return parser.parse_args()


def _compute_cells(
    dataset: GeometryDataset,
    root: Path,
    layers: list,
    positions: list[str],
    n_components: int,
    seed: int,
    jobs: int,
) -> dict[str, dict]:
    """PCA every (layer-view x position) cell, optionally across worker threads.

    Each cell is independent, so we fan the ~(layers x positions) grid over a
    joblib THREADING pool: the silhouette / pairwise-distance hot path is numpy /
    sklearn C that releases the GIL, and threads (unlike processes) share the
    module-level ``_load_tensor_cached`` LRU so the "mean" view still reuses the
    int-layer tensors. Returns ``{str(layer): {position: block}}`` (empty cells
    and empty layers dropped), identical to the serial nesting it replaces.
    """
    cells = [(layer, ptype) for layer in layers for ptype in positions]
    blocks = Parallel(n_jobs=jobs, backend="threading")(
        delayed(analyze_position)(dataset, root, ptype, layer, n_components, seed)
        for layer, ptype in cells
    )
    results: dict[str, dict] = {}
    for (layer, ptype), block in zip(cells, blocks):
        if block is not None:
            results.setdefault(str(layer), {})[ptype] = block
    return results


def main() -> None:
    """Load the GeometryDataset + tensors, run per-cell PCA, write projections.json."""
    args = parse_args()
    log_header("ANALYZE SESGO GEOMETRY (PCA / PROJECTION)")

    dataset = GeometryDataset.from_json(args.samples)
    root = args.samples.resolve().parent  # activation paths are relative to here
    log(f"[analyze] loaded {len(dataset.samples)} samples (model={dataset.model_name})")

    layers = _resolve_layers(args.layer, dataset)
    positions = _resolve_positions(dataset, args.position)
    log(f"[analyze] layers={layers} positions={positions} "
        f"n_components={args.n_components} seed={args.seed} jobs={args.jobs}")

    results = _compute_cells(
        dataset, root, layers, positions, args.n_components, args.seed, args.jobs
    )

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
        # Flat (layer x axis) silhouette table for the depth-of-separation
        # heatmap + per-axis layer sweep (built from the per-cell silhouettes).
        "layer_axis_silhouette": layer_axis_silhouette(results, positions),
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
