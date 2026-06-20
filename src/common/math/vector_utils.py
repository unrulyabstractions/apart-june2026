"""Vector math utilities.

Supports both numpy arrays and torch tensors for the array-based operations
(cosine similarity, projection, normalization, etc.), as well as pure-Python
``list[float]`` operations used by the estimation pipeline (L2 norm/distance
and orientation vectors).
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


# ── Pure-Python list[float] operations ───────────────────────────────────────
# Used by the estimation pipeline, which works with plain Python lists of floats
# (system cores). Kept separate from the array-based helpers below so callers do
# not need numpy/torch for these lightweight operations.


def l2_norm(v: list[float]) -> float:
    """Compute L2 (Euclidean) norm of a vector.

    Args:
        v: Input vector

    Returns:
        ||v||_2 = sqrt(sum(v_i^2))
    """
    return math.sqrt(sum(x * x for x in v))


def l2_distance(a: list[float], b: list[float]) -> float:
    """Compute L2 (Euclidean) distance between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        ||a - b||_2

    Raises:
        ValueError: If vectors have different lengths
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_orientation_vector(
    source_core: list[float],
    reference_core: list[float] | None,
) -> tuple[list[float], float]:
    """Compute orientation vector and its norm from source to reference.

    Orientation = source_core - reference_core (vector difference).
    The norm is the Euclidean distance (L2 norm).

    Args:
        source_core: The source core vector (e.g., branch core)
        reference_core: The reference core vector (e.g., trunk core).
            If None or empty, returns empty vector and zero norm.

    Returns:
        Tuple of (orientation_vector, orientation_norm).
        If reference_core is None or either core is empty, returns ([], 0.0).
    """
    if reference_core is None or not source_core or not reference_core:
        return [], 0.0

    if len(source_core) != len(reference_core):
        raise ValueError(
            f"Core dimension mismatch: source has {len(source_core)}, "
            f"reference has {len(reference_core)}"
        )

    orientation = [source_core[i] - reference_core[i] for i in range(len(source_core))]
    norm = math.sqrt(sum(v * v for v in orientation))

    return orientation, norm


# ── Array-based helpers (numpy / torch) ──────────────────────────────────────


def _is_torch_tensor(x: ArrayLike) -> bool:
    """Check if x is a torch tensor without importing torch at module level."""
    return type(x).__module__.startswith("torch")


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert to numpy array if torch tensor."""
    if _is_torch_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _get_norm(x: ArrayLike) -> float:
    """Compute L2 norm."""
    if _is_torch_tensor(x):
        import torch
        return float(torch.linalg.norm(x))
    return float(np.linalg.norm(x))


def _get_dot(a: ArrayLike, b: ArrayLike) -> float:
    """Compute dot product."""
    if _is_torch_tensor(a) or _is_torch_tensor(b):
        import torch
        if not _is_torch_tensor(a):
            a = torch.from_numpy(a)
        if not _is_torch_tensor(b):
            b = torch.from_numpy(b)
        return float(torch.dot(a.flatten(), b.flatten()))
    return float(np.dot(a.flatten(), b.flatten()))


def cosine_similarity(a: ArrayLike, b: ArrayLike, eps: float = 1e-8) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (numpy array or torch tensor)
        b: Second vector (numpy array or torch tensor)
        eps: Small value to avoid division by zero

    Returns:
        Cosine similarity in range [-1, 1], or 0.0 if either vector has near-zero norm
    """
    norm_a = _get_norm(a)
    norm_b = _get_norm(b)
    if norm_a < eps or norm_b < eps:
        return 0.0
    return _get_dot(a, b) / (norm_a * norm_b)


def cosine_distance(a: ArrayLike, b: ArrayLike, eps: float = 1e-8) -> float:
    """Compute cosine distance between two vectors.

    Args:
        a: First vector
        b: Second vector
        eps: Small value to avoid division by zero

    Returns:
        Cosine distance in range [0, 2], where 0 = identical, 1 = orthogonal, 2 = opposite
    """
    return 1.0 - cosine_similarity(a, b, eps)


def angle_between(a: ArrayLike, b: ArrayLike, eps: float = 1e-8) -> float:
    """Compute angle in degrees between two vectors.

    Args:
        a: First vector
        b: Second vector
        eps: Small value to avoid division by zero

    Returns:
        Angle in degrees in range [0, 180]
    """
    cos_sim = cosine_similarity(a, b, eps)
    # Clamp to avoid numerical issues with arccos
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_sim)))


def normalize(x: ArrayLike, eps: float = 1e-8) -> ArrayLike:
    """Normalize vector to unit length.

    Args:
        x: Vector to normalize
        eps: Small value to avoid division by zero

    Returns:
        Normalized vector (same type as input), or zeros if input has near-zero norm
    """
    norm = _get_norm(x)
    if norm < eps:
        if _is_torch_tensor(x):
            import torch
            return torch.zeros_like(x)
        return np.zeros_like(x)
    return x / norm


def project_onto(v: ArrayLike, direction: ArrayLike, eps: float = 1e-8) -> ArrayLike:
    """Project vector v onto direction.

    Args:
        v: Vector to project
        direction: Direction to project onto (will be normalized)
        eps: Small value to avoid division by zero

    Returns:
        Projection of v onto direction (same type as v)
    """
    norm_dir = _get_norm(direction)
    if norm_dir < eps:
        if _is_torch_tensor(v):
            import torch
            return torch.zeros_like(v)
        return np.zeros_like(v)

    # Normalize direction
    unit_dir = direction / norm_dir

    # Project: (v · d̂) * d̂
    dot = _get_dot(v, unit_dir)

    if _is_torch_tensor(v):
        import torch
        if not _is_torch_tensor(unit_dir):
            unit_dir = torch.from_numpy(unit_dir).to(v.device)
        return dot * unit_dir
    return dot * _to_numpy(unit_dir)


def reject_from(v: ArrayLike, direction: ArrayLike, eps: float = 1e-8) -> ArrayLike:
    """Compute component of v orthogonal to direction.

    Args:
        v: Vector
        direction: Direction to reject from
        eps: Small value to avoid division by zero

    Returns:
        Component of v orthogonal to direction (same type as v)
    """
    projection = project_onto(v, direction, eps)
    if _is_torch_tensor(v):
        import torch
        if not _is_torch_tensor(projection):
            projection = torch.from_numpy(projection).to(v.device)
    return v - projection


def batch_cosine_similarity(
    a: ArrayLike,
    b: ArrayLike,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute cosine similarity for batches of vectors.

    Args:
        a: First batch of vectors [..., d]
        b: Second batch of vectors [..., d] (must broadcast with a)
        eps: Small value to avoid division by zero

    Returns:
        Array of cosine similarities with shape broadcast(a.shape[:-1], b.shape[:-1])
    """
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)

    # Compute norms along last axis
    norm_a = np.linalg.norm(a_np, axis=-1, keepdims=True)
    norm_b = np.linalg.norm(b_np, axis=-1, keepdims=True)

    # Normalize (with protection against zero norms)
    a_normalized = np.where(norm_a > eps, a_np / norm_a, 0.0)
    b_normalized = np.where(norm_b > eps, b_np / norm_b, 0.0)

    # Dot product along last axis
    return np.sum(a_normalized * b_normalized, axis=-1)


def pairwise_cosine_similarity(
    X: ArrayLike,
    Y: ArrayLike | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        X: First set of vectors [n, d]
        Y: Second set of vectors [m, d], or None to compute X vs X
        eps: Small value to avoid division by zero

    Returns:
        Similarity matrix [n, m] where entry [i,j] is cosine_similarity(X[i], Y[j])
    """
    X_np = _to_numpy(X)
    if Y is None:
        Y_np = X_np
    else:
        Y_np = _to_numpy(Y)

    # Normalize rows
    X_norm = np.linalg.norm(X_np, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y_np, axis=1, keepdims=True)

    X_normalized = np.where(X_norm > eps, X_np / X_norm, 0.0)
    Y_normalized = np.where(Y_norm > eps, Y_np / Y_norm, 0.0)

    return X_normalized @ Y_normalized.T
