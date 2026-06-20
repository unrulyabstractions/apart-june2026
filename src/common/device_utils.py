"""Device and memory utilities for GPU/MPS/CPU operations."""

from __future__ import annotations

import gc
import os
import sys

from src.common.logging import log

# Track memory across iterations for leak detection
_memory_history: list[dict] = []

_torch_available = True
try:
    import torch
except ImportError:
    _torch_available = False
    torch = None  # type: ignore


def get_device() -> str:
    """Return the best available device: cuda, mps, or cpu."""
    if not _torch_available:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_memory_usage() -> dict:
    """Return current memory usage statistics for available accelerators and system RAM."""
    stats = {}

    if not _torch_available:
        # Still attempt to report system RAM below.
        pass
    else:
        # GPU memory (PyTorch)
        if torch.cuda.is_available():
            stats["cuda_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        if hasattr(torch.mps, "current_allocated_memory"):
            try:
                stats["mps_alloc_gb"] = torch.mps.current_allocated_memory() / 1e9
            except Exception:
                pass

    # MLX memory (Apple Silicon)
    try:
        import mlx.core as mx

        # Use new API (mx.get_*) if available, fall back to deprecated (mx.metal.get_*)
        if hasattr(mx, "get_active_memory"):
            stats["mlx_alloc_gb"] = mx.get_active_memory() / 1e9
        elif hasattr(mx.metal, "get_active_memory"):
            stats["mlx_alloc_gb"] = mx.metal.get_active_memory() / 1e9
        if hasattr(mx, "get_peak_memory"):
            stats["mlx_peak_gb"] = mx.get_peak_memory() / 1e9
        elif hasattr(mx.metal, "get_peak_memory"):
            stats["mlx_peak_gb"] = mx.metal.get_peak_memory() / 1e9
    except (ImportError, AttributeError):
        pass

    # System RAM (cross-platform)
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        stats["ram_gb"] = proc.memory_info().rss / 1e9
        stats["ram_percent"] = proc.memory_percent()
    except ImportError:
        pass

    return stats


def log_memory(stage: str, verbose: bool = False, iteration: int = -1) -> None:
    """Print memory usage at a given stage and optionally track history.

    Args:
        stage: Label describing where the measurement was taken.
        verbose: If True, print the memory line.
        iteration: If >= 0, record the measurement in the memory history for
            later trend/leak analysis (see ``check_memory_trend``).
    """
    mem = get_memory_usage()
    if mem:
        if verbose:
            mem_str = ", ".join(f"{k}={v:.2f}" for k, v in mem.items())
            log(f"  [Memory @ {stage}] {mem_str}")
        sys.stdout.flush()
        sys.stderr.flush()

        # Track for leak detection
        if iteration >= 0:
            _memory_history.append({"iteration": iteration, "stage": stage, **mem})


def log_mem(label: str) -> None:
    """Log current memory usage with a label (always prints)."""
    mem = get_memory_usage()
    ram = mem.get("ram_gb", 0)
    mps = mem.get("mps_alloc_gb", 0)
    cuda = mem.get("cuda_alloc_gb", 0)
    accel = mps if mps > 0 else cuda
    log(f"  [{label}] RAM: {ram:.2f}GB, Accelerator: {accel:.2f}GB")


def check_memory_trend() -> None:
    """Print memory trend analysis to detect leaks."""
    if len(_memory_history) < 2:
        return

    # Compare first and last entries
    first = _memory_history[0]
    last = _memory_history[-1]

    log("\n  [Memory Trend Analysis]")
    for key in ["ram_gb", "mps_alloc_gb", "cuda_alloc_gb", "mlx_alloc_gb"]:
        if key in first and key in last:
            delta = last[key] - first[key]
            if abs(delta) > 0.1:  # Only report if > 100MB change
                log(
                    f"    {key}: {first[key]:.2f} -> {last[key]:.2f} (delta: {delta:+.2f} GB)"
                )
    log()


def clear_gpu_memory(aggressive: bool = False) -> None:
    """Clear GPU memory caches for CUDA, MPS, and MLX.

    Args:
        aggressive: If True, run more thorough cleanup (slower but frees more memory)
    """
    # First GC pass
    gc.collect()

    if _torch_available:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if torch.backends.mps.is_available():
            # Synchronize to ensure all operations complete before clearing
            torch.mps.synchronize()
            torch.mps.empty_cache()

            if aggressive:
                # Extra cleanup for MPS memory pressure
                gc.collect(generation=0)
                gc.collect(generation=1)
                gc.collect(generation=2)
                torch.mps.synchronize()
                torch.mps.empty_cache()

    # Clear MLX memory cache
    try:
        import mlx.core as mx

        # Use new API (mx.clear_cache) if available, fall back to deprecated (mx.metal.clear_cache)
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except (ImportError, AttributeError):
        pass

    # Final GC pass
    if aggressive:
        gc.collect()


class ProgressTracker:
    """Track iteration progress with periodic memory logging and cleanup.

    Usage:
        tracker = ProgressTracker(total=100, progress_every=10, memory_every=50)
        for i, item in enumerate(items):
            tracker.step(i)  # Handles progress, memory logging, and cleanup
            process(item)
    """

    def __init__(
        self,
        total: int,
        progress_every: int = 10,
        memory_every: int = 50,
        prefix: str = "  ",
        log_memory_verbose: bool = True,
    ):
        self.total = total
        self.progress_every = progress_every
        self.memory_every = memory_every
        self.prefix = prefix
        self.log_memory_verbose = log_memory_verbose

    def step(self, i: int) -> None:
        """Called each iteration to handle progress/memory tracking."""
        iteration = i + 1

        if iteration % self.progress_every == 0:
            self._log_progress(iteration)

        if iteration % self.memory_every == 0:
            log_memory(
                f"after sample {iteration}",
                verbose=self.log_memory_verbose,
                iteration=i,
            )

    def _log_progress(self, iteration: int) -> None:
        log(f"{self.prefix}{iteration}/{self.total}", end="")
