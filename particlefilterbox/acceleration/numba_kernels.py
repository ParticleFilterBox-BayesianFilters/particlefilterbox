"""Numba JIT-compiled kernels for critical particle filter operations.

Provides accelerated versions of systematic resampling, log-sum-exp,
weighted mean, and log-weight normalization. Handles the case where
numba is not installed gracefully.

Reference:
    Lam, S.K., Pitrou, A. & Seibert, S. (2015). Numba: A LLVM-based
    Python JIT compiler. Proceedings of the Second Workshop on the LLVM
    Compiler Infrastructure in HPC.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Type aliases for kernel signatures
_ResampleFn = Callable[[NDArray[np.float64], float], NDArray[np.int64]]
_LseFn = Callable[[NDArray[np.float64]], float]
_WmeanFn = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
_NormFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _check_numba() -> bool:
    """Check if numba is importable."""
    try:
        importlib.import_module("numba")
        return True
    except ImportError:
        return False


_NUMBA_AVAILABLE: bool = _check_numba()


def is_numba_available() -> bool:
    """Check if numba is available.

    Returns:
        True if numba is installed and importable.
    """
    return _NUMBA_AVAILABLE


# ============================================================
# Pure Python reference implementations
# ============================================================


def systematic_resample_python(
    weights: NDArray[np.float64],
    u: float,
) -> NDArray[np.int64]:
    """Systematic resampling (pure Python).

    Parameters:
        weights: Normalized weights summing to 1, shape (N,).
        u: Uniform random number in [0, 1/N).

    Returns:
        Array of resampled indices, shape (N,).
    """
    n = len(weights)
    indices = np.zeros(n, dtype=np.int64)
    cumsum = np.cumsum(weights)
    i = 0
    for j in range(n):
        target = (j + u) / n
        while i < n - 1 and cumsum[i] < target:
            i += 1
        indices[j] = i
    return indices


def log_sum_exp_python(log_w: NDArray[np.float64]) -> float:
    """Log-sum-exp with numerical stability (pure Python).

    Computes log(sum(exp(log_w))) = max(log_w) + log(sum(exp(log_w - max(log_w))))

    Parameters:
        log_w: Log-weights, shape (N,).

    Returns:
        log(sum(exp(log_w))).
    """
    max_log_w = np.max(log_w)
    if np.isinf(max_log_w) and max_log_w < 0:
        return float("-inf")
    return float(max_log_w + np.log(np.sum(np.exp(log_w - max_log_w))))


def weighted_mean_python(
    particles: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute weighted mean of particles (pure Python).

    Parameters:
        particles: Particle values, shape (N,) or (N, D).
        weights: Normalized weights, shape (N,).

    Returns:
        Weighted mean, shape (D,) or scalar.
    """
    if particles.ndim == 1:
        return np.array([np.sum(particles * weights)], dtype=np.float64)
    return np.einsum("i,ij->j", weights, particles).astype(np.float64)


def normalize_log_weights_python(
    log_w: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize log-weights to probabilities (pure Python).

    Parameters:
        log_w: Unnormalized log-weights, shape (N,).

    Returns:
        Normalized weights summing to 1, shape (N,).
    """
    lse = log_sum_exp_python(log_w)
    return np.exp(log_w - lse).astype(np.float64)


# ============================================================
# Numba JIT-compiled implementations
# ============================================================


def _build_numba_kernels() -> tuple[_ResampleFn, _LseFn, _WmeanFn, _NormFn] | None:
    """Build Numba JIT kernels if numba is available.

    Returns:
        Tuple of (resample, lse, wmean, normalize) functions, or None.
    """
    if not _NUMBA_AVAILABLE:
        return None

    numba: Any = importlib.import_module("numba")
    _njit: Any = numba.njit
    _prange: Any = numba.prange

    @_njit(cache=True)
    def _systematic_resample_nb(  # pyright: ignore[reportUnknownParameterType]
        weights: NDArray[np.float64],
        u: float,
    ) -> NDArray[np.int64]:
        n = len(weights)
        indices = np.zeros(n, dtype=np.int64)
        cumsum = np.cumsum(weights)
        i = 0
        for j in range(n):
            target = (j + u) / n
            while i < n - 1 and cumsum[i] < target:
                i += 1
            indices[j] = i
        return indices

    @_njit(cache=True)
    def _log_sum_exp_nb(  # pyright: ignore[reportUnknownParameterType]
        log_w: NDArray[np.float64],
    ) -> float:
        max_log_w = np.max(log_w)
        if np.isinf(max_log_w) and max_log_w < 0:
            return -np.inf  # type: ignore[return-value]
        total = 0.0
        for i in range(len(log_w)):
            total += np.exp(log_w[i] - max_log_w)
        return max_log_w + np.log(total)  # type: ignore[return-value]

    @_njit(parallel=True, cache=True)
    def _weighted_mean_nb(  # pyright: ignore[reportUnknownParameterType]
        particles: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n = particles.shape[0]
        d = particles.shape[1]
        result = np.zeros(d, dtype=np.float64)
        for j in _prange(d):  # pyright: ignore[reportUnknownVariableType]
            total = 0.0
            for i in range(n):
                total += weights[i] * particles[i, j]
            result[j] = total
        return result

    @_njit(cache=True)
    def _normalize_log_weights_nb(  # pyright: ignore[reportUnknownParameterType]
        log_w: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        max_log_w = np.max(log_w)
        n = len(log_w)
        total = 0.0
        for i in range(n):
            total += np.exp(log_w[i] - max_log_w)
        lse = max_log_w + np.log(total)
        result = np.empty(n, dtype=np.float64)
        for i in range(n):
            result[i] = np.exp(log_w[i] - lse)
        return result

    resample_fn: _ResampleFn = _systematic_resample_nb  # pyright: ignore[reportAssignmentType]
    lse_fn: _LseFn = _log_sum_exp_nb  # pyright: ignore[reportAssignmentType]
    wmean_fn: _WmeanFn = _weighted_mean_nb  # pyright: ignore[reportAssignmentType]
    norm_fn: _NormFn = _normalize_log_weights_nb  # pyright: ignore[reportAssignmentType]
    return (resample_fn, lse_fn, wmean_fn, norm_fn)


_numba_kernels = _build_numba_kernels()

systematic_resample_numba: _ResampleFn
log_sum_exp_numba: _LseFn
weighted_mean_numba: _WmeanFn
normalize_log_weights_numba: _NormFn

if _numba_kernels is not None:
    systematic_resample_numba = _numba_kernels[0]
    log_sum_exp_numba = _numba_kernels[1]
    weighted_mean_numba = _numba_kernels[2]
    normalize_log_weights_numba = _numba_kernels[3]
else:
    systematic_resample_numba = systematic_resample_python
    log_sum_exp_numba = log_sum_exp_python
    weighted_mean_numba = weighted_mean_python
    normalize_log_weights_numba = normalize_log_weights_python


# ============================================================
# Enable/Disable Numba acceleration
# ============================================================

_numba_enabled = False
_original_functions: dict[str, Any] = {}


def enable_numba() -> bool:
    """Enable Numba JIT acceleration for particle filter operations.

    Monkey-patches the core particle filter functions with Numba-compiled
    versions. If numba is not installed, emits a warning and returns False.

    Returns:
        True if Numba was successfully enabled, False otherwise.
    """
    global _numba_enabled

    if not _NUMBA_AVAILABLE:
        warnings.warn(
            "Numba is not installed. Install with: pip install numba",
            UserWarning,
            stacklevel=2,
        )
        return False

    if _numba_enabled:
        return True

    # Store original functions and replace with numba versions
    try:
        from particlefilterbox import resampling as resamp_mod

        if hasattr(resamp_mod, "systematic_resample"):
            _original_functions["systematic_resample"] = resamp_mod.systematic_resample
            resamp_mod.systematic_resample = systematic_resample_numba  # type: ignore[assignment]
    except ImportError:
        pass

    _numba_enabled = True
    return True


def disable_numba() -> None:
    """Disable Numba JIT acceleration and restore Python implementations.

    Restores the original Python functions that were monkey-patched
    by enable_numba().
    """
    global _numba_enabled

    if not _numba_enabled:
        return

    # Restore original functions
    try:
        from particlefilterbox import resampling as resamp_mod

        if "systematic_resample" in _original_functions:
            resamp_mod.systematic_resample = _original_functions[  # type: ignore[assignment]
                "systematic_resample"
            ]
    except ImportError:
        pass

    _original_functions.clear()
    _numba_enabled = False


def is_numba_enabled() -> bool:
    """Check if Numba acceleration is currently enabled.

    Returns:
        True if Numba is currently active.
    """
    return _numba_enabled
