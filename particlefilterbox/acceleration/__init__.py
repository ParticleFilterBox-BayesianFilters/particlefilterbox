"""Acceleration backends for particle filters.

This module provides JIT compilation (Numba), GPU backends (CuPy/JAX),
parallel execution, and adaptive particle count.

Numba JIT:
    - enable_numba: Activate Numba JIT acceleration
    - disable_numba: Deactivate Numba JIT acceleration
    - is_numba_available: Check if Numba is installed
    - is_numba_enabled: Check if Numba is currently active

GPU:
    - GPUBackend: GPU-accelerated array operations (CuPy/JAX)

Parallel:
    - ParallelRunner: Multiprocessing for chains and filters

Adaptive:
    - AdaptiveN: Dynamic particle count adjustment
"""

from particlefilterbox.acceleration.adaptive_n import AdaptiveN
from particlefilterbox.acceleration.gpu import GPUBackend
from particlefilterbox.acceleration.numba_kernels import (
    disable_numba,
    enable_numba,
    is_numba_available,
    is_numba_enabled,
    log_sum_exp_numba,
    log_sum_exp_python,
    normalize_log_weights_numba,
    normalize_log_weights_python,
    systematic_resample_numba,
    systematic_resample_python,
    weighted_mean_numba,
    weighted_mean_python,
)
from particlefilterbox.acceleration.parallel import ParallelRunner

__all__ = [
    # Numba
    "enable_numba",
    "disable_numba",
    "is_numba_available",
    "is_numba_enabled",
    "systematic_resample_python",
    "systematic_resample_numba",
    "log_sum_exp_python",
    "log_sum_exp_numba",
    "weighted_mean_python",
    "weighted_mean_numba",
    "normalize_log_weights_python",
    "normalize_log_weights_numba",
    # GPU
    "GPUBackend",
    # Parallel
    "ParallelRunner",
    # Adaptive
    "AdaptiveN",
]
