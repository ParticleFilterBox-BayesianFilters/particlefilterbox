"""Systematic resampling - low variance, O(N), default method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.utils.random import get_rng


def systematic_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Systematic resampling.

    Uses a single uniform random number to generate all N samples.
    Much lower variance than multinomial, and only O(N) complexity.

    Algorithm:
        1. Draw U ~ Uniform(0, 1/N)
        2. Compute CDF: C_i = sum_{j=1}^{i} w_j
        3. For i = 0, ..., N-1: u_i = U + i/N
        4. indices[i] = min{j : C_j >= u_i}

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Normalized weights (must sum to 1).
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    ndarray, shape (N,), dtype intp
        Resampling indices.
    """
    if rng is None:
        rng = get_rng()
    n = len(weights)
    cdf = np.cumsum(weights)
    u0 = rng.uniform(0.0, 1.0 / n)
    u = u0 + np.arange(n, dtype=np.float64) / n
    indices = np.searchsorted(cdf, u)
    # Clip to valid range (numerical edge cases)
    np.clip(indices, 0, n - 1, out=indices)
    return indices.astype(np.intp)
