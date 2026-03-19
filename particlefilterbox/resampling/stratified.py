"""Stratified resampling - similar to systematic but with more randomness."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.utils.random import get_rng


def stratified_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Stratified resampling.

    Each stratum [i/N, (i+1)/N) gets one independent uniform draw.
    More random than systematic, but still lower variance than multinomial.

    Algorithm:
        1. For i = 0, ..., N-1: U_i ~ Uniform(0, 1/N)
        2. u_i = (i + U_i) / N
        3. indices[i] = min{j : CDF_j >= u_i}

    Complexity: O(N)

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
    u = (np.arange(n, dtype=np.float64) + rng.uniform(0.0, 1.0, size=n)) / n
    indices = np.searchsorted(cdf, u)
    np.clip(indices, 0, n - 1, out=indices)
    return indices.astype(np.intp)
