"""Residual resampling - deterministic + stochastic two-phase method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.resampling.systematic import systematic_resample
from particlefilterbox.utils.random import get_rng


def residual_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Residual resampling.

    Two phases:
    1. Deterministic: assign floor(N * w_i) copies of particle i.
    2. Stochastic: resample remaining slots using residual weights.

    Variance strictly lower than multinomial.
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
    nw = n * weights

    # Deterministic phase
    n_det = np.floor(nw).astype(np.intp)
    n_det_total = int(np.sum(n_det))

    # Build deterministic indices
    indices = np.repeat(np.arange(n, dtype=np.intp), n_det)

    # Stochastic phase with residual weights
    n_res = n - n_det_total
    if n_res > 0:
        w_res = nw - n_det
        w_res_sum = np.sum(w_res)
        if w_res_sum > 0:
            w_res_normalized = w_res / w_res_sum
            # Use systematic for the residual part (lower variance)
            res_indices = systematic_resample(w_res_normalized, rng)
            indices = np.concatenate([indices, res_indices])

    return indices[:n].astype(np.intp)
