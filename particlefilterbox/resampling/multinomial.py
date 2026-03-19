"""Multinomial resampling - the simplest but highest variance method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.utils.random import get_rng


def multinomial_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Multinomial resampling.

    Draws N independent samples from Categorical(weights).
    Simplest method but has the highest variance among resampling schemes.

    Complexity: O(N log N)

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Normalized weights (must sum to 1).
    rng : np.random.Generator or None
        Random number generator. If None, creates a new one.

    Returns
    -------
    ndarray, shape (N,), dtype intp
        Resampling indices.
    """
    if rng is None:
        rng = get_rng()
    n = len(weights)
    return rng.choice(n, size=n, replace=True, p=weights).astype(np.intp)
