"""Killing resampling - survival-based elimination of low-weight particles."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.utils.random import get_rng


def killing_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Killing resampling.

    Each particle survives with probability min(1, N * w_i).
    Particles with w_i >= 1/N always survive.
    Dead particle slots are filled by duplicating survivors.

    Good for models with few modes.

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

    # Survival probability
    p_survive = np.minimum(1.0, n * weights)

    # Decide survival
    u = rng.uniform(0.0, 1.0, size=n)
    survivors = np.where(u < p_survive)[0]

    if len(survivors) == 0:
        # Edge case: no survivors, pick the highest weight particle
        survivors = np.array([np.argmax(weights)])

    # Fill all N slots
    indices = np.empty(n, dtype=np.intp)
    n_survivors = len(survivors)

    # First, assign survivors to their own slots
    indices[:n_survivors] = survivors

    # Fill remaining slots by sampling from survivors proportionally
    if n_survivors < n:
        n_fill = n - n_survivors
        surv_weights = weights[survivors]
        surv_weights = surv_weights / surv_weights.sum()
        fill_idx = rng.choice(survivors, size=n_fill, replace=True, p=surv_weights)
        indices[n_survivors:] = fill_idx

    # Shuffle to avoid ordering bias
    rng.shuffle(indices)

    return indices
