"""Adaptive resampling - resample only when ESS drops below threshold."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.resampling.systematic import systematic_resample
from particlefilterbox.utils.log_ops import ess_from_weights
from particlefilterbox.utils.random import get_rng

_METHODS = {
    "systematic": systematic_resample,
}


def should_resample(weights: NDArray[np.float64], threshold: float) -> bool:
    """Check if resampling is needed based on ESS.

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Normalized weights.
    threshold : float
        Fraction of N below which to resample (0 < threshold <= 1).

    Returns
    -------
    bool
        True if ESS < threshold * N.
    """
    n = len(weights)
    ess = ess_from_weights(weights)
    return ess < threshold * n


def adaptive_resample(
    weights: NDArray[np.float64],
    threshold: float = 0.5,
    base_method: str = "systematic",
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp] | None:
    """Adaptive resampling: resample only when ESS is low.

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Normalized weights.
    threshold : float
        Resample when ESS < threshold * N (default: 0.5).
    base_method : str
        Base resampling method (default: 'systematic').
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    ndarray or None
        Resampling indices if ESS < threshold * N, else None.
    """
    if rng is None:
        rng = get_rng()

    if not should_resample(weights, threshold):
        return None

    if base_method not in _METHODS:
        # Import on demand to avoid circular imports
        from particlefilterbox.resampling.multinomial import multinomial_resample
        from particlefilterbox.resampling.residual import residual_resample
        from particlefilterbox.resampling.stratified import stratified_resample

        _METHODS["multinomial"] = multinomial_resample
        _METHODS["stratified"] = stratified_resample
        _METHODS["residual"] = residual_resample

    resample_fn = _METHODS.get(base_method)
    if resample_fn is None:
        msg = f"Unknown base method: {base_method}"
        raise ValueError(msg)

    return resample_fn(weights, rng=rng)
