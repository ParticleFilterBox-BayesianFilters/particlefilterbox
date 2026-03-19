"""Resampling methods for particle filters.

Available methods:
- multinomial: Classic, highest variance, O(N log N)
- systematic: Low variance, O(N), default
- stratified: Similar to systematic, more random, O(N)
- residual: Deterministic + stochastic, O(N)
- optimal_transport: Minimizes particle displacement, O(N^2)
- adaptive: Resample only when ESS is low
- killing: Survival-based elimination
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.resampling.adaptive import adaptive_resample, should_resample
from particlefilterbox.resampling.killing import killing_resample
from particlefilterbox.resampling.multinomial import multinomial_resample
from particlefilterbox.resampling.optimal_transport import optimal_transport_resample
from particlefilterbox.resampling.residual import residual_resample
from particlefilterbox.resampling.stratified import stratified_resample
from particlefilterbox.resampling.systematic import systematic_resample

_RESAMPLING_REGISTRY: dict[str, Callable[..., NDArray[np.intp]]] = {
    "systematic": systematic_resample,
    "multinomial": multinomial_resample,
    "stratified": stratified_resample,
    "residual": residual_resample,
}


def get_resampling_fn(method: str) -> Callable[..., NDArray[np.intp]]:
    """Get a resampling function by name.

    Parameters
    ----------
    method : str
        One of 'systematic', 'multinomial', 'stratified', 'residual'.

    Returns
    -------
    Callable
        Resampling function with signature (weights, rng=) -> indices.
    """
    if method not in _RESAMPLING_REGISTRY:
        msg = f"Unknown resampling method: {method!r}. Available: {list(_RESAMPLING_REGISTRY)}"
        raise ValueError(msg)
    return _RESAMPLING_REGISTRY[method]


__all__ = [
    "adaptive_resample",
    "get_resampling_fn",
    "killing_resample",
    "multinomial_resample",
    "optimal_transport_resample",
    "residual_resample",
    "should_resample",
    "stratified_resample",
    "systematic_resample",
]
