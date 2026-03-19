"""Optimal transport resampling (Reich 2013) - minimizes particle displacement."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from particlefilterbox.utils.random import get_rng


def optimal_transport_resample(
    weights: NDArray[np.float64],
    particles: NDArray[np.float64],
    rng: np.random.Generator | None = None,
    method: str = "sinkhorn",
    reg: float = 0.1,
    max_iter: int = 100,
) -> NDArray[np.float64]:
    """Optimal transport resampling.

    Computes new particles as weighted combinations of the originals,
    minimizing the transport cost (squared Euclidean distance).

    NOTE: Unlike other resampling methods, this returns NEW PARTICLES
    (not indices), since the result is convex combinations.

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Normalized weights.
    particles : ndarray, shape (N, k_states)
        Current particle positions.
    rng : np.random.Generator or None
        Random number generator (unused, for interface consistency).
    method : str
        'sinkhorn' (entropic regularization) or 'exact' (linear assignment).
    reg : float
        Regularization parameter for Sinkhorn (default: 0.1).
    max_iter : int
        Maximum iterations for Sinkhorn (default: 100).

    Returns
    -------
    ndarray, shape (N, k_states)
        New particles with uniform weights.
    """
    if rng is None:
        rng = get_rng()

    if method == "sinkhorn":
        return _sinkhorn_transport(weights, particles, reg, max_iter)
    elif method == "exact":
        return _exact_transport(weights, particles)
    else:
        msg = f"Unknown OT method: {method}. Use 'sinkhorn' or 'exact'."
        raise ValueError(msg)


def _sinkhorn_transport(
    weights: NDArray[np.float64],
    particles: NDArray[np.float64],
    reg: float,
    max_iter: int,
) -> NDArray[np.float64]:
    """Sinkhorn-based approximate optimal transport."""
    n = len(weights)
    target = np.ones(n) / n  # uniform target

    # Cost matrix: squared Euclidean distances
    cost = cdist(particles, particles, metric="sqeuclidean")

    # Sinkhorn algorithm
    k_mat = np.exp(-cost / reg)
    u = np.ones(n)

    for _ in range(max_iter):
        v = target / (k_mat.T @ u)
        u = weights / (k_mat @ v)

    # Transport plan
    transport = np.diag(u) @ k_mat @ np.diag(v)

    # New particles: each new particle is a weighted combination
    new_particles = np.zeros_like(particles)
    for j in range(n):
        col = transport[:, j]
        col_sum = np.sum(col)
        if col_sum > 0:
            new_particles[j] = (col / col_sum) @ particles

    return new_particles


def _exact_transport(
    weights: NDArray[np.float64],
    particles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Exact optimal transport via linear assignment (for small N)."""
    from scipy.optimize import linear_sum_assignment

    n = len(weights)

    # Expand particles according to integer copies
    # Approximate: use integer rounding with systematic residual
    n_copies = np.round(n * weights).astype(int)
    diff = n - np.sum(n_copies)

    # Adjust to ensure exactly N copies
    if diff > 0:
        # Add copies to particles with largest fractional parts
        frac = n * weights - n_copies
        add_idx = np.argsort(-frac)[:diff]
        n_copies[add_idx] += 1
    elif diff < 0:
        # Remove copies from particles with smallest fractional parts
        frac = n * weights - n_copies
        rem_idx = np.argsort(frac)[: abs(diff)]
        n_copies[rem_idx] -= 1
        n_copies = np.maximum(n_copies, 0)
        # Re-check
        diff2 = n - np.sum(n_copies)
        if diff2 > 0:
            add_idx = np.argsort(-weights)[:diff2]
            n_copies[add_idx] += 1

    # Build expanded set
    expanded_idx = np.repeat(np.arange(n), n_copies)[:n]
    expanded_particles = particles[expanded_idx]

    # Solve assignment: minimize total squared distance to original positions
    cost = cdist(particles, expanded_particles, metric="sqeuclidean")
    _row_ind, col_ind = linear_sum_assignment(cost)

    new_particles = expanded_particles[col_ind]
    return new_particles
