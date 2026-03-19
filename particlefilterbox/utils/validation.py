"""Input validation utilities for particle filtering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_weights(w: NDArray[np.float64], tol: float = 1e-6) -> None:
    """Validate that weights are a proper probability distribution.

    Parameters
    ----------
    w : ndarray
        Weights to validate.
    tol : float
        Tolerance for sum-to-one check.

    Raises
    ------
    ValueError
        If weights are invalid.
    """
    if w.ndim != 1:
        msg = f"Weights must be 1D, got shape {w.shape}"
        raise ValueError(msg)
    if np.any(w < 0):
        msg = "Weights must be non-negative"
        raise ValueError(msg)
    if abs(np.sum(w) - 1.0) > tol:
        msg = f"Weights must sum to 1, got {np.sum(w):.10f}"
        raise ValueError(msg)


def validate_particles(particles: NDArray[np.float64], k_states: int) -> None:
    """Validate particle array shape.

    Parameters
    ----------
    particles : ndarray
        Particle array, expected shape (N, k_states).
    k_states : int
        Expected state dimension.

    Raises
    ------
    ValueError
        If shape is invalid.
    """
    if particles.ndim != 2:
        msg = f"Particles must be 2D, got shape {particles.shape}"
        raise ValueError(msg)
    if particles.shape[1] != k_states:
        msg = f"Particles second dim must be {k_states}, got {particles.shape[1]}"
        raise ValueError(msg)
