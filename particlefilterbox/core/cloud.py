"""ParticleCloud - Container for particles and their weights."""

from __future__ import annotations

import copy

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.utils.log_ops import (
    ess_from_weights,
    log_sum_exp,
    normalize_log_weights,
)


class ParticleCloud:
    """Container for a set of weighted particles.

    Stores N particles in k-dimensional state space with log-weights
    for numerical stability. Provides methods for weight manipulation,
    resampling, and weighted statistics.

    Parameters
    ----------
    n_particles : int
        Number of particles N.
    k_states : int
        Dimension of the state space k.

    Attributes
    ----------
    particles : ndarray, shape (N, k_states)
        Particle positions.
    log_weights : ndarray, shape (N,)
        Unnormalized log-weights.
    ancestors : ndarray, shape (N,), dtype int
        Ancestor indices after last resampling.
    """

    def __init__(self, n_particles: int, k_states: int) -> None:
        self.n_particles = n_particles
        self.k_states = k_states
        self.particles: NDArray[np.float64] = np.zeros((n_particles, k_states), dtype=np.float64)
        self._log_weights: NDArray[np.float64] = np.zeros(n_particles, dtype=np.float64)
        self.ancestors: NDArray[np.intp] = np.arange(n_particles)

        # Cached values (recomputed on weight change)
        self._normalized_weights: NDArray[np.float64] | None = None
        self._ess: float | None = None
        self._log_likelihood_increment: float | None = None

    # --- Weight properties ---

    @property
    def log_weights(self) -> NDArray[np.float64]:
        """Unnormalized log-weights."""
        return self._log_weights

    @property
    def normalized_weights(self) -> NDArray[np.float64]:
        """Normalized weights (sum to 1)."""
        if self._normalized_weights is None:
            self._normalized_weights = normalize_log_weights(self._log_weights)
        return self._normalized_weights

    @property
    def ess(self) -> float:
        """Effective Sample Size: 1 / sum(w_i^2)."""
        if self._ess is None:
            # Uniform log-weights (all equal) => ESS = N exactly
            if np.all(self._log_weights == self._log_weights[0]):
                self._ess = float(self.n_particles)
            else:
                self._ess = ess_from_weights(self.normalized_weights)
        return self._ess

    @property
    def log_likelihood_increment(self) -> float:
        """Log-likelihood contribution: log(1/N * sum exp(log_weights))."""
        if self._log_likelihood_increment is None:
            self._log_likelihood_increment = log_sum_exp(self._log_weights) - np.log(
                self.n_particles
            )
        return self._log_likelihood_increment

    def _invalidate_cache(self) -> None:
        """Invalidate cached derived quantities."""
        self._normalized_weights = None
        self._ess = None
        self._log_likelihood_increment = None

    # --- Weight manipulation ---

    def set_uniform_weights(self) -> None:
        """Set all log-weights to 0 (uniform weights)."""
        self._log_weights = np.zeros(self.n_particles, dtype=np.float64)
        self._invalidate_cache()

    def set_log_weights(self, log_w: NDArray[np.float64]) -> None:
        """Set log-weights directly.

        Parameters
        ----------
        log_w : ndarray, shape (N,)
            New log-weights.
        """
        if log_w.shape != (self.n_particles,):
            msg = f"Expected shape ({self.n_particles},), got {log_w.shape}"
            raise ValueError(msg)
        self._log_weights = log_w.astype(np.float64)
        self._invalidate_cache()

    def add_log_weights(self, log_increments: NDArray[np.float64]) -> None:
        """Add increments to log-weights: log_weights += log_increments.

        Parameters
        ----------
        log_increments : ndarray, shape (N,)
            Log-weight increments to add.
        """
        if log_increments.shape != (self.n_particles,):
            msg = f"Expected shape ({self.n_particles},), got {log_increments.shape}"
            raise ValueError(msg)
        self._log_weights = self._log_weights + log_increments
        self._invalidate_cache()

    # --- Resampling ---

    def resample(self, indices: NDArray[np.intp]) -> None:
        """Resample particles according to given indices.

        Reorders particles, updates ancestors, and resets weights to uniform.

        Parameters
        ----------
        indices : ndarray, shape (N,), dtype int
            Resampling indices.
        """
        if indices.shape != (self.n_particles,):
            msg = f"Expected {self.n_particles} indices, got {indices.shape[0]}"
            raise ValueError(msg)
        self.particles = self.particles[indices].copy()
        self.ancestors = indices.copy()
        self.set_uniform_weights()

    # --- Weighted statistics ---

    def weighted_mean(self) -> NDArray[np.float64]:
        """Compute weighted mean of particles.

        Returns
        -------
        ndarray, shape (k_states,)
            Weighted mean: sum(w_i * x_i).
        """
        w = self.normalized_weights
        return np.einsum("i,ij->j", w, self.particles)

    def weighted_cov(self) -> NDArray[np.float64]:
        """Compute weighted covariance of particles.

        Returns
        -------
        ndarray, shape (k_states, k_states)
            Weighted covariance matrix.
        """
        w = self.normalized_weights
        mean = self.weighted_mean()
        diff = self.particles - mean  # (N, k)
        return np.einsum("i,ij,ik->jk", w, diff, diff)

    def weighted_quantile(self, q: float | list[float]) -> NDArray[np.float64]:
        """Compute weighted quantile(s) of particles.

        Parameters
        ----------
        q : float or list of float
            Quantile(s) in [0, 1].

        Returns
        -------
        ndarray
            Quantile values. Shape (k_states,) for single q,
            or (len(q), k_states) for multiple q.
        """
        qs = np.atleast_1d(np.asarray(q, dtype=np.float64))
        w = self.normalized_weights
        result = np.zeros((len(qs), self.k_states), dtype=np.float64)

        for k in range(self.k_states):
            values = self.particles[:, k]
            sort_idx = np.argsort(values)
            sorted_vals = values[sort_idx]
            cumw = np.cumsum(w[sort_idx])
            for i, qi in enumerate(qs):
                idx = np.searchsorted(cumw, qi)
                idx = min(idx, len(sorted_vals) - 1)
                result[i, k] = sorted_vals[idx]

        if np.isscalar(q):
            return result[0]
        return result

    # --- Utility ---

    def clone(self) -> ParticleCloud:
        """Create a deep copy of this cloud.

        Returns
        -------
        ParticleCloud
            Independent copy.
        """
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"ParticleCloud(N={self.n_particles}, k={self.k_states}, ESS={self.ess:.1f})"
