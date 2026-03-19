"""Fixed-Lag Smoother for online particle smoothing.

Estimates p(x_{t-L} | y_{1:t}) using ancestor tracing with a fixed delay L.

References
----------
- Doucet, A. & Johansen, A.M. (2009). A tutorial on particle filtering and
  smoothing: Fifteen years later.
- Kitagawa, G. (1996). Monte Carlo filter and smoother for non-Gaussian
  nonlinear state space models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from particlefilterbox._logging import get_logger
from particlefilterbox.core.smooth_results import ParticleSmootherResults
from particlefilterbox.smoothers.base import BaseParticleSmoother

logger = get_logger("smoothers.fixed_lag")


class FixedLagSmoother(BaseParticleSmoother):
    """Fixed-Lag Smoother using ancestor tracing.

    Estimates p(x_{t-L} | y_{1:t}) by tracing particle ancestors L steps
    back in time. This provides online smoothed estimates with a fixed delay.

    Properties:
    - lag=0: equivalent to filtering (no smoothing)
    - lag=T: approximates fixed-interval smoothing
    - Intermediate lag: trade-off between latency and estimation quality

    Parameters
    ----------
    lag : int
        Number of steps to look back. Must be >= 0.
    quantiles : list[float] | None
        Quantile levels to compute.

    Examples
    --------
    >>> smoother = FixedLagSmoother(lag=10)
    >>> result = smoother.smooth(filter_results, model)
    >>> # result.smoothed_mean[t] estimates E[x_t | y_{1:t+L}]
    """

    def __init__(
        self,
        lag: int = 5,
        quantiles: list[float] | None = None,
    ) -> None:
        """Initialize the Fixed-Lag Smoother.

        Parameters
        ----------
        lag : int
            Number of steps to look back. Must be >= 0.
        quantiles : list[float] | None
            Quantile levels to compute.

        Raises
        ------
        ValueError
            If lag is negative.
        """
        super().__init__(quantiles=quantiles)
        if lag < 0:
            raise ValueError(f"Lag must be non-negative, got {lag}")
        self.lag = lag

    def _smooth_impl(
        self,
        filter_results: Any,
        model: Any,
        **kwargs: Any,
    ) -> ParticleSmootherResults:
        """Implement fixed-lag smoothing using ancestor tracing.

        Parameters
        ----------
        filter_results : ParticleFilterResults
            Results from a particle filter with stored particles and weights.
        model : ParticleFilterModel
            State-space model (not directly used, but kept for interface).
        **kwargs : Any
            Not used.

        Returns
        -------
        ParticleSmootherResults
            Smoothed estimates.
        """
        particles_history = filter_results.particles_history
        weights_history = filter_results.weights_history
        n_steps = len(particles_history)
        n_particles = particles_history[0].shape[0]
        k = particles_history[0].shape[1] if particles_history[0].ndim > 1 else 1

        lag = min(self.lag, n_steps - 1)  # Cap lag at T-1

        logger.info(
            "Running FixedLagSmoother: T=%d, N=%d, k=%d, lag=%d (effective=%d)",
            n_steps,
            n_particles,
            k,
            self.lag,
            lag,
        )

        # Ensure 2D particles
        for t in range(n_steps):
            if particles_history[t].ndim == 1:
                particles_history[t] = particles_history[t].reshape(-1, 1)

        # Get ancestor indices if available, otherwise reconstruct
        ancestor_indices = self._get_or_reconstruct_ancestors(
            filter_results, particles_history, n_steps, n_particles
        )

        # Compute smoothed weights using ancestor tracing
        smoothed_weights = np.zeros((n_steps, n_particles))

        for t in range(n_steps):
            smoothed_weights[t] = self._smooth_step_internal(
                t=t,
                lag=lag,
                weights_history=weights_history,
                n_steps=n_steps,
            )

        # Compute smoothed particles using ancestor tracing
        smoothed_particles = self._trace_all_ancestors(
            particles_history=particles_history,
            ancestor_indices=ancestor_indices,
            lag=lag,
            n_steps=n_steps,
            n_particles=n_particles,
        )

        # Compute statistics from traced particles and weights
        smoothed_mean, smoothed_cov, smoothed_quantiles = self._compute_smoothed_statistics(
            smoothed_particles, smoothed_weights
        )

        return ParticleSmootherResults(
            smoothed_mean=smoothed_mean,
            smoothed_cov=smoothed_cov,
            smoothed_quantiles=smoothed_quantiles,
            smoothed_weights=smoothed_weights,
        )

    def smooth_step(
        self,
        t: int,
        particles_history: list[np.ndarray],
        weights_history: list[np.ndarray],
        ancestor_indices: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform one step of fixed-lag smoothing (for online use).

        Parameters
        ----------
        t : int
            Current timestep.
        particles_history : list[np.ndarray]
            Particles at each timestep up to t, each (N, k).
        weights_history : list[np.ndarray]
            Weights at each timestep up to t, each (N,).
        ancestor_indices : list[np.ndarray]
            Ancestor indices from resampling, each (N,).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - smoothed_mean: smoothed estimate at t-lag, shape (k,)
            - smoothed_weights: weights for the smoothed estimate, shape (N,)
        """
        n_particles = particles_history[0].shape[0]
        lag = min(self.lag, t)

        target_t = t - lag

        # Trace ancestors L steps back
        indices = np.arange(n_particles)
        for s in range(t, target_t, -1):
            if s < len(ancestor_indices):
                indices = ancestor_indices[s][indices]

        # Get traced particles at target_t
        traced_particles = particles_history[target_t][indices]  # (N, k)
        current_weights = weights_history[t]

        # Smoothed mean
        if traced_particles.ndim == 1:
            traced_particles = traced_particles.reshape(-1, 1)
        smoothed_mean = np.sum(current_weights[:, np.newaxis] * traced_particles, axis=0)

        return smoothed_mean, current_weights

    def _smooth_step_internal(
        self,
        t: int,
        lag: int,
        weights_history: list[np.ndarray],
        n_steps: int,
    ) -> np.ndarray:
        """Internal smoothing step for batch mode.

        Parameters
        ----------
        t : int
            Target timestep for smoothing.
        lag : int
            Effective lag.
        weights_history : list[np.ndarray]
            Weights at each timestep.
        n_steps : int
            Total timesteps.

        Returns
        -------
        np.ndarray
            Smoothed weights at time t, shape (N,).
        """
        # Use weights from min(t + lag, T - 1)
        future_t = min(t + lag, n_steps - 1)
        return weights_history[future_t].copy()

    def _trace_ancestors(
        self,
        t: int,
        lag: int,
        ancestor_indices: list[np.ndarray],
        n_steps: int,
        n_particles: int,
    ) -> np.ndarray:
        """Trace ancestors L steps back from time t+L to time t.

        Parameters
        ----------
        t : int
            Target timestep.
        lag : int
            Number of steps to trace back.
        ancestor_indices : list[np.ndarray]
            Ancestor indices from resampling.
        n_steps : int
            Total timesteps.
        n_particles : int
            Number of particles.

        Returns
        -------
        np.ndarray
            Traced particle indices at time t, shape (N,).
        """
        future_t = min(t + lag, n_steps - 1)

        indices = np.arange(n_particles)
        for s in range(future_t, t, -1):
            if s < len(ancestor_indices):
                indices = ancestor_indices[s][indices]
            indices = np.clip(indices, 0, n_particles - 1)

        return indices

    def _trace_all_ancestors(
        self,
        particles_history: list[np.ndarray],
        ancestor_indices: list[np.ndarray],
        lag: int,
        n_steps: int,
        n_particles: int,
    ) -> list[np.ndarray]:
        """Trace ancestors for all timesteps.

        Parameters
        ----------
        particles_history : list[np.ndarray]
            Particles at each timestep.
        ancestor_indices : list[np.ndarray]
            Ancestor indices.
        lag : int
            Lag value.
        n_steps : int
            Total timesteps.
        n_particles : int
            Number of particles.

        Returns
        -------
        list[np.ndarray]
            Traced particles at each timestep, each (N, k).
        """
        traced_particles = []
        for t in range(n_steps):
            indices = self._trace_ancestors(t, lag, ancestor_indices, n_steps, n_particles)
            traced_particles.append(particles_history[t][indices])
        return traced_particles

    def _get_or_reconstruct_ancestors(
        self,
        filter_results: Any,
        particles_history: list[np.ndarray],
        n_steps: int,
        n_particles: int,
    ) -> list[np.ndarray]:
        """Get ancestor indices from filter results or reconstruct them.

        If the filter results have ancestor_indices, use them.
        Otherwise, reconstruct approximate ancestors by finding nearest
        neighbors between consecutive particle sets.

        Parameters
        ----------
        filter_results : Any
            Filter results.
        particles_history : list[np.ndarray]
            Particles at each timestep.
        n_steps : int
            Total timesteps.
        n_particles : int
            Number of particles.

        Returns
        -------
        list[np.ndarray]
            Ancestor indices for each timestep, each (N,) of ints.
        """
        if (
            hasattr(filter_results, "ancestor_indices")
            and filter_results.ancestor_indices is not None
        ):
            return filter_results.ancestor_indices

        # Reconstruct approximate ancestors
        logger.warning(
            "No ancestor_indices found in filter results. "
            "Reconstructing approximate ancestors using nearest neighbors. "
            "For best results, use a filter that stores ancestor indices."
        )

        ancestor_indices: list[np.ndarray] = [np.arange(n_particles)]  # t=0: identity

        for t in range(1, n_steps):
            particles_prev = particles_history[t - 1]  # (N, k)
            particles_curr = particles_history[t]  # (N, k)

            # Find nearest ancestor for each current particle
            indices = np.zeros(n_particles, dtype=np.int64)
            for i in range(n_particles):
                dists = np.sum((particles_prev - particles_curr[i]) ** 2, axis=1)
                indices[i] = np.argmin(dists)

            ancestor_indices.append(indices)

        return ancestor_indices
