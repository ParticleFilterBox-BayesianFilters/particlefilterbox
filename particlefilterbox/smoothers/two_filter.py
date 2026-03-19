"""Two-Filter Smoother for particle filtering.

Combines a forward particle filter with a backward information filter
to produce smoothed estimates.

References
----------
- Briers, M., Doucet, A. & Maskell, S. (2010). Smoothing algorithms for
  state-space models. Annals of the Institute of Statistical Mathematics.
- Kitagawa, G. (1996). Monte Carlo filter and smoother for non-Gaussian
  nonlinear state space models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from particlefilterbox._logging import get_logger
from particlefilterbox.core.smooth_results import ParticleSmootherResults
from particlefilterbox.smoothers.base import BaseParticleSmoother

logger = get_logger("smoothers.two_filter")


class TwoFilterSmoother(BaseParticleSmoother):
    """Two-Filter Smoother combining forward and backward particle filters.

    Estimates p(x_t | y_{1:T}) by combining:
    - Forward filter: p(x_t | y_{1:t})
    - Backward information filter: p(y_{t+1:T} | x_t)

    The smoothed distribution is proportional to the product:
        p(x_t | y_{1:T}) ~ p(x_t | y_{1:t}) * p(y_{t+1:T} | x_t)

    Parameters
    ----------
    quantiles : list[float] | None
        Quantile levels to compute.
    seed : int | None
        Random seed for the backward filter.

    Notes
    -----
    Requires the model to implement:
    - log_transition_density(x_new, x_old, t)
    - log_observation_density(y, x, t)
    - transition_sample(x_old, t, rng) (for backward filter)

    Examples
    --------
    >>> smoother = TwoFilterSmoother(seed=42)
    >>> result = smoother.smooth(filter_results, model)
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the Two-Filter Smoother.

        Parameters
        ----------
        quantiles : list[float] | None
            Quantile levels to compute.
        seed : int | None
            Random seed for backward filter.
        """
        super().__init__(quantiles=quantiles)
        self.seed = seed

    def _smooth_impl(
        self,
        filter_results: Any,
        model: Any,
        **kwargs: Any,
    ) -> ParticleSmootherResults:
        """Implement two-filter smoothing.

        Parameters
        ----------
        filter_results : ParticleFilterResults
            Results from the forward particle filter.
        model : ParticleFilterModel
            State-space model.
        **kwargs : Any
            Not used.

        Returns
        -------
        ParticleSmootherResults
            Smoothed estimates.
        """
        self._validate_model(model)

        particles_history = filter_results.particles_history
        weights_history = filter_results.weights_history
        n_steps = len(particles_history)
        n_particles = particles_history[0].shape[0]
        k = particles_history[0].shape[1] if particles_history[0].ndim > 1 else 1

        logger.info(
            "Running TwoFilterSmoother: T=%d, N=%d, k=%d",
            n_steps,
            n_particles,
            k,
        )

        # Ensure 2D particles
        for t in range(n_steps):
            if particles_history[t].ndim == 1:
                particles_history[t] = particles_history[t].reshape(-1, 1)

        # Get observations from filter results
        observations = self._get_observations(filter_results)

        # Run backward information filter
        backward_log_weights = self._backward_filter(
            particles_history=particles_history,
            observations=observations,
            model=model,
            n_steps=n_steps,
            n_particles=n_particles,
        )

        # Combine forward and backward
        smoothed_weights = self._combine(
            forward_weights=weights_history,
            backward_log_weights=backward_log_weights,
            n_steps=n_steps,
            n_particles=n_particles,
        )

        # Compute statistics
        smoothed_mean, smoothed_cov, smoothed_quantiles = self._compute_smoothed_statistics(
            particles_history, smoothed_weights
        )

        return ParticleSmootherResults(
            smoothed_mean=smoothed_mean,
            smoothed_cov=smoothed_cov,
            smoothed_quantiles=smoothed_quantiles,
            smoothed_weights=smoothed_weights,
        )

    def _backward_filter(
        self,
        particles_history: list[np.ndarray],
        observations: np.ndarray,
        model: Any,
        n_steps: int,
        n_particles: int,
    ) -> np.ndarray:
        """Run backward information filter.

        Computes backward weights representing p(y_{t+1:T} | x_t) for
        each particle at each timestep.

        Parameters
        ----------
        particles_history : list[np.ndarray]
            Forward filter particles at each timestep, each (N, k).
        observations : np.ndarray
            Observations, shape (T, m).
        model : Any
            State-space model.
        n_steps : int
            Number of timesteps.
        n_particles : int
            Number of particles.

        Returns
        -------
        np.ndarray
            Backward log-weights, shape (T, N).
        """
        backward_log_weights = np.zeros((n_steps, n_particles))

        # Last timestep: no future data, backward weight = 0 (log(1) = 0)
        backward_log_weights[n_steps - 1] = 0.0

        # Backward pass
        for t in range(n_steps - 2, -1, -1):
            particles_t = particles_history[t]  # (N, k)
            particles_tp1 = particles_history[t + 1]  # (N, k)
            y_tp1 = observations[t + 1]  # (m,)

            for i in range(n_particles):
                # p(y_{t+1:T} | x_t^(i))
                # Approximate using particles at t+1:
                # sum_j p(x_{t+1}^(j) | x_t^(i)) * p(y_{t+1} | x_{t+1}^(j))
                #        * backward_{t+1}^(j)
                x_t_i = particles_t[i : i + 1]  # (1, k)
                x_t_i_rep = np.tile(x_t_i, (n_particles, 1))  # (N, k)

                log_trans = model.log_transition_density(particles_tp1, x_t_i_rep, t)  # (N,)
                log_obs = model.log_observation_density(y_tp1, particles_tp1, t + 1)  # (N,)

                log_terms = log_trans + log_obs + backward_log_weights[t + 1]
                max_log = np.max(log_terms)
                backward_log_weights[t, i] = max_log + np.log(np.sum(np.exp(log_terms - max_log)))

        return backward_log_weights

    def _combine(
        self,
        forward_weights: list[np.ndarray],
        backward_log_weights: np.ndarray,
        n_steps: int,
        n_particles: int,
    ) -> np.ndarray:
        """Combine forward and backward weights.

        Smoothed weights:
            w_{t|T}^(i) ~ w_{t|t}^(i) * exp(backward_log_weight_t^(i))

        Parameters
        ----------
        forward_weights : list[np.ndarray]
            Forward filtered weights, each (N,).
        backward_log_weights : np.ndarray
            Backward log-weights, shape (T, N).
        n_steps : int
            Number of timesteps.
        n_particles : int
            Number of particles.

        Returns
        -------
        np.ndarray
            Normalized smoothed weights, shape (T, N).
        """
        smoothed_weights = np.zeros((n_steps, n_particles))

        for t in range(n_steps):
            log_forward = np.log(np.maximum(forward_weights[t], 1e-300))
            log_combined = log_forward + backward_log_weights[t]

            # Normalize
            max_log = np.max(log_combined)
            w = np.exp(log_combined - max_log)
            w_sum = np.sum(w)
            if w_sum > 0:
                smoothed_weights[t] = w / w_sum
            else:
                smoothed_weights[t] = np.ones(n_particles) / n_particles

        return smoothed_weights

    def _get_observations(self, filter_results: Any) -> np.ndarray:
        """Extract observations from filter results.

        Parameters
        ----------
        filter_results : Any
            Filter results (should have observations attribute).

        Returns
        -------
        np.ndarray
            Observations array, shape (T, m).
        """
        if hasattr(filter_results, "observations"):
            return filter_results.observations
        elif hasattr(filter_results, "y"):
            return filter_results.y
        else:
            raise ValueError(
                "Filter results must have 'observations' or 'y' attribute for TwoFilterSmoother."
            )

    def _validate_model(self, model: Any) -> None:
        """Validate model has required methods.

        Parameters
        ----------
        model : Any
            Model to validate.

        Raises
        ------
        AttributeError
            If model is missing required methods.
        """
        required = ["log_transition_density", "log_observation_density"]
        missing = [m for m in required if not hasattr(model, m)]
        if missing:
            raise AttributeError(
                f"Model must implement: {', '.join(missing)} for TwoFilterSmoother."
            )
