"""Base class for particle smoothers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from particlefilterbox._logging import get_logger
from particlefilterbox.core.smooth_results import ParticleSmootherResults

logger = get_logger("smoothers.base")


class BaseParticleSmoother(ABC):
    """Abstract base class for particle smoothers.

    Particle smoothers estimate the smoothing distribution p(x_t | y_{1:T})
    using ALL available data, both past and future observations. This provides
    improved estimates compared to filtering which only uses p(x_t | y_{1:t}).

    Key property: Var[x_t | y_{1:T}] <= Var[x_t | y_{1:t}]

    All smoothers require that the particle filter was run with:
    - store_particles=True
    - store_weights=True

    Subclasses must implement the `_smooth_impl` method.
    """

    def __init__(self, quantiles: list[float] | None = None) -> None:
        """Initialize the base particle smoother.

        Parameters
        ----------
        quantiles : list[float] | None
            Quantile levels to compute (e.g. [0.025, 0.5, 0.975]).
            Defaults to [0.025, 0.25, 0.5, 0.75, 0.975].
        """
        if quantiles is None:
            self.quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
        else:
            self.quantiles = quantiles

    def smooth(
        self,
        filter_results: Any,
        model: Any,
        **kwargs: Any,
    ) -> ParticleSmootherResults:
        """Run the smoother on filter results.

        Parameters
        ----------
        filter_results : ParticleFilterResults
            Results from a particle filter run. Must have store_particles=True
            and store_weights=True.
        model : ParticleFilterModel
            The state-space model used for filtering.
        **kwargs : Any
            Additional keyword arguments passed to the specific smoother.

        Returns
        -------
        ParticleSmootherResults
            Smoothed estimates of the state.

        Raises
        ------
        ValueError
            If filter_results does not have stored particles/weights.
        """
        self._validate_filter_results(filter_results)

        logger.info("Starting %s smoother", self.__class__.__name__)
        start_time = time.perf_counter()

        result = self._smooth_impl(filter_results, model, **kwargs)

        elapsed = time.perf_counter() - start_time
        result.computation_time_seconds = elapsed
        result.method = self.__class__.__name__
        result.filter_results = filter_results

        logger.info(
            "%s smoother completed in %.3f seconds",
            self.__class__.__name__,
            elapsed,
        )

        return result

    @abstractmethod
    def _smooth_impl(
        self,
        filter_results: Any,
        model: Any,
        **kwargs: Any,
    ) -> ParticleSmootherResults:
        """Implementation of the smoothing algorithm.

        Subclasses must implement this method.

        Parameters
        ----------
        filter_results : ParticleFilterResults
            Results from a particle filter run.
        model : ParticleFilterModel
            The state-space model.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ParticleSmootherResults
            Smoothed estimates.
        """
        ...

    def _validate_filter_results(self, filter_results: Any) -> None:
        """Validate that filter results have required data for smoothing.

        Parameters
        ----------
        filter_results : ParticleFilterResults
            Results to validate.

        Raises
        ------
        ValueError
            If particles_history or weights_history is not available.
        """
        if not hasattr(filter_results, "particles_history"):
            raise ValueError(
                "Filter results must have 'particles_history'. "
                "Run the filter with store_particles=True."
            )
        if filter_results.particles_history is None:
            raise ValueError("particles_history is None. Run the filter with store_particles=True.")
        if not hasattr(filter_results, "weights_history"):
            raise ValueError(
                "Filter results must have 'weights_history'. "
                "Run the filter with store_weights=True."
            )
        if filter_results.weights_history is None:
            raise ValueError("weights_history is None. Run the filter with store_weights=True.")

        particles_history = filter_results.particles_history
        weights_history = filter_results.weights_history

        if len(particles_history) == 0:
            raise ValueError("particles_history is empty.")
        if len(weights_history) == 0:
            raise ValueError("weights_history is empty.")
        if len(particles_history) != len(weights_history):
            raise ValueError(
                f"particles_history length ({len(particles_history)}) does not match "
                f"weights_history length ({len(weights_history)})."
            )

        logger.debug(
            "Validated filter results: T=%d, N=%d, k=%d",
            len(particles_history),
            particles_history[0].shape[0],
            particles_history[0].shape[1] if particles_history[0].ndim > 1 else 1,
        )

    def _compute_smoothed_statistics(
        self,
        particles_history: list[np.ndarray],
        smoothed_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
        """Compute smoothed mean, covariance and quantiles from weights.

        Parameters
        ----------
        particles_history : list[np.ndarray]
            List of particle arrays, each of shape (N, k).
        smoothed_weights : np.ndarray
            Smoothed weights, shape (T, N). Must be normalized (sum to 1 per row).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]
            - smoothed_mean: shape (T, k)
            - smoothed_cov: shape (T, k, k)
            - smoothed_quantiles: dict mapping quantile level to array (T, k)
        """
        n_steps = len(particles_history)
        k = particles_history[0].shape[1] if particles_history[0].ndim > 1 else 1

        smoothed_mean = np.zeros((n_steps, k))
        smoothed_cov = np.zeros((n_steps, k, k))
        smoothed_quantiles: dict[float, np.ndarray] = {
            q: np.zeros((n_steps, k)) for q in self.quantiles
        }

        for t in range(n_steps):
            particles_t = particles_history[t]  # (N, k)
            if particles_t.ndim == 1:
                particles_t = particles_t.reshape(-1, 1)
            w_t = smoothed_weights[t]  # (N,)

            # Smoothed mean: E[x_t | y_{1:T}] = sum_i w_t^(i) * x_t^(i)
            mean_t = np.sum(w_t[:, np.newaxis] * particles_t, axis=0)
            smoothed_mean[t] = mean_t

            # Smoothed covariance
            diff = particles_t - mean_t[np.newaxis, :]  # (N, k)
            cov_t = np.sum(
                w_t[:, np.newaxis, np.newaxis] * (diff[:, :, np.newaxis] * diff[:, np.newaxis, :]),
                axis=0,
            )
            smoothed_cov[t] = cov_t

            # Weighted quantiles
            for q in self.quantiles:
                for j in range(k):
                    sorted_indices = np.argsort(particles_t[:, j])
                    sorted_particles = particles_t[sorted_indices, j]
                    sorted_weights = w_t[sorted_indices]
                    cumsum = np.cumsum(sorted_weights)
                    idx = np.searchsorted(cumsum, q)
                    idx = min(idx, len(sorted_particles) - 1)
                    smoothed_quantiles[q][t, j] = sorted_particles[idx]

        return smoothed_mean, smoothed_cov, smoothed_quantiles
