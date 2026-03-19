"""Forward Filtering Backward Smoothing (FFBSm) particle smoother.

The FFBSm smoother recomputes particle weights using future information,
providing exact smoothing estimates with O(T*N^2) complexity.

References
----------
- Godsill, S.J., Doucet, A. & West, M. (2004). Monte Carlo smoothing for
  nonlinear time series. JASA, 99(465), 156-168.
- Doucet, A. & Johansen, A.M. (2009). A tutorial on particle filtering and
  smoothing: Fifteen years later.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from particlefilterbox._logging import get_logger
from particlefilterbox.core.smooth_results import ParticleSmootherResults
from particlefilterbox.smoothers.base import BaseParticleSmoother

logger = get_logger("smoothers.ffbsm")


class FFBSm(BaseParticleSmoother):
    """Forward Filtering Backward Smoothing (FFBSm) particle smoother.

    Computes exact smoothing weights by reweighting filtering particles
    using backward recursion. Complexity: O(T * N^2).

    The algorithm iterates backward from T-1 to 0, computing smoothed
    weights using the transition density p(x_{t+1} | x_t).

    Parameters
    ----------
    quantiles : list[float] | None
        Quantile levels to compute. Defaults to [0.025, 0.25, 0.5, 0.75, 0.975].

    Notes
    -----
    Requires the model to implement `log_transition_density(x_new, x_old, t)`.
    All weight computations are done in log-scale for numerical stability.

    Examples
    --------
    >>> smoother = FFBSm()
    >>> smooth_results = smoother.smooth(filter_results, model)
    >>> print(smooth_results.smoothed_mean.shape)  # (T, k)
    """

    def __init__(self, quantiles: list[float] | None = None) -> None:
        """Initialize the FFBSm smoother.

        Parameters
        ----------
        quantiles : list[float] | None
            Quantile levels to compute.
        """
        super().__init__(quantiles=quantiles)

    def _smooth_impl(
        self,
        filter_results: Any,
        model: Any,
        **kwargs: Any,
    ) -> ParticleSmootherResults:
        """Implement FFBSm backward smoothing.

        Parameters
        ----------
        filter_results : ParticleFilterResults
            Results from a particle filter with stored particles and weights.
        model : ParticleFilterModel
            Model with log_transition_density method.
        **kwargs : Any
            Not used.

        Returns
        -------
        ParticleSmootherResults
            Smoothed estimates.

        Raises
        ------
        AttributeError
            If model does not have log_transition_density method.
        """
        self._validate_transition_density(model)

        particles_history = filter_results.particles_history
        weights_history = filter_results.weights_history
        n_steps = len(particles_history)
        n_particles = particles_history[0].shape[0]
        k = particles_history[0].shape[1] if particles_history[0].ndim > 1 else 1

        logger.info(
            "Running FFBSm: T=%d, N=%d, k=%d (complexity: O(T*N^2) = O(%d))",
            n_steps,
            n_particles,
            k,
            n_steps * n_particles * n_particles,
        )

        # Ensure particles are 2D
        for t in range(n_steps):
            if particles_history[t].ndim == 1:
                particles_history[t] = particles_history[t].reshape(-1, 1)

        # Initialize smoothed log-weights
        smoothed_log_weights = np.zeros((n_steps, n_particles))

        # Last step: smoothed weights = filtered weights
        w_last = weights_history[n_steps - 1]
        w_last = np.maximum(w_last, 1e-300)  # Avoid log(0)
        smoothed_log_weights[n_steps - 1] = np.log(w_last)

        # Backward recursion
        for t in range(n_steps - 2, -1, -1):
            if t % 10 == 0:
                logger.debug("FFBSm backward step t=%d", t)

            smoothed_log_weights[t] = self._backward_step(
                particles_t=particles_history[t],
                particles_tp1=particles_history[t + 1],
                log_weights_filtered_t=np.log(np.maximum(weights_history[t], 1e-300)),
                log_weights_smoothed_tp1=smoothed_log_weights[t + 1],
                model=model,
                t=t,
            )

        # Normalize smoothed weights
        smoothed_weights = np.zeros((n_steps, n_particles))
        for t in range(n_steps):
            log_w = smoothed_log_weights[t]
            max_log_w = np.max(log_w)
            w = np.exp(log_w - max_log_w)
            smoothed_weights[t] = w / np.sum(w)

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

    def _backward_step(
        self,
        particles_t: np.ndarray,
        particles_tp1: np.ndarray,
        log_weights_filtered_t: np.ndarray,
        log_weights_smoothed_tp1: np.ndarray,
        model: Any,
        t: int,
    ) -> np.ndarray:
        """Perform one backward smoothing step.

        Computes smoothed log-weights at time t given smoothed weights at t+1.

        Parameters
        ----------
        particles_t : np.ndarray
            Particles at time t, shape (N, k).
        particles_tp1 : np.ndarray
            Particles at time t+1, shape (N, k).
        log_weights_filtered_t : np.ndarray
            Log filtered weights at time t, shape (N,).
        log_weights_smoothed_tp1 : np.ndarray
            Log smoothed weights at time t+1, shape (N,).
        model : Any
            Model with log_transition_density.
        t : int
            Time index.

        Returns
        -------
        np.ndarray
            Log smoothed weights at time t, shape (N,).
        """
        n = particles_t.shape[0]

        # Compute transition matrix: log_trans[j, i] = log p(x_{t+1}^(j) | x_t^(i))
        log_trans = self._compute_transition_matrix(
            particles_tp1, particles_t, model, t
        )  # (n, n) -- [j, i]

        # Compute denominators via logsumexp for numerical stability
        log_denom = np.zeros(n)
        for j in range(n):
            log_terms = log_weights_filtered_t + log_trans[j, :]
            max_term = np.max(log_terms)
            log_denom[j] = max_term + np.log(np.sum(np.exp(log_terms - max_term)))

        # Compute smoothed weights for each i at time t using logsumexp
        log_weights_smoothed_t = np.zeros(n)
        for i in range(n):
            log_terms = log_weights_smoothed_tp1 + log_trans[:, i] - log_denom
            max_term = np.max(log_terms)
            log_sum = max_term + np.log(np.sum(np.exp(log_terms - max_term)))
            log_weights_smoothed_t[i] = log_weights_filtered_t[i] + log_sum

        return log_weights_smoothed_t

    def _compute_transition_matrix(
        self,
        particles_new: np.ndarray,
        particles_old: np.ndarray,
        model: Any,
        t: int,
    ) -> np.ndarray:
        """Compute the transition density matrix.

        Parameters
        ----------
        particles_new : np.ndarray
            Particles at time t+1, shape (N, k).
        particles_old : np.ndarray
            Particles at time t, shape (N, k).
        model : Any
            Model with log_transition_density.
        t : int
            Time index.

        Returns
        -------
        np.ndarray
            Log transition matrix, shape (N_new, N_old) where
            entry [j, i] = log p(x_{t+1}^(j) | x_t^(i)).
        """
        n_new = particles_new.shape[0]
        n_old = particles_old.shape[0]
        log_trans = np.zeros((n_new, n_old))

        for j in range(n_new):
            # Broadcast: evaluate log p(x_{t+1}^(j) | x_t^(i)) for all i
            x_new_j = np.tile(particles_new[j : j + 1], (n_old, 1))  # (n_old, k)
            log_trans[j, :] = model.log_transition_density(x_new_j, particles_old, t)

        return log_trans

    def _validate_transition_density(self, model: Any) -> None:
        """Check that the model has log_transition_density method.

        Parameters
        ----------
        model : Any
            The model to validate.

        Raises
        ------
        AttributeError
            If model does not have log_transition_density.
        """
        if not hasattr(model, "log_transition_density"):
            raise AttributeError(
                "Model must implement 'log_transition_density(x_new, x_old, t)' "
                "for FFBSm smoother. This method should return log p(x_{t+1} | x_t)."
            )
