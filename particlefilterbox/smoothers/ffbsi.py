"""Forward Filtering Backward Simulation (FFBSi) particle smoother.

The FFBSi smoother generates smoothed trajectories by backward simulation,
with O(T*N*M) complexity where M is the number of trajectories.

References
----------
- Godsill, S.J., Doucet, A. & West, M. (2004). Monte Carlo smoothing for
  nonlinear time series. JASA, 99(465), 156-168.
- Lindsten, F. & Schon, T.B. (2013). Backward simulation methods for Monte
  Carlo statistical inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from particlefilterbox._logging import get_logger
from particlefilterbox.core.smooth_results import ParticleSmootherResults
from particlefilterbox.smoothers.base import BaseParticleSmoother

logger = get_logger("smoothers.ffbsi")


class FFBSi(BaseParticleSmoother):
    """Forward Filtering Backward Simulation (FFBSi) particle smoother.

    Generates M independent smoothed trajectories by backward simulation.
    Complexity: O(T * N * M), which is advantageous when M << N.

    Parameters
    ----------
    quantiles : list[float] | None
        Quantile levels to compute.
    seed : int | None
        Random seed for reproducibility.

    Notes
    -----
    Requires the model to implement `log_transition_density(x_new, x_old, t)`.
    All weight computations are done in log-scale for numerical stability.

    Examples
    --------
    >>> smoother = FFBSi(seed=42)
    >>> result = smoother.smooth(filter_results, model, n_trajectories=100)
    >>> print(result.trajectories.shape)  # (100, T, k)
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the FFBSi smoother.

        Parameters
        ----------
        quantiles : list[float] | None
            Quantile levels to compute.
        seed : int | None
            Random seed for reproducibility.
        """
        super().__init__(quantiles=quantiles)
        self.seed = seed

    def _smooth_impl(
        self,
        filter_results: Any,
        model: Any,
        **kwargs: Any,
    ) -> ParticleSmootherResults:
        """Implement FFBSi backward simulation.

        Parameters
        ----------
        filter_results : ParticleFilterResults
            Results from a particle filter with stored particles and weights.
        model : ParticleFilterModel
            Model with log_transition_density method.
        **kwargs : Any
            - n_trajectories (int): Number of trajectories to simulate. Default 100.

        Returns
        -------
        ParticleSmootherResults
            Smoothed estimates with trajectories.

        Raises
        ------
        AttributeError
            If model does not have log_transition_density method.
        """
        self._validate_transition_density(model)

        n_trajectories: int = kwargs.get("n_trajectories", 100)

        particles_history = filter_results.particles_history
        weights_history = filter_results.weights_history
        n_steps = len(particles_history)
        n_particles = particles_history[0].shape[0]
        k = particles_history[0].shape[1] if particles_history[0].ndim > 1 else 1

        logger.info(
            "Running FFBSi: T=%d, N=%d, k=%d, M=%d (complexity: O(T*N*M) = O(%d))",
            n_steps,
            n_particles,
            k,
            n_trajectories,
            n_steps * n_particles * n_trajectories,
        )

        # Ensure particles are 2D
        for t in range(n_steps):
            if particles_history[t].ndim == 1:
                particles_history[t] = particles_history[t].reshape(-1, 1)

        rng = np.random.default_rng(self.seed)

        # Generate M trajectories
        trajectories = np.zeros((n_trajectories, n_steps, k))

        for m in range(n_trajectories):
            if m % 20 == 0:
                logger.debug("FFBSi trajectory %d/%d", m + 1, n_trajectories)

            trajectory = self._backward_simulate_one(
                particles_history=particles_history,
                weights_history=weights_history,
                model=model,
                rng=rng,
            )
            trajectories[m] = trajectory

        # Compute smoothed statistics from trajectories
        smoothed_mean = np.mean(trajectories, axis=0)  # (n_steps, k)
        smoothed_cov = np.zeros((n_steps, k, k))
        for t in range(n_steps):
            diff = trajectories[:, t, :] - smoothed_mean[t]  # (M, k)
            smoothed_cov[t] = (diff.T @ diff) / n_trajectories

        # Compute quantiles from trajectories
        smoothed_quantiles: dict[float, np.ndarray] = {}
        for q in self.quantiles:
            smoothed_quantiles[q] = np.quantile(
                trajectories,
                q,
                axis=0,
            )  # (n_steps, k)

        # For FFBSi, we don't have explicit weights; set empty
        smoothed_weights = np.array([])

        return ParticleSmootherResults(
            smoothed_mean=smoothed_mean,
            smoothed_cov=smoothed_cov,
            smoothed_quantiles=smoothed_quantiles,
            smoothed_weights=smoothed_weights,
            trajectories=trajectories,
        )

    def _backward_simulate_one(
        self,
        particles_history: list[np.ndarray],
        weights_history: list[np.ndarray],
        model: Any,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate one backward trajectory.

        Parameters
        ----------
        particles_history : list[np.ndarray]
            Particles at each timestep, each (N, k).
        weights_history : list[np.ndarray]
            Normalized weights at each timestep, each (N,).
        model : Any
            Model with log_transition_density.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Smoothed trajectory, shape (T, k).
        """
        n_steps = len(particles_history)
        k = particles_history[0].shape[1]
        trajectory = np.zeros((n_steps, k))

        # Step 1: Sample x_T from filtered weights
        w_last = weights_history[n_steps - 1]
        w_last = np.maximum(w_last, 0)
        w_last_sum = np.sum(w_last)
        w_last = w_last / w_last_sum if w_last_sum > 0 else np.ones_like(w_last) / len(w_last)

        b_last = rng.choice(len(w_last), p=w_last)
        trajectory[n_steps - 1] = particles_history[n_steps - 1][b_last]

        # Step 2: Backward simulation
        for t in range(n_steps - 2, -1, -1):
            x_next = trajectory[t + 1]  # (k,)
            trajectory[t] = self._backward_sample(
                x_next=x_next,
                particles_t=particles_history[t],
                weights_t=weights_history[t],
                model=model,
                t=t,
                rng=rng,
            )

        return trajectory

    def _backward_sample(
        self,
        x_next: np.ndarray,
        particles_t: np.ndarray,
        weights_t: np.ndarray,
        model: Any,
        t: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample one particle at time t given x_{t+1}.

        Computes backward weights:
            v_t^(i) = w_{t|t}^(i) * p(x_{t+1} | x_t^(i))
        and samples from the normalized distribution.

        Parameters
        ----------
        x_next : np.ndarray
            Next state in the trajectory, shape (k,).
        particles_t : np.ndarray
            Particles at time t, shape (N, k).
        weights_t : np.ndarray
            Normalized filtered weights at time t, shape (N,).
        model : Any
            Model with log_transition_density.
        t : int
            Time index.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Sampled particle at time t, shape (k,).
        """
        n = particles_t.shape[0]

        # Compute backward weights in log scale
        log_bw = self._backward_weights(
            x_next=x_next,
            particles_t=particles_t,
            weights_t=weights_t,
            model=model,
            t=t,
        )

        # Normalize using log-sum-exp
        max_log_bw = np.max(log_bw)
        bw = np.exp(log_bw - max_log_bw)
        bw_sum = np.sum(bw)

        if bw_sum > 0:
            bw_normalized = bw / bw_sum
        else:
            # Fallback to uniform if all weights are zero
            bw_normalized = np.ones(n) / n
            logger.warning("All backward weights are zero at t=%d, using uniform", t)

        # Sample ancestor index
        b_t = rng.choice(n, p=bw_normalized)
        return particles_t[b_t]

    def _backward_weights(
        self,
        x_next: np.ndarray,
        particles_t: np.ndarray,
        weights_t: np.ndarray,
        model: Any,
        t: int,
    ) -> np.ndarray:
        """Compute backward weights in log scale.

        log v_t^(i) = log w_{t|t}^(i) + log p(x_{t+1} | x_t^(i))

        Parameters
        ----------
        x_next : np.ndarray
            Next state, shape (k,).
        particles_t : np.ndarray
            Particles at time t, shape (N, k).
        weights_t : np.ndarray
            Normalized filtered weights at time t, shape (N,).
        model : Any
            Model with log_transition_density.
        t : int
            Time index.

        Returns
        -------
        np.ndarray
            Log backward weights, shape (N,).
        """
        n = particles_t.shape[0]

        # Log filtered weights
        log_w = np.log(np.maximum(weights_t, 1e-300))

        # Log transition density: p(x_next | x_t^(i)) for all i
        x_next_tiled = np.tile(x_next.reshape(1, -1), (n, 1))  # (n, k)
        log_trans = model.log_transition_density(
            x_next_tiled,
            particles_t,
            t,
        )  # (n,)

        # Backward log-weights
        log_bw = log_w + log_trans
        return log_bw

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
                "for FFBSi smoother. This method should return log p(x_{t+1} | x_t)."
            )
