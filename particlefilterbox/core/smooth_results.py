"""Particle smoother results container."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ParticleSmootherResults:
    """Container for particle smoother output.

    Stores smoothed estimates p(x_t | y_{1:T}) computed using all observations.

    The fundamental property of smoothing is:
        Var[x_t | y_{1:T}] <= Var[x_t | y_{1:t}]

    Attributes
    ----------
    smoothed_mean : np.ndarray
        Smoothed state means, shape (T, k) where T is the number of timesteps
        and k is the state dimension.
    smoothed_cov : np.ndarray
        Smoothed state covariances, shape (T, k, k).
    smoothed_quantiles : dict[float, np.ndarray]
        Mapping from quantile level (e.g. 0.025, 0.975) to arrays of shape (T, k).
    smoothed_weights : np.ndarray
        Smoothed weights for each particle at each timestep, shape (T, N) where
        N is the number of particles. Weights are normalized (sum to 1).
    trajectories : np.ndarray | None
        Sampled smoothed trajectories, shape (M, T, k) where M is the number
        of trajectories. Only available for simulation-based smoothers (e.g. FFBSi).
    method : str
        Name of the smoothing method used (e.g. 'FFBSm', 'FFBSi').
    filter_results : Any
        Reference to the original ParticleFilterResults used for smoothing.
    computation_time_seconds : float
        Wall-clock time for the smoothing computation.
    n_particles : int
        Number of particles used in filtering.
    n_timesteps : int
        Number of timesteps.
    state_dim : int
        Dimension of the state vector.
    """

    smoothed_mean: np.ndarray
    smoothed_cov: np.ndarray
    smoothed_quantiles: dict[float, np.ndarray] = field(default_factory=dict)
    smoothed_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    trajectories: np.ndarray | None = None
    method: str = ""
    filter_results: Any = None
    computation_time_seconds: float = 0.0
    n_particles: int = 0
    n_timesteps: int = 0
    state_dim: int = 0

    def __post_init__(self) -> None:
        """Validate and set derived attributes."""
        if self.smoothed_mean.ndim == 1:
            self.smoothed_mean = self.smoothed_mean.reshape(-1, 1)
        self.n_timesteps, self.state_dim = self.smoothed_mean.shape
        if self.smoothed_weights.size > 0:
            self.n_particles = (
                self.smoothed_weights.shape[1] if self.smoothed_weights.ndim == 2 else 0
            )

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of the smoothing results.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - method: smoothing method name
            - n_timesteps: number of timesteps
            - n_particles: number of particles
            - state_dim: state dimension
            - has_trajectories: whether trajectories are available
            - n_trajectories: number of trajectories (0 if not available)
            - smoothed_mean_range: (min, max) of smoothed means
            - smoothed_std_range: (min, max) of smoothed standard deviations
            - computation_time_seconds: wall-clock time
        """
        smoothed_std = np.sqrt(
            np.array([np.diag(self.smoothed_cov[t]) for t in range(self.n_timesteps)])
        )
        n_trajectories = 0
        if self.trajectories is not None:
            n_trajectories = self.trajectories.shape[0]

        return {
            "method": self.method,
            "n_timesteps": self.n_timesteps,
            "n_particles": self.n_particles,
            "state_dim": self.state_dim,
            "has_trajectories": self.trajectories is not None,
            "n_trajectories": n_trajectories,
            "smoothed_mean_range": (
                float(np.min(self.smoothed_mean)),
                float(np.max(self.smoothed_mean)),
            ),
            "smoothed_std_range": (
                float(np.min(smoothed_std)),
                float(np.max(smoothed_std)),
            ),
            "computation_time_seconds": self.computation_time_seconds,
        }

    def functional_estimate(self, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Compute E[f(x_t) | y_{1:T}] for each timestep.

        Uses the smoothed weights to compute weighted averages of f applied
        to the particles at each timestep.

        Parameters
        ----------
        func : Callable[[np.ndarray], np.ndarray]
            Function to apply to particles. Takes array of shape (N, k) and
            returns array of shape (N,) or (N, d).

        Returns
        -------
        np.ndarray
            Functional estimates, shape (T,) or (T, d).

        Raises
        ------
        ValueError
            If smoothed_weights or filter_results particles are not available.
        """
        if self.smoothed_weights.size == 0:
            raise ValueError("Smoothed weights not available. Cannot compute functional estimate.")
        if self.filter_results is None:
            raise ValueError("Filter results reference not available. Cannot access particles.")

        particles_history = self.filter_results.particles_history
        n_steps = self.n_timesteps
        results = []

        for t in range(n_steps):
            particles_t = particles_history[t]  # (N, k)
            weights_t = self.smoothed_weights[t]  # (N,)
            f_vals = func(particles_t)  # (N,) or (N, d)

            if f_vals.ndim == 1:
                estimate = np.sum(weights_t * f_vals)
            else:
                estimate = np.sum(weights_t[:, np.newaxis] * f_vals, axis=0)
            results.append(estimate)

        return np.array(results)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert smoothed results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - t: timestep index
            - state_{i}_mean: smoothed mean for state i
            - state_{i}_std: smoothed standard deviation for state i
            - state_{i}_q{level}: quantile for state i (if available)
        """
        data: dict[str, list[float]] = {"t": list(range(self.n_timesteps))}

        for i in range(self.state_dim):
            data[f"state_{i}_mean"] = self.smoothed_mean[:, i].tolist()
            std_vals = np.sqrt(self.smoothed_cov[:, i, i])
            data[f"state_{i}_std"] = std_vals.tolist()

        for level, quantile_array in sorted(self.smoothed_quantiles.items()):
            for i in range(self.state_dim):
                col_name = f"state_{i}_q{level:.3f}"
                data[col_name] = quantile_array[:, i].tolist()

        return pd.DataFrame(data)
