"""
Bounded-state models for particle filtering.

Handles physical constraints like:
- Zero Lower Bound (ZLB) for interest rates
- Positive volatility
- General lower/upper bounds
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class BoundedStates:
    """Model with bounded state constraints.

    Parameters
    ----------
    bounds : list[tuple[float | None, float | None]]
        Bounds for each state dimension. (lower, upper).
        None means unbounded.
    base_phi : float
        AR(1) persistence. Default 0.95.
    base_sigma : float
        Innovation std. Default 0.1.
    base_mu : NDArray | None
        Mean of states. If None, zeros.
    obs_sigma : float
        Observation noise. Default 0.1.
    params : dict[str, float] | None
        Additional parameters.
    """

    def __init__(
        self,
        bounds: list[tuple[float | None, float | None]] | None = None,
        base_phi: float = 0.95,
        base_sigma: float = 0.1,
        base_mu: NDArray[np.float64] | None = None,
        obs_sigma: float = 0.1,
        params: dict[str, float] | None = None,
    ) -> None:
        if bounds is None:
            bounds = [(0.0, None)]  # Default: positive state (e.g., rate)
        self.bounds = bounds
        self.k_states = len(bounds)
        self.k_obs = self.k_states
        self.base_phi = base_phi
        self.base_sigma = base_sigma
        self.obs_sigma = obs_sigma

        if base_mu is not None:
            self.base_mu = np.asarray(base_mu)
        else:
            self.base_mu = np.zeros(self.k_states)
            for i, (lo, _) in enumerate(bounds):
                if lo is not None and lo > 0:
                    self.base_mu[i] = lo + 0.5

        self.params = params if params is not None else {
            "phi": base_phi,
            "sigma": base_sigma,
            "obs_sigma": obs_sigma,
        }
        self.param_names = ["phi", "sigma", "obs_sigma"]

    def _apply_bounds(
        self, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply bounds to state via reflection/clipping.

        Parameters
        ----------
        state : NDArray
            States, shape (n, k_states).

        Returns
        -------
        NDArray
            Bounded states.
        """
        bounded = state.copy()
        for i, (lo, hi) in enumerate(self.bounds):
            if lo is not None:
                # Reflection at lower bound
                below = bounded[:, i] < lo
                bounded[below, i] = 2 * lo - bounded[below, i]
                bounded[:, i] = np.maximum(bounded[:, i], lo)
            if hi is not None:
                above = bounded[:, i] > hi
                bounded[above, i] = 2 * hi - bounded[above, i]
                bounded[:, i] = np.minimum(bounded[:, i], hi)
        return bounded

    def _bounded_transition(
        self,
        state: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Transition with built-in bound enforcement.

        Parameters
        ----------
        state : NDArray
            Current state.
        rng : np.random.Generator
            RNG.

        Returns
        -------
        NDArray
            Next state (bounded).
        """
        n = state.shape[0]
        phi = self.params["phi"]
        sigma = self.params["sigma"]
        eta = rng.standard_normal((n, self.k_states))
        x_new = self.base_mu + phi * (state - self.base_mu) + sigma * eta
        return self._apply_bounds(x_new)

    def default_prior(self) -> dict[str, dict[str, Any]]:
        """Return default prior distributions."""
        return {
            "phi": {"distribution": "beta", "a": 20.0, "b": 1.5},
            "sigma": {"distribution": "inverse_gamma", "a": 3.0, "b": 0.05},
            "obs_sigma": {"distribution": "inverse_gamma", "a": 3.0, "b": 0.05},
        }

    def initial_state(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample initial state."""
        state = np.tile(
            self.base_mu, (n_particles, 1)
        ) + rng.standard_normal((n_particles, self.k_states)) * 0.1
        return self._apply_bounds(state)

    def transition(
        self,
        state: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Propagate state forward one step."""
        return self._bounded_transition(state, rng)

    def log_observation_density(
        self,
        y: float | NDArray[np.float64],
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t)."""
        y_arr = np.atleast_1d(y).astype(np.float64)
        obs_sigma = self.params["obs_sigma"]
        diff = y_arr[np.newaxis, :] - state
        log_dens = np.sum(
            -0.5 * np.log(2 * np.pi)
            - np.log(obs_sigma)
            - 0.5 * (diff / obs_sigma) ** 2,
            axis=1,
        )
        return log_dens

    def simulate(
        self, T: int, seed: int | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate from the model."""
        rng = np.random.default_rng(seed)
        phi = self.params["phi"]
        sigma = self.params["sigma"]
        obs_sigma = self.params["obs_sigma"]

        states = np.zeros((T, self.k_states))
        obs = np.zeros((T, self.k_obs))

        x = self.base_mu.copy()
        for t in range(T):
            eta = rng.standard_normal(self.k_states)
            x = self.base_mu + phi * (x - self.base_mu) + sigma * eta
            # Apply bounds
            for i, (lo, hi) in enumerate(self.bounds):
                if lo is not None:
                    x[i] = max(lo, x[i])
                if hi is not None:
                    x[i] = min(hi, x[i])
            states[t] = x
            obs[t] = x + obs_sigma * rng.standard_normal(self.k_obs)

        return {"observations": obs, "states": states}
