"""
Nonlinear Regime-Switching model for particle filtering.

State: (x_t, s_t) where s_t is a discrete Markov regime.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any


class NonlinearRegime:
    """Nonlinear regime-switching model.

    Parameters
    ----------
    n_regimes : int
        Number of regimes. Default 2.
    k_continuous : int
        Dimension of continuous state x_t. Default 1.
    transition_matrix : NDArray | None
        Markov transition matrix P, shape (n_regimes, n_regimes).
        P[i,j] = P(s_t=j | s_{t-1}=i).
    regime_params : list[dict[str, float]] | None
        Parameters for each regime.
    params : dict[str, float] | None
        Global parameters.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        k_continuous: int = 1,
        transition_matrix: NDArray[np.float64] | None = None,
        regime_params: list[dict[str, float]] | None = None,
        params: dict[str, float] | None = None,
    ) -> None:
        self.n_regimes = n_regimes
        self.k_continuous = k_continuous
        self.k_states = k_continuous + 1  # (x_t, s_t)
        self.k_obs = 1

        if transition_matrix is not None:
            self.P = np.asarray(transition_matrix, dtype=np.float64)
        else:
            # Default: persistent regimes
            self.P = np.full(
                (n_regimes, n_regimes), 0.05 / (n_regimes - 1)
            )
            np.fill_diagonal(self.P, 0.95)

        if regime_params is not None:
            self.regime_params = regime_params
        else:
            # Default: two regimes with different means and volatilities
            self.regime_params = [
                {"mu": 0.05, "phi": 0.9, "sigma": 0.1, "obs_sigma": 0.5},
                {"mu": -0.1, "phi": 0.8, "sigma": 0.3, "obs_sigma": 1.5},
            ]

        self.params = params if params is not None else {}
        self.param_names = self._get_param_names()

    def _get_param_names(self) -> list[str]:
        names: list[str] = []
        for k in range(self.n_regimes):
            for key in self.regime_params[k]:
                names.append(f"regime{k}_{key}")
        # Transition probabilities
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                if i != j:
                    names.append(f"p_{i}{j}")
        return names

    def default_prior(self) -> dict[str, dict[str, Any]]:
        """Return default prior distributions."""
        priors: dict[str, dict[str, Any]] = {}
        for k in range(self.n_regimes):
            priors[f"regime{k}_phi"] = {
                "distribution": "beta", "a": 10.0, "b": 2.0
            }
            priors[f"regime{k}_sigma"] = {
                "distribution": "inverse_gamma", "a": 3.0, "b": 0.1
            }
            priors[f"regime{k}_mu"] = {
                "distribution": "normal", "loc": 0.0, "scale": 1.0
            }
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                if i != j:
                    priors[f"p_{i}{j}"] = {
                        "distribution": "beta", "a": 1.0, "b": 10.0
                    }
        return priors

    def initial_state(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample initial state (x_0, s_0)."""
        states = np.zeros((n_particles, self.k_states))
        # Regime: uniform initial
        states[:, -1] = rng.integers(0, self.n_regimes, size=n_particles)
        # Continuous state: small noise
        states[:, :self.k_continuous] = rng.standard_normal(
            (n_particles, self.k_continuous)
        ) * 0.1
        return states

    def transition(
        self,
        state: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Propagate (x_t, s_t) forward one step."""
        n = state.shape[0]
        new_state = np.zeros_like(state)

        # Regime transition
        s = state[:, -1].astype(int)
        for i in range(n):
            new_state[i, -1] = rng.choice(
                self.n_regimes, p=self.P[s[i]]
            )

        # Continuous state transition (depends on new regime)
        s_new = new_state[:, -1].astype(int)
        for k in range(self.n_regimes):
            mask = s_new == k
            if not np.any(mask):
                continue
            n_k = int(mask.sum())
            rp = self.regime_params[k]
            phi = rp["phi"]
            sigma = rp["sigma"]
            mu = rp["mu"]
            x = state[mask, :self.k_continuous]
            eta = rng.standard_normal((n_k, self.k_continuous))
            new_state[mask, :self.k_continuous] = (
                mu + phi * (x - mu) + sigma * eta
            )

        return new_state

    def log_observation_density(
        self,
        y: float | NDArray[np.float64],
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t, s_t)."""
        n = state.shape[0]
        log_dens = np.zeros(n)
        s = state[:, -1].astype(int)
        x = state[:, 0]
        y_val = float(y)

        for k in range(self.n_regimes):
            mask = s == k
            if not np.any(mask):
                continue
            rp = self.regime_params[k]
            obs_sigma = rp.get("obs_sigma", 1.0)
            diff = y_val - x[mask]
            log_dens[mask] = (
                -0.5 * np.log(2 * np.pi)
                - np.log(obs_sigma)
                - 0.5 * (diff / obs_sigma) ** 2
            )

        return log_dens

    def simulate(
        self, T: int, seed: int | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate from the model."""
        rng = np.random.default_rng(seed)

        states = np.zeros((T, self.k_states))
        observations = np.zeros((T, 1))

        # Initial state
        s = rng.integers(0, self.n_regimes)
        rp = self.regime_params[s]
        x = rng.standard_normal(self.k_continuous) * 0.1

        for t in range(T):
            # Regime transition
            s = rng.choice(self.n_regimes, p=self.P[s])
            rp = self.regime_params[s]

            # Continuous transition
            x = (
                rp["mu"]
                + rp["phi"] * (x - rp["mu"])
                + rp["sigma"] * rng.standard_normal(self.k_continuous)
            )

            # Observation
            obs_sigma = rp.get("obs_sigma", 1.0)
            y = x[0] + obs_sigma * rng.standard_normal()

            states[t, :self.k_continuous] = x
            states[t, -1] = s
            observations[t, 0] = y

        return {"observations": observations, "states": states}
