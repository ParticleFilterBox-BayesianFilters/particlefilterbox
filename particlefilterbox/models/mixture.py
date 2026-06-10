"""
Mixture observation models for particle filtering.

y_t | x_t ~ sum_{k=1}^K pi_k * f_k(y_t | x_t)

Uses log-sum-exp for numerical stability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class Mixture:
    """Mixture observation model.

    Parameters
    ----------
    n_components : int
        Number of mixture components K. Default 2.
    weights : NDArray | None
        Mixture weights pi_k, shape (K,). Must sum to 1.
    component_means : NDArray | None
        Mean offsets for each component, shape (K,).
    component_stds : NDArray | None
        Standard deviations for each component, shape (K,).
    state_phi : float
        AR(1) persistence for latent state. Default 0.95.
    state_sigma : float
        Innovation std for latent state. Default 0.2.
    params : dict[str, float] | None
        Additional parameters.
    """

    def __init__(
        self,
        n_components: int = 2,
        weights: NDArray[np.float64] | None = None,
        component_means: NDArray[np.float64] | None = None,
        component_stds: NDArray[np.float64] | None = None,
        state_phi: float = 0.95,
        state_sigma: float = 0.2,
        params: dict[str, float] | None = None,
    ) -> None:
        self.n_components = n_components
        self.k_states = 1
        self.k_obs = 1

        if weights is not None:
            self.weights = np.asarray(weights)
        else:
            self.weights = np.ones(n_components) / n_components

        if component_means is not None:
            self.component_means = np.asarray(component_means)
        else:
            self.component_means = np.linspace(-1, 1, n_components)

        if component_stds is not None:
            self.component_stds = np.asarray(component_stds)
        else:
            self.component_stds = np.ones(n_components) * 0.5

        self.state_phi = state_phi
        self.state_sigma = state_sigma

        self.params = params if params is not None else {
            "phi": state_phi,
            "sigma": state_sigma,
        }
        self.param_names = [
            "phi", "sigma",
        ] + [f"weight_{k}" for k in range(n_components)] + [
            f"mean_{k}" for k in range(n_components)
        ] + [f"std_{k}" for k in range(n_components)]

    def _logsumexp(
        self, log_vals: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Numerically stable log-sum-exp.

        Parameters
        ----------
        log_vals : NDArray
            Shape (n_particles, K).

        Returns
        -------
        NDArray
            Shape (n_particles,).
        """
        max_val = np.max(log_vals, axis=1, keepdims=True)
        return (
            max_val.squeeze()
            + np.log(np.sum(np.exp(log_vals - max_val), axis=1))
        )

    def default_prior(self) -> dict[str, dict[str, Any]]:
        """Return default prior distributions."""
        priors: dict[str, dict[str, Any]] = {
            "phi": {"distribution": "beta", "a": 20.0, "b": 1.5},
            "sigma": {"distribution": "inverse_gamma", "a": 3.0, "b": 0.1},
        }
        for k in range(self.n_components):
            priors[f"mean_{k}"] = {
                "distribution": "normal", "loc": 0.0, "scale": 2.0
            }
            priors[f"std_{k}"] = {
                "distribution": "inverse_gamma", "a": 3.0, "b": 0.5
            }
        return priors

    def initial_state(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample initial state."""
        var_stat = self.state_sigma**2 / (1 - self.state_phi**2)
        x0 = rng.normal(0, np.sqrt(var_stat), size=n_particles)
        return x0.reshape(-1, 1)

    def transition(
        self,
        state: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Propagate state forward one step."""
        n = state.shape[0]
        x = state[:, 0]
        x_new = self.state_phi * x + self.state_sigma * rng.standard_normal(n)
        return x_new.reshape(-1, 1)

    def log_observation_density(
        self,
        y: float | NDArray[np.float64],
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t) via log-sum-exp over components.

        log p(y|x) = logsumexp_k [log(pi_k) + log N(y; x + mu_k, sigma_k^2)]
        """
        n = state.shape[0]
        x = state[:, 0]
        y_val = float(y)

        # Shape: (n, K)
        log_components = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            mu_k = x + self.component_means[k]
            sigma_k = self.component_stds[k]
            log_components[:, k] = (
                np.log(self.weights[k])
                - 0.5 * np.log(2 * np.pi)
                - np.log(sigma_k)
                - 0.5 * ((y_val - mu_k) / sigma_k) ** 2
            )

        return self._logsumexp(log_components)

    def simulate(
        self, T: int, seed: int | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate from the model."""
        rng = np.random.default_rng(seed)
        var_stat = self.state_sigma**2 / (1 - self.state_phi**2)

        states = np.zeros((T, 1))
        obs = np.zeros((T, 1))

        x = rng.normal(0, np.sqrt(var_stat))
        for t in range(T):
            x = self.state_phi * x + self.state_sigma * rng.standard_normal()
            # Sample component
            k = rng.choice(self.n_components, p=self.weights)
            y = x + self.component_means[k] + self.component_stds[k] * rng.standard_normal()
            states[t, 0] = x
            obs[t, 0] = y

        return {"observations": obs, "states": states}
