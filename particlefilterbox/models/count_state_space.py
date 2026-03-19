"""
Count State-Space models for particle filtering.

Variants:
- 'poisson': Poisson observations with AR(1) log-intensity
- 'binomial': Binomial observations with AR(1) logistic probability
- 'sir': SIR epidemic model with sub-reporting
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import gammaln
from typing import Any


class CountStateSpace:
    """Count state-space model.

    Parameters
    ----------
    variant : str
        One of 'poisson', 'binomial', 'sir'.
    params : dict[str, float] | None
        Model parameters.
    population : int
        Population size for SIR model. Default 10000.
    """

    VARIANTS = ("poisson", "binomial", "sir")

    def __init__(
        self,
        variant: str = "poisson",
        params: dict[str, float] | None = None,
        population: int = 10000,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {self.VARIANTS}"
            )
        self.variant = variant
        self.population = population

        if variant == "sir":
            self.k_states = 3  # (S, I, R)
            self.k_obs = 1     # reported cases
        elif variant == "binomial":
            self.k_states = 1  # latent AR(1)
            self.k_obs = 1
        else:
            self.k_states = 1  # latent AR(1)
            self.k_obs = 1

        self.param_names = self._get_param_names()
        self.params = params if params is not None else self._default_params()

    def _get_param_names(self) -> list[str]:
        if self.variant == "poisson":
            return ["phi", "sigma", "mu"]
        elif self.variant == "binomial":
            return ["phi", "sigma", "mu", "n_trials"]
        elif self.variant == "sir":
            return ["beta", "gamma", "rho", "sigma_S", "sigma_I"]
        return []

    def _default_params(self) -> dict[str, float]:
        if self.variant == "poisson":
            return {"phi": 0.95, "sigma": 0.2, "mu": 2.0}
        elif self.variant == "binomial":
            return {"phi": 0.95, "sigma": 0.3, "mu": 0.0, "n_trials": 100.0}
        elif self.variant == "sir":
            return {
                "beta": 0.3,
                "gamma": 0.1,
                "rho": 0.5,
                "sigma_S": 10.0,
                "sigma_I": 5.0,
            }
        return {}

    def default_prior(self) -> dict[str, dict[str, Any]]:
        """Return default prior distributions."""
        if self.variant == "poisson":
            return {
                "phi": {"distribution": "beta", "a": 20.0, "b": 1.5},
                "sigma": {"distribution": "inverse_gamma", "a": 3.0, "b": 0.1},
                "mu": {"distribution": "normal", "loc": 2.0, "scale": 2.0},
            }
        elif self.variant == "binomial":
            return {
                "phi": {"distribution": "beta", "a": 20.0, "b": 1.5},
                "sigma": {"distribution": "inverse_gamma", "a": 3.0, "b": 0.1},
                "mu": {"distribution": "normal", "loc": 0.0, "scale": 2.0},
                "n_trials": {"distribution": "fixed", "value": 100.0},
            }
        elif self.variant == "sir":
            return {
                "beta": {"distribution": "gamma", "a": 3.0, "b": 0.1},
                "gamma": {"distribution": "gamma", "a": 1.0, "b": 0.1},
                "rho": {"distribution": "beta", "a": 5.0, "b": 5.0},
                "sigma_S": {"distribution": "inverse_gamma", "a": 3.0, "b": 10.0},
                "sigma_I": {"distribution": "inverse_gamma", "a": 3.0, "b": 5.0},
            }
        return {}

    def initial_state(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample initial state."""
        if self.variant == "sir":
            N = self.population
            I0 = max(1, int(0.01 * N))
            S0 = N - I0
            R0 = 0
            states = np.zeros((n_particles, 3))
            states[:, 0] = S0 + rng.standard_normal(n_particles) * 10
            states[:, 1] = I0 + rng.standard_normal(n_particles) * 5
            states[:, 2] = R0
            states[:, 0] = np.maximum(states[:, 0], 0)
            states[:, 1] = np.maximum(states[:, 1], 1)
            return states
        else:
            mu = self.params["mu"]
            phi = self.params["phi"]
            sigma = self.params["sigma"]
            var_stat = sigma**2 / (1 - phi**2)
            x0 = rng.normal(mu / (1 - phi), np.sqrt(var_stat), size=n_particles)
            return x0.reshape(-1, 1)

    def transition(
        self,
        state: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Propagate state forward one step."""
        n = state.shape[0]

        if self.variant == "sir":
            beta = self.params["beta"]
            gamma = self.params["gamma"]
            sigma_S = self.params["sigma_S"]
            sigma_I = self.params["sigma_I"]
            N = self.population

            S = state[:, 0]
            I = state[:, 1]
            R = state[:, 2]

            new_infections = beta * S * I / N
            recoveries = gamma * I

            S_new = S - new_infections + rng.standard_normal(n) * sigma_S
            I_new = I + new_infections - recoveries + rng.standard_normal(n) * sigma_I
            R_new = R + recoveries

            S_new = np.clip(S_new, 0, N)
            I_new = np.clip(I_new, 0, N)
            R_new = np.clip(R_new, 0, N)

            return np.column_stack([S_new, I_new, R_new])

        else:
            phi = self.params["phi"]
            sigma = self.params["sigma"]
            mu = self.params["mu"]
            x = state[:, 0]
            x_new = mu + phi * (x - mu) + sigma * rng.standard_normal(n)
            return x_new.reshape(-1, 1)

    def log_observation_density(
        self,
        y: float | NDArray[np.float64],
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t)."""
        y_int = int(round(float(y)))

        if self.variant == "poisson":
            x = state[:, 0]
            lam = np.exp(x)
            lam = np.clip(lam, 1e-10, 1e6)
            log_dens = y_int * np.log(lam) - lam - gammaln(y_int + 1)
            return log_dens

        elif self.variant == "binomial":
            x = state[:, 0]
            n_trials = int(self.params["n_trials"])
            p = 1.0 / (1.0 + np.exp(-x))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            log_dens: NDArray[np.float64] = stats.binom.logpmf(y_int, n_trials, p)
            return log_dens

        elif self.variant == "sir":
            I = state[:, 1]
            rho = self.params["rho"]
            I_int = np.maximum(np.round(I).astype(int), 1)
            p = np.clip(rho, 1e-10, 1 - 1e-10)
            log_dens_arr = np.zeros(state.shape[0])
            for i in range(state.shape[0]):
                n_i = max(int(I_int[i]), y_int)
                log_dens_arr[i] = stats.binom.logpmf(y_int, n_i, p)
            return log_dens_arr

        raise ValueError(f"Unknown variant: {self.variant}")

    def simulate(
        self, T: int, seed: int | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate from the model."""
        rng = np.random.default_rng(seed)

        if self.variant == "sir":
            N = self.population
            I0 = max(1, int(0.01 * N))
            S0 = N - I0
            beta = self.params["beta"]
            gamma = self.params["gamma"]
            rho = self.params["rho"]
            sigma_S = self.params["sigma_S"]
            sigma_I = self.params["sigma_I"]

            states = np.zeros((T, 3))
            obs = np.zeros((T, 1))
            S, I, R = float(S0), float(I0), 0.0

            for t in range(T):
                new_inf = beta * S * I / N
                rec = gamma * I
                S = max(0, S - new_inf + rng.standard_normal() * sigma_S)
                I = max(1, I + new_inf - rec + rng.standard_normal() * sigma_I)
                R = max(0, R + rec)
                states[t] = [S, I, R]
                obs[t, 0] = rng.binomial(max(1, int(round(I))), rho)

            return {"observations": obs, "states": states}

        elif self.variant == "poisson":
            phi = self.params["phi"]
            sigma = self.params["sigma"]
            mu = self.params["mu"]
            var_stat = sigma**2 / (1 - phi**2)

            x = np.zeros(T)
            y = np.zeros(T)
            x[0] = rng.normal(mu / (1 - phi), np.sqrt(var_stat))
            y[0] = rng.poisson(np.exp(x[0]))
            for t in range(1, T):
                x[t] = mu + phi * (x[t - 1] - mu) + sigma * rng.standard_normal()
                y[t] = rng.poisson(np.exp(x[t]))
            return {
                "observations": y.reshape(-1, 1),
                "states": x.reshape(-1, 1),
            }

        elif self.variant == "binomial":
            phi = self.params["phi"]
            sigma = self.params["sigma"]
            mu = self.params["mu"]
            n_trials = int(self.params["n_trials"])
            var_stat = sigma**2 / (1 - phi**2)

            x = np.zeros(T)
            y = np.zeros(T)
            x[0] = rng.normal(mu / (1 - phi), np.sqrt(var_stat))
            p0 = 1.0 / (1.0 + np.exp(-x[0]))
            y[0] = rng.binomial(n_trials, p0)
            for t in range(1, T):
                x[t] = mu + phi * (x[t - 1] - mu) + sigma * rng.standard_normal()
                p = 1.0 / (1.0 + np.exp(-x[t]))
                y[t] = rng.binomial(n_trials, p)
            return {
                "observations": y.reshape(-1, 1),
                "states": x.reshape(-1, 1),
            }

        raise ValueError(f"Unknown variant: {self.variant}")

    def r0(self) -> float:
        """Compute basic reproduction number R0 for SIR model."""
        if self.variant != "sir":
            raise ValueError("R0 only defined for SIR variant")
        return self.params["beta"] / self.params["gamma"]
