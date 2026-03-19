"""
Continuous-time models with Euler-Maruyama discretization.

Variants:
- 'cir': Cox-Ingersoll-Ross (interest rate)
- 'vasicek': Vasicek (exact Gaussian discretization)
- 'heston': Heston stochastic volatility (2D state)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any


class ContinuousTime:
    """Continuous-time model with Euler-Maruyama discretization.

    Parameters
    ----------
    variant : str
        One of 'cir', 'vasicek', 'heston'.
    params : dict[str, float] | None
        Model parameters.
    dt : float
        Time step. Default 1/252 (daily).
    n_substeps : int
        Number of substeps per dt. Default 1.
    """

    VARIANTS = ("cir", "vasicek", "heston")

    def __init__(
        self,
        variant: str = "cir",
        params: dict[str, float] | None = None,
        dt: float = 1.0 / 252.0,
        n_substeps: int = 1,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {self.VARIANTS}"
            )
        self.variant = variant
        self.dt = dt
        self.n_substeps = n_substeps
        self.sub_dt = dt / n_substeps

        if variant == "heston":
            self.k_states = 2  # (log_S, v)
        else:
            self.k_states = 1  # r
        self.k_obs = 1

        self.param_names = self._get_param_names()
        self.params = params if params is not None else self._default_params()

    def _get_param_names(self) -> list[str]:
        if self.variant == "cir":
            return ["kappa", "theta", "sigma"]
        elif self.variant == "vasicek":
            return ["kappa", "theta", "sigma"]
        elif self.variant == "heston":
            return ["mu", "kappa", "theta", "sigma_v", "rho"]
        return []

    def _default_params(self) -> dict[str, float]:
        if self.variant == "cir":
            return {"kappa": 0.5, "theta": 0.05, "sigma": 0.1}
        elif self.variant == "vasicek":
            return {"kappa": 0.5, "theta": 0.05, "sigma": 0.02}
        elif self.variant == "heston":
            return {
                "mu": 0.05,
                "kappa": 5.0,
                "theta": 0.04,
                "sigma_v": 0.5,
                "rho": -0.7,
            }
        return {}

    def default_prior(self) -> dict[str, dict[str, Any]]:
        """Return default prior distributions."""
        priors: dict[str, dict[str, Any]] = {
            "kappa": {"distribution": "gamma", "a": 2.0, "b": 0.5},
            "theta": {"distribution": "gamma", "a": 2.0, "b": 0.05},
        }
        if self.variant in ("cir", "vasicek"):
            priors["sigma"] = {
                "distribution": "inverse_gamma", "a": 3.0, "b": 0.05
            }
        elif self.variant == "heston":
            priors["mu"] = {
                "distribution": "normal", "loc": 0.05, "scale": 0.1
            }
            priors["sigma_v"] = {
                "distribution": "inverse_gamma", "a": 3.0, "b": 0.5
            }
            priors["rho"] = {
                "distribution": "uniform", "low": -1.0, "high": 0.0
            }
        return priors

    def initial_state(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample initial state."""
        if self.variant == "heston":
            theta = self.params["theta"]
            log_s = np.zeros(n_particles)
            v = np.full(n_particles, theta) + rng.standard_normal(n_particles) * 0.005
            v = np.maximum(v, 1e-8)
            return np.column_stack([log_s, v])
        else:
            theta = self.params["theta"]
            r0 = theta + rng.standard_normal(n_particles) * 0.005
            if self.variant == "cir":
                r0 = np.maximum(r0, 1e-8)
            return r0.reshape(-1, 1)

    def transition(
        self,
        state: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Propagate state forward one time step with substeps."""
        n = state.shape[0]
        new_state = state.copy()

        for _ in range(self.n_substeps):
            sdt = self.sub_dt

            if self.variant == "cir":
                kappa = self.params["kappa"]
                theta = self.params["theta"]
                sigma = self.params["sigma"]
                r = np.maximum(new_state[:, 0], 1e-8)
                z = rng.standard_normal(n)
                dr = kappa * (theta - r) * sdt + sigma * np.sqrt(r * sdt) * z
                r_new = r + dr
                # Reflection at zero
                new_state[:, 0] = np.abs(r_new)

            elif self.variant == "vasicek":
                kappa = self.params["kappa"]
                theta = self.params["theta"]
                sigma = self.params["sigma"]
                r = new_state[:, 0]
                # Exact Gaussian discretization
                mean = theta + (r - theta) * np.exp(-kappa * sdt)
                var = (sigma**2 / (2 * kappa)) * (
                    1 - np.exp(-2 * kappa * sdt)
                )
                new_state[:, 0] = rng.normal(mean, np.sqrt(var))

            elif self.variant == "heston":
                mu = self.params["mu"]
                kappa = self.params["kappa"]
                theta = self.params["theta"]
                sigma_v = self.params["sigma_v"]
                rho = self.params["rho"]

                log_s = new_state[:, 0]
                v = np.maximum(new_state[:, 1], 1e-8)

                z1 = rng.standard_normal(n)
                z2 = rng.standard_normal(n)
                w1 = z1
                w2 = rho * z1 + np.sqrt(1 - rho**2) * z2

                new_state[:, 0] = log_s + (
                    (mu - v / 2) * sdt + np.sqrt(v * sdt) * w1
                )
                v_new = v + kappa * (theta - v) * sdt + sigma_v * np.sqrt(v * sdt) * w2
                new_state[:, 1] = np.maximum(v_new, 1e-8)

        return new_state

    def log_observation_density(
        self,
        y: float | NDArray[np.float64],
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t).

        For CIR/Vasicek: observe rate with noise.
        For Heston: observe log-return.
        """
        y_val = float(y)
        obs_noise = 0.001  # small measurement noise

        if self.variant in ("cir", "vasicek"):
            r = state[:, 0]
            log_dens = (
                -0.5 * np.log(2 * np.pi)
                - np.log(obs_noise)
                - 0.5 * ((y_val - r) / obs_noise) ** 2
            )
            return log_dens

        elif self.variant == "heston":
            log_s = state[:, 0]
            v = np.maximum(state[:, 1], 1e-8)
            obs_std = np.sqrt(v * self.dt)
            log_dens = (
                -0.5 * np.log(2 * np.pi)
                - np.log(obs_std)
                - 0.5 * ((y_val - log_s) / obs_std) ** 2
            )
            return log_dens

        raise ValueError(f"Unknown variant: {self.variant}")

    def simulate(
        self, T: int, seed: int | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate from the model."""
        rng = np.random.default_rng(seed)

        if self.variant == "cir":
            kappa = self.params["kappa"]
            theta = self.params["theta"]
            sigma = self.params["sigma"]

            r = np.zeros(T)
            r[0] = theta
            for t in range(1, T):
                r_curr = r[t - 1]
                for _sub in range(self.n_substeps):
                    sdt = self.sub_dt
                    z = rng.standard_normal()
                    r_curr = max(r_curr, 1e-8)
                    dr = kappa * (theta - r_curr) * sdt + sigma * np.sqrt(r_curr * sdt) * z
                    r_curr = abs(r_curr + dr)
                r[t] = r_curr
            obs = r + 0.001 * rng.standard_normal(T)
            return {
                "observations": obs.reshape(-1, 1),
                "states": r.reshape(-1, 1),
            }

        elif self.variant == "vasicek":
            kappa = self.params["kappa"]
            theta = self.params["theta"]
            sigma = self.params["sigma"]

            r = np.zeros(T)
            r[0] = theta
            for t in range(1, T):
                mean = theta + (r[t - 1] - theta) * np.exp(-kappa * self.dt)
                var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * self.dt))
                r[t] = rng.normal(mean, np.sqrt(var))
            obs = r + 0.001 * rng.standard_normal(T)
            return {
                "observations": obs.reshape(-1, 1),
                "states": r.reshape(-1, 1),
            }

        elif self.variant == "heston":
            mu = self.params["mu"]
            kappa = self.params["kappa"]
            theta = self.params["theta"]
            sigma_v = self.params["sigma_v"]
            rho = self.params["rho"]

            log_s = np.zeros(T + 1)
            v = np.zeros(T + 1)
            v[0] = theta
            for t in range(T):
                sdt = self.dt
                v_curr = max(v[t], 1e-8)
                z1 = rng.standard_normal()
                z2 = rng.standard_normal()
                w1 = z1
                w2 = rho * z1 + np.sqrt(1 - rho**2) * z2
                log_s[t + 1] = log_s[t] + (mu - v_curr / 2) * sdt + np.sqrt(v_curr * sdt) * w1
                v[t + 1] = max(
                    v_curr + kappa * (theta - v_curr) * sdt + sigma_v * np.sqrt(v_curr * sdt) * w2,
                    1e-8,
                )
            returns = np.diff(log_s)
            return {
                "observations": returns.reshape(-1, 1),
                "states": np.column_stack([log_s[1:], v[1:]]),
                "prices": np.exp(log_s),
            }

        raise ValueError(f"Unknown variant: {self.variant}")
