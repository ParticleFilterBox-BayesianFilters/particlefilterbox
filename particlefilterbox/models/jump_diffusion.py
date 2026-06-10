"""
Jump-Diffusion models for particle filtering.

Variants:
- 'merton' (Merton 1976)
- 'kou' (Kou 2002)
- 'bates' (Bates 1996 = Merton + Heston)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class JumpDiffusion:
    """Jump-Diffusion model with multiple variants.

    Parameters
    ----------
    variant : str
        One of 'merton', 'kou', 'bates'.
    params : dict[str, float] | None
        Model parameters. If None, uses defaults.
    dt : float
        Time step size. Default 1/252 (daily).
    n_substeps : int
        Number of substeps per dt for accuracy. Default 1.
    """

    VARIANTS = ("merton", "kou", "bates")

    def __init__(
        self,
        variant: str = "merton",
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

        if variant == "bates":
            self.k_states = 2  # (log_S, v)
        else:
            self.k_states = 1  # log_S
        self.k_obs = 1

        self.param_names = self._get_param_names()
        self.params = params if params is not None else self._default_params()

    def _get_param_names(self) -> list[str]:
        """Return parameter names for the variant."""
        if self.variant == "merton":
            return [
                "mu", "sigma", "lambda_jump", "mu_jump", "sigma_jump"
            ]
        elif self.variant == "kou":
            return [
                "mu", "sigma", "lambda_jump",
                "p_up", "eta1", "eta2",
            ]
        elif self.variant == "bates":
            return [
                "mu", "kappa", "theta", "sigma_v", "rho",
                "lambda_jump", "mu_jump", "sigma_jump",
            ]
        return []

    def _default_params(self) -> dict[str, float]:
        """Return default parameters for the variant."""
        if self.variant == "merton":
            return {
                "mu": 0.08,
                "sigma": 0.2,
                "lambda_jump": 3.0,
                "mu_jump": -0.02,
                "sigma_jump": 0.05,
            }
        elif self.variant == "kou":
            return {
                "mu": 0.08,
                "sigma": 0.2,
                "lambda_jump": 3.0,
                "p_up": 0.4,
                "eta1": 10.0,
                "eta2": 5.0,
            }
        elif self.variant == "bates":
            return {
                "mu": 0.08,
                "kappa": 5.0,
                "theta": 0.04,
                "sigma_v": 0.5,
                "rho": -0.7,
                "lambda_jump": 3.0,
                "mu_jump": -0.02,
                "sigma_jump": 0.05,
            }
        return {}

    def default_prior(self) -> dict[str, dict[str, Any]]:
        """Return default prior distributions for PMMH."""
        priors: dict[str, dict[str, Any]] = {
            "mu": {"distribution": "normal", "loc": 0.05, "scale": 0.1},
            "sigma": {"distribution": "inverse_gamma", "a": 5.0, "b": 0.2},
        }
        if self.variant in ("merton", "kou", "bates"):
            priors["lambda_jump"] = {
                "distribution": "gamma", "a": 2.0, "b": 1.0
            }
        if self.variant == "merton":
            priors["mu_jump"] = {
                "distribution": "normal", "loc": 0.0, "scale": 0.1
            }
            priors["sigma_jump"] = {
                "distribution": "inverse_gamma", "a": 5.0, "b": 0.05
            }
        elif self.variant == "kou":
            priors["p_up"] = {
                "distribution": "beta", "a": 2.0, "b": 3.0
            }
            priors["eta1"] = {
                "distribution": "gamma", "a": 5.0, "b": 0.5
            }
            priors["eta2"] = {
                "distribution": "gamma", "a": 3.0, "b": 0.5
            }
        elif self.variant == "bates":
            priors["kappa"] = {
                "distribution": "gamma", "a": 5.0, "b": 1.0
            }
            priors["theta"] = {
                "distribution": "inverse_gamma", "a": 5.0, "b": 0.04
            }
            priors["sigma_v"] = {
                "distribution": "inverse_gamma", "a": 5.0, "b": 0.5
            }
            priors["rho"] = {
                "distribution": "uniform", "low": -1.0, "high": 0.0
            }
            priors["mu_jump"] = {
                "distribution": "normal", "loc": 0.0, "scale": 0.1
            }
            priors["sigma_jump"] = {
                "distribution": "inverse_gamma", "a": 5.0, "b": 0.05
            }
        return priors

    def _kou_jump(
        self, n: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample Kou double-exponential jump sizes."""
        p_up = self.params["p_up"]
        eta1 = self.params["eta1"]
        eta2 = self.params["eta2"]
        u = rng.uniform(size=n)
        jumps = np.where(
            u < p_up,
            rng.exponential(1.0 / eta1, size=n),
            -rng.exponential(1.0 / eta2, size=n),
        )
        return jumps

    def initial_state(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample initial state."""
        if self.variant == "bates":
            log_s = np.zeros(n_particles)
            v0 = np.full(n_particles, self.params["theta"])
            return np.column_stack([log_s, v0])
        else:
            return np.zeros((n_particles, 1))

    def transition(
        self,
        state: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Propagate state forward one time step (dt) with substeps."""
        n = state.shape[0]
        new_state = state.copy()

        for _ in range(self.n_substeps):
            sdt = self.sub_dt

            if self.variant == "merton":
                mu = self.params["mu"]
                sigma = self.params["sigma"]
                lam = self.params["lambda_jump"]
                mu_j = self.params["mu_jump"]
                sigma_j = self.params["sigma_jump"]
                k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

                z = rng.standard_normal(n)
                n_jumps = rng.poisson(lam * sdt, size=n)
                jump_sum = np.zeros(n)
                for i in range(n):
                    if n_jumps[i] > 0:
                        jump_sum[i] = rng.normal(
                            mu_j * n_jumps[i],
                            sigma_j * np.sqrt(n_jumps[i]),
                        )
                new_state[:, 0] += (
                    (mu - 0.5 * sigma**2 - lam * k) * sdt
                    + sigma * np.sqrt(sdt) * z
                    + jump_sum
                )

            elif self.variant == "kou":
                mu = self.params["mu"]
                sigma = self.params["sigma"]
                lam = self.params["lambda_jump"]
                p_up = self.params["p_up"]
                eta1 = self.params["eta1"]
                eta2 = self.params["eta2"]
                k = p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1

                z = rng.standard_normal(n)
                n_jumps = rng.poisson(lam * sdt, size=n)
                jump_sum = np.zeros(n)
                for i in range(n):
                    if n_jumps[i] > 0:
                        jumps = self._kou_jump(n_jumps[i], rng)
                        jump_sum[i] = jumps.sum()
                new_state[:, 0] += (
                    (mu - 0.5 * sigma**2 - lam * k) * sdt
                    + sigma * np.sqrt(sdt) * z
                    + jump_sum
                )

            elif self.variant == "bates":
                mu = self.params["mu"]
                kappa = self.params["kappa"]
                theta = self.params["theta"]
                sigma_v = self.params["sigma_v"]
                rho = self.params["rho"]
                lam = self.params["lambda_jump"]
                mu_j = self.params["mu_jump"]
                sigma_j = self.params["sigma_jump"]
                k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

                log_s = new_state[:, 0]
                v = np.maximum(new_state[:, 1], 1e-8)

                z1 = rng.standard_normal(n)
                z2 = rng.standard_normal(n)
                w1 = z1
                w2 = rho * z1 + np.sqrt(1 - rho**2) * z2

                n_jumps = rng.poisson(lam * sdt, size=n)
                jump_sum = np.zeros(n)
                for i in range(n):
                    if n_jumps[i] > 0:
                        jump_sum[i] = rng.normal(
                            mu_j * n_jumps[i],
                            sigma_j * np.sqrt(n_jumps[i]),
                        )

                new_state[:, 0] = log_s + (
                    (mu - v / 2 - lam * k) * sdt
                    + np.sqrt(v * sdt) * w1
                    + jump_sum
                )
                new_state[:, 1] = np.maximum(
                    v + kappa * (theta - v) * sdt
                    + sigma_v * np.sqrt(v * sdt) * w2,
                    1e-8,
                )

        return new_state

    def log_observation_density(
        self,
        y: float | NDArray[np.float64],
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t).

        For JumpDiffusion, observations are log-returns.
        With the state being log_S, y_t = log_S_t - log_S_{t-1}.
        We use a simple Gaussian approximation for the observation.
        """
        n = state.shape[0]
        log_s = state[:, 0]

        if self.variant == "bates":
            v = np.maximum(state[:, 1], 1e-8)
            obs_std = np.sqrt(v * self.dt)
        else:
            sigma = self.params["sigma"]
            obs_std = np.full(n, sigma * np.sqrt(self.dt))

        diff = float(y) - log_s
        log_dens = (
            -0.5 * np.log(2 * np.pi)
            - np.log(obs_std)
            - 0.5 * (diff / obs_std) ** 2
        )
        return log_dens

    def simulate(
        self, T: int, seed: int | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate from the model.

        Parameters
        ----------
        T : int
            Number of time steps.
        seed : int | None
            Random seed.

        Returns
        -------
        dict
            Keys: 'observations' (log returns), 'states', 'prices'.
        """
        rng = np.random.default_rng(seed)

        if self.variant == "merton":
            mu = self.params["mu"]
            sigma = self.params["sigma"]
            lam = self.params["lambda_jump"]
            mu_j = self.params["mu_jump"]
            sigma_j = self.params["sigma_jump"]
            k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

            log_s = np.zeros(T + 1)
            for t in range(T):
                for _ in range(self.n_substeps):
                    sdt = self.sub_dt
                    z = rng.standard_normal()
                    nj = rng.poisson(lam * sdt)
                    jump = 0.0
                    if nj > 0:
                        jump = rng.normal(
                            mu_j * nj, sigma_j * np.sqrt(nj)
                        )
                    log_s[t + 1] = log_s[t] + (
                        (mu - 0.5 * sigma**2 - lam * k) * sdt
                        + sigma * np.sqrt(sdt) * z
                        + jump
                    )
            returns = np.diff(log_s)
            return {
                "observations": returns.reshape(-1, 1),
                "states": log_s[1:].reshape(-1, 1),
                "prices": np.exp(log_s),
            }

        elif self.variant == "kou":
            mu = self.params["mu"]
            sigma = self.params["sigma"]
            lam = self.params["lambda_jump"]
            k = (
                self.params["p_up"] * self.params["eta1"]
                / (self.params["eta1"] - 1)
                + (1 - self.params["p_up"]) * self.params["eta2"]
                / (self.params["eta2"] + 1)
                - 1
            )

            log_s = np.zeros(T + 1)
            for t in range(T):
                sdt = self.dt
                z = rng.standard_normal()
                nj = rng.poisson(lam * sdt)
                jump = 0.0
                if nj > 0:
                    jumps = self._kou_jump(nj, rng)
                    jump = jumps.sum()
                log_s[t + 1] = log_s[t] + (
                    (mu - 0.5 * sigma**2 - lam * k) * sdt
                    + sigma * np.sqrt(sdt) * z
                    + jump
                )
            returns = np.diff(log_s)
            return {
                "observations": returns.reshape(-1, 1),
                "states": log_s[1:].reshape(-1, 1),
                "prices": np.exp(log_s),
            }

        elif self.variant == "bates":
            mu = self.params["mu"]
            kappa = self.params["kappa"]
            theta = self.params["theta"]
            sigma_v = self.params["sigma_v"]
            rho = self.params["rho"]
            lam = self.params["lambda_jump"]
            mu_j = self.params["mu_jump"]
            sigma_j = self.params["sigma_jump"]
            k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

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
                nj = rng.poisson(lam * sdt)
                jump = 0.0
                if nj > 0:
                    jump = rng.normal(mu_j * nj, sigma_j * np.sqrt(nj))
                log_s[t + 1] = log_s[t] + (
                    (mu - v_curr / 2 - lam * k) * sdt
                    + np.sqrt(v_curr * sdt) * w1
                    + jump
                )
                v[t + 1] = max(
                    v_curr + kappa * (theta - v_curr) * sdt
                    + sigma_v * np.sqrt(v_curr * sdt) * w2,
                    1e-8,
                )
            returns = np.diff(log_s)
            return {
                "observations": returns.reshape(-1, 1),
                "states": np.column_stack([log_s[1:], v[1:]]),
                "prices": np.exp(log_s),
            }

        raise ValueError(f"Unknown variant: {self.variant}")
