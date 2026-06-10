"""
Stochastic Volatility models for particle filtering.

Variants:
- 'basic' (Kim, Shephard & Chib 1998)
- 'leverage' (Omori et al. 2007)
- 'jumps' (Eraker et al. 2003)
- 'factor' (Chib et al. 2006)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class StochasticVolatility:
    """Stochastic Volatility model with multiple variants.

    Parameters
    ----------
    variant : str
        One of 'basic', 'leverage', 'jumps', 'factor'.
    params : dict[str, float] | None
        Model parameters. If None, uses defaults for the variant.
    k_factor_series : int
        Number of observed series for 'factor' variant. Default 3.
    """

    VARIANTS = ("basic", "leverage", "jumps", "factor")

    def __init__(
        self,
        variant: str = "basic",
        params: dict[str, float] | None = None,
        k_factor_series: int = 3,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {self.VARIANTS}"
            )
        self.variant = variant
        self.k_factor_series = k_factor_series

        # State and observation dimensions
        if variant == "jumps":
            self.k_states = 2  # (h_t, q_t)
            self.k_obs = 1
        elif variant == "factor":
            self.k_states = 1 + k_factor_series  # common + idiosyncratic
            self.k_obs = k_factor_series
        else:
            self.k_states = 1  # h_t
            self.k_obs = 1

        # Parameters
        self.param_names = self._get_param_names()
        self.params = params if params is not None else self._default_params()

    def _get_param_names(self) -> list[str]:
        """Return parameter names for the variant."""
        base = ["mu", "phi", "sigma"]
        if self.variant == "leverage":
            return base + ["rho"]
        elif self.variant == "jumps":
            return base + ["lambda_jump", "mu_jump", "sigma_jump"]
        elif self.variant == "factor":
            return base + [
                f"phi_{k}" for k in range(self.k_factor_series)
            ] + [
                f"sigma_{k}" for k in range(self.k_factor_series)
            ] + [
                f"beta_{k}" for k in range(self.k_factor_series)
            ]
        return base

    def _default_params(self) -> dict[str, float]:
        """Return default parameters for the variant."""
        base: dict[str, float] = {
            "mu": -1.0,
            "phi": 0.97,
            "sigma": 0.15,
        }
        if self.variant == "leverage":
            base["rho"] = -0.5
        elif self.variant == "jumps":
            base["lambda_jump"] = 0.05
            base["mu_jump"] = -0.5
            base["sigma_jump"] = 1.0
        elif self.variant == "factor":
            for k in range(self.k_factor_series):
                base[f"phi_{k}"] = 0.95
                base[f"sigma_{k}"] = 0.2
                base[f"beta_{k}"] = 1.0
        return base

    def default_prior(self) -> dict[str, dict[str, Any]]:
        """Return default prior distributions for PMMH.

        Returns
        -------
        dict
            Keys are parameter names, values are dicts with
            'distribution', 'loc', 'scale' (or other params).
        """
        priors: dict[str, dict[str, Any]] = {
            "mu": {"distribution": "normal", "loc": 0.0, "scale": 5.0},
            "phi": {"distribution": "beta", "a": 20.0, "b": 1.5},
            "sigma": {"distribution": "inverse_gamma", "a": 2.5, "b": 0.025},
        }
        if self.variant == "leverage":
            priors["rho"] = {
                "distribution": "uniform",
                "low": -1.0,
                "high": 1.0,
            }
        elif self.variant == "jumps":
            priors["lambda_jump"] = {
                "distribution": "beta",
                "a": 2.0,
                "b": 40.0,
            }
            priors["mu_jump"] = {
                "distribution": "normal",
                "loc": 0.0,
                "scale": 2.0,
            }
            priors["sigma_jump"] = {
                "distribution": "inverse_gamma",
                "a": 2.5,
                "b": 1.0,
            }
        elif self.variant == "factor":
            for k in range(self.k_factor_series):
                priors[f"phi_{k}"] = {
                    "distribution": "beta",
                    "a": 20.0,
                    "b": 1.5,
                }
                priors[f"sigma_{k}"] = {
                    "distribution": "inverse_gamma",
                    "a": 2.5,
                    "b": 0.025,
                }
                priors[f"beta_{k}"] = {
                    "distribution": "normal",
                    "loc": 1.0,
                    "scale": 1.0,
                }
        return priors

    def initial_state(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample initial state h_0 from stationary distribution.

        Parameters
        ----------
        n_particles : int
            Number of particles to sample.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray
            Shape (n_particles, k_states).
        """
        mu = self.params["mu"]
        phi = self.params["phi"]
        sigma = self.params["sigma"]
        var_stationary = sigma**2 / (1.0 - phi**2)

        if self.variant == "jumps":
            h0 = rng.normal(mu, np.sqrt(var_stationary), size=n_particles)
            q0 = np.zeros(n_particles)
            return np.column_stack([h0, q0])
        elif self.variant == "factor":
            states = np.zeros((n_particles, self.k_states))
            states[:, 0] = rng.normal(
                mu, np.sqrt(var_stationary), size=n_particles
            )
            for k in range(self.k_factor_series):
                phi_k = self.params[f"phi_{k}"]
                sigma_k = self.params[f"sigma_{k}"]
                var_k = sigma_k**2 / (1.0 - phi_k**2)
                states[:, 1 + k] = rng.normal(
                    0.0, np.sqrt(var_k), size=n_particles
                )
            return states
        else:
            h0 = rng.normal(mu, np.sqrt(var_stationary), size=n_particles)
            return h0.reshape(-1, 1)

    def transition(
        self,
        state: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Propagate state forward one step.

        Parameters
        ----------
        state : NDArray
            Current state, shape (n_particles, k_states).
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray
            Next state, shape (n_particles, k_states).
        """
        mu = self.params["mu"]
        phi = self.params["phi"]
        sigma = self.params["sigma"]
        n = state.shape[0]

        # NOTE on the 'leverage' variant (Omori et al. 2007):
        # The leverage effect couples the volatility innovation eta_t with the
        # CONTEMPORANEOUS return innovation eps_t via the correlation rho, i.e.
        #     eta_t = rho * eps_t + sqrt(1 - rho^2) * z_t,   y_t = exp(h_t/2)*eps_t
        # (see simulate() for the data-generating process). The prior transition
        # p(x_t | x_{t-1}) used by a bootstrap particle filter samples eta_t
        # BEFORE y_t is available, so it cannot represent this coupling without
        # conditioning on y_t. We therefore deliberately propagate the 'leverage'
        # state with the same marginal prior dynamics as 'basic' (rho is integrated
        # out, giving eta_t ~ N(0, 1) marginally). The rho correlation is exercised
        # in simulate() (data generation); fully exploiting leverage at inference
        # time would require a non-bootstrap proposal q(x_t | x_{t-1}, y_t) or a
        # mixture-sampler likelihood as in Omori et al. (2007), which this prior
        # transition intentionally does not implement.
        if self.variant in ("basic", "leverage"):
            h = state[:, 0]
            eta = rng.standard_normal(n)
            h_next = mu + phi * (h - mu) + sigma * eta
            return h_next.reshape(-1, 1)

        elif self.variant == "jumps":
            h = state[:, 0]
            lam = self.params["lambda_jump"]
            eta = rng.standard_normal(n)
            h_next = mu + phi * (h - mu) + sigma * eta
            q_next = rng.binomial(1, lam, size=n).astype(np.float64)
            return np.column_stack([h_next, q_next])

        elif self.variant == "factor":
            new_state = np.zeros_like(state)
            h = state[:, 0]
            eta = rng.standard_normal(n)
            new_state[:, 0] = mu + phi * (h - mu) + sigma * eta
            for k in range(self.k_factor_series):
                phi_k = self.params[f"phi_{k}"]
                sigma_k = self.params[f"sigma_{k}"]
                hk = state[:, 1 + k]
                eta_k = rng.standard_normal(n)
                new_state[:, 1 + k] = phi_k * hk + sigma_k * eta_k
            return new_state

        raise ValueError(f"Unknown variant: {self.variant}")

    def log_observation_density(
        self,
        y: float | NDArray[np.float64],
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t) for each particle.

        Parameters
        ----------
        y : float or NDArray
            Observation at time t.
        state : NDArray
            Particles, shape (n_particles, k_states).

        Returns
        -------
        NDArray
            Log-density for each particle, shape (n_particles,).
        """
        if self.variant in ("basic", "leverage"):
            h = state[:, 0]
            vol = np.exp(h / 2.0)
            log_dens = -0.5 * np.log(2.0 * np.pi) - np.log(vol) - 0.5 * (
                float(y) / vol
            ) ** 2
            return log_dens

        elif self.variant == "jumps":
            h = state[:, 0]
            q = state[:, 1]
            mu_j = self.params["mu_jump"]
            sigma_j = self.params["sigma_jump"]
            vol = np.exp(h / 2.0)
            y_val = float(y)

            # When q=0: y ~ N(0, vol^2)
            # When q=1: y ~ N(mu_j, vol^2 + sigma_j^2)
            var_no_jump = vol**2
            var_jump = vol**2 + sigma_j**2
            mean_jump = mu_j

            log_dens_no = (
                -0.5 * np.log(2 * np.pi * var_no_jump)
                - 0.5 * y_val**2 / var_no_jump
            )
            log_dens_yes = (
                -0.5 * np.log(2 * np.pi * var_jump)
                - 0.5 * (y_val - mean_jump) ** 2 / var_jump
            )
            log_dens = np.where(q > 0.5, log_dens_yes, log_dens_no)
            return log_dens

        elif self.variant == "factor":
            y_arr = np.atleast_1d(y)
            h_common = state[:, 0]
            n = state.shape[0]
            log_dens = np.zeros(n)
            for k in range(self.k_factor_series):
                beta_k = self.params[f"beta_{k}"]
                hk = state[:, 1 + k]
                vol_k = np.exp((beta_k * h_common + hk) / 2.0)
                log_dens += (
                    -0.5 * np.log(2 * np.pi)
                    - np.log(vol_k)
                    - 0.5 * (y_arr[k] / vol_k) ** 2
                )
            return log_dens

        raise ValueError(f"Unknown variant: {self.variant}")

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
            Keys: 'observations' shape (T, k_obs),
                  'states' shape (T, k_states).
        """
        rng = np.random.default_rng(seed)
        mu = self.params["mu"]
        phi = self.params["phi"]
        sigma = self.params["sigma"]
        var_stationary = sigma**2 / (1.0 - phi**2)

        if self.variant == "basic":
            h = np.zeros(T)
            y = np.zeros(T)
            h[0] = rng.normal(mu, np.sqrt(var_stationary))
            y[0] = np.exp(h[0] / 2.0) * rng.standard_normal()
            for t in range(1, T):
                h[t] = mu + phi * (h[t - 1] - mu) + sigma * rng.standard_normal()
                y[t] = np.exp(h[t] / 2.0) * rng.standard_normal()
            return {
                "observations": y.reshape(-1, 1),
                "states": h.reshape(-1, 1),
            }

        elif self.variant == "leverage":
            rho = self.params["rho"]
            h = np.zeros(T)
            y = np.zeros(T)
            h[0] = rng.normal(mu, np.sqrt(var_stationary))
            eps0 = rng.standard_normal()
            y[0] = np.exp(h[0] / 2.0) * eps0
            for t in range(1, T):
                eps = rng.standard_normal()
                eta = rho * eps + np.sqrt(1.0 - rho**2) * rng.standard_normal()
                h[t] = mu + phi * (h[t - 1] - mu) + sigma * eta
                y[t] = np.exp(h[t] / 2.0) * eps
            return {
                "observations": y.reshape(-1, 1),
                "states": h.reshape(-1, 1),
            }

        elif self.variant == "jumps":
            lam = self.params["lambda_jump"]
            mu_j = self.params["mu_jump"]
            sigma_j = self.params["sigma_jump"]
            h = np.zeros(T)
            q = np.zeros(T)
            y = np.zeros(T)
            h[0] = rng.normal(mu, np.sqrt(var_stationary))
            q[0] = rng.binomial(1, lam)
            jump = q[0] * rng.normal(mu_j, sigma_j)
            y[0] = np.exp(h[0] / 2.0) * rng.standard_normal() + jump
            for t in range(1, T):
                h[t] = mu + phi * (h[t - 1] - mu) + sigma * rng.standard_normal()
                q[t] = rng.binomial(1, lam)
                jump = q[t] * rng.normal(mu_j, sigma_j)
                y[t] = np.exp(h[t] / 2.0) * rng.standard_normal() + jump
            return {
                "observations": y.reshape(-1, 1),
                "states": np.column_stack([h, q]),
            }

        elif self.variant == "factor":
            K = self.k_factor_series
            h_common = np.zeros(T)
            h_idio = np.zeros((T, K))
            y = np.zeros((T, K))
            h_common[0] = rng.normal(mu, np.sqrt(var_stationary))
            for k in range(K):
                phi_k = self.params[f"phi_{k}"]
                sigma_k = self.params[f"sigma_{k}"]
                var_k = sigma_k**2 / (1.0 - phi_k**2)
                h_idio[0, k] = rng.normal(0, np.sqrt(var_k))
                beta_k = self.params[f"beta_{k}"]
                vol = np.exp(
                    (beta_k * h_common[0] + h_idio[0, k]) / 2.0
                )
                y[0, k] = vol * rng.standard_normal()
            for t in range(1, T):
                h_common[t] = (
                    mu
                    + phi * (h_common[t - 1] - mu)
                    + sigma * rng.standard_normal()
                )
                for k in range(K):
                    phi_k = self.params[f"phi_{k}"]
                    sigma_k = self.params[f"sigma_{k}"]
                    h_idio[t, k] = (
                        phi_k * h_idio[t - 1, k]
                        + sigma_k * rng.standard_normal()
                    )
                    beta_k = self.params[f"beta_{k}"]
                    vol = np.exp(
                        (beta_k * h_common[t] + h_idio[t, k]) / 2.0
                    )
                    y[t, k] = vol * rng.standard_normal()
            states = np.column_stack([h_common.reshape(-1, 1), h_idio])
            return {"observations": y, "states": states}

        raise ValueError(f"Unknown variant: {self.variant}")
