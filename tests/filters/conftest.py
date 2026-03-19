"""Shared fixtures for particle filter tests.

Provides:
- LinearGaussian model fixture for convergence testing
- Stochastic Volatility (SV) model fixture for nonlinear testing
- Common configurations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Linear Gaussian Model (for Kalman comparison)
# ---------------------------------------------------------------------------
# x_t = phi * x_{t-1} + eta_t,  eta_t ~ N(0, sigma_eta^2)
# y_t = x_t + eps_t,            eps_t ~ N(0, sigma_eps^2)


@dataclass
class LinearGaussianParams:
    """Parameters for the linear Gaussian model."""

    phi: float = 0.9
    sigma_eta: float = 1.0
    sigma_eps: float = 0.5
    x0_mean: float = 0.0
    x0_std: float = 1.0


class LinearGaussianModel:
    """Simple linear Gaussian state-space model.

    x_t = phi * x_{t-1} + N(0, sigma_eta^2)
    y_t = x_t + N(0, sigma_eps^2)

    This model is used for testing convergence against the Kalman filter.
    Implements the ParticleFilterModel interface without inheriting from it,
    to avoid coupling test fixtures to the ABC.
    """

    k_states: int = 1
    k_obs: int = 1

    def __init__(self, params: LinearGaussianParams | None = None) -> None:
        self.params = params or LinearGaussianParams()
        self.state_dim = 1
        self.obs_dim = 1

    @property
    def phi(self) -> float:
        return self.params.phi

    @property
    def sigma_eta(self) -> float:
        return self.params.sigma_eta

    @property
    def sigma_eps(self) -> float:
        return self.params.sigma_eps

    def initial_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample from p(x_0)."""
        return rng.normal(
            self.params.x0_mean, self.params.x0_std, size=(n_particles, 1)
        )

    def transition(
        self,
        particles: NDArray[np.float64],
        _t: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample from p(x_t | x_{t-1})."""
        mean = self.phi * particles
        return mean + self.sigma_eta * rng.standard_normal(particles.shape)

    def log_transition_density(
        self,
        x_new: NDArray[np.float64],
        x_old: NDArray[np.float64],
        _t: int,
    ) -> NDArray[np.float64]:
        """Compute log p(x_t | x_{t-1})."""
        mean = self.phi * x_old
        diff = x_new - mean
        return -0.5 * np.sum(diff**2, axis=-1) / self.sigma_eta**2 - 0.5 * np.log(
            2 * np.pi * self.sigma_eta**2
        )

    def log_observation_likelihood(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        _t: int,
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t)."""
        diff = y_t - particles
        return -0.5 * np.sum(diff**2, axis=-1) / self.sigma_eps**2 - 0.5 * np.log(
            2 * np.pi * self.sigma_eps**2
        )

    def simulate(
        self,
        n_steps: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulate states and observations.

        Returns
        -------
        tuple[NDArray, NDArray]
            (states, observations) each of shape (T,)
        """
        if rng is None:
            rng = np.random.default_rng()

        states = np.zeros(n_steps)
        obs = np.zeros(n_steps)

        states[0] = rng.normal(self.params.x0_mean, self.params.x0_std)
        obs[0] = states[0] + self.sigma_eps * rng.standard_normal()

        for t in range(1, n_steps):
            states[t] = self.phi * states[t - 1] + self.sigma_eta * rng.standard_normal()
            obs[t] = states[t] + self.sigma_eps * rng.standard_normal()

        return states, obs


# ---------------------------------------------------------------------------
# Kalman Filter (reference implementation for testing)
# ---------------------------------------------------------------------------


def kalman_filter(
    y: NDArray[np.float64],
    phi: float,
    sigma_eta: float,
    sigma_eps: float,
    x0_mean: float = 0.0,
    x0_var: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Run the Kalman filter for the linear Gaussian model.

    Parameters
    ----------
    y : NDArray[np.float64]
        Observations of shape (T,).
    phi : float
        AR coefficient.
    sigma_eta : float
        State noise std.
    sigma_eps : float
        Observation noise std.
    x0_mean : float
        Prior mean for x_0.
    x0_var : float
        Prior variance for x_0.

    Returns
    -------
    tuple[NDArray, NDArray, float]
        (filtered_means, filtered_vars, log_likelihood)
    """
    n_obs = len(y)
    filtered_means = np.zeros(n_obs)
    filtered_vars = np.zeros(n_obs)

    q_var = sigma_eta**2
    r_var = sigma_eps**2

    log_likelihood = 0.0

    # Initialize
    x_pred = x0_mean
    p_pred = x0_var

    for t in range(n_obs):
        if t > 0:
            # Predict
            x_pred = phi * filtered_means[t - 1]
            p_pred = phi**2 * filtered_vars[t - 1] + q_var

        # Update
        s_innov = p_pred + r_var  # innovation variance
        k_gain = p_pred / s_innov  # Kalman gain

        innovation = y[t] - x_pred
        filtered_means[t] = x_pred + k_gain * innovation
        filtered_vars[t] = (1 - k_gain) * p_pred

        # Log-likelihood increment
        log_likelihood += -0.5 * (
            np.log(2 * np.pi * s_innov) + innovation**2 / s_innov
        )

    return filtered_means, filtered_vars, log_likelihood


# ---------------------------------------------------------------------------
# Stochastic Volatility Model
# ---------------------------------------------------------------------------


@dataclass
class SVParams:
    """Parameters for the stochastic volatility model."""

    mu: float = -1.0
    phi: float = 0.97
    sigma_eta: float = 0.15
    x0_mean: float = -1.0
    x0_std: float = 0.5


class StochasticVolatilityModel:
    """Stochastic Volatility model.

    x_t = mu + phi * (x_{t-1} - mu) + sigma_eta * eta_t
    y_t = exp(x_t / 2) * eps_t

    where eta_t, eps_t ~ N(0, 1).

    Implements the ParticleFilterModel interface without inheriting from it,
    to avoid coupling test fixtures to the ABC.
    """

    k_states: int = 1
    k_obs: int = 1

    def __init__(self, params: SVParams | None = None) -> None:
        self.params = params or SVParams()
        self.state_dim = 1
        self.obs_dim = 1

    def initial_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample from p(x_0)."""
        return rng.normal(self.params.x0_mean, self.params.x0_std, size=(n_particles, 1))

    def transition(
        self,
        particles: NDArray[np.float64],
        _t: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample from p(x_t | x_{t-1})."""
        mu = self.params.mu
        phi = self.params.phi
        mean = mu + phi * (particles - mu)
        return mean + self.params.sigma_eta * rng.standard_normal(particles.shape)

    def log_transition_density(
        self,
        x_new: NDArray[np.float64],
        x_old: NDArray[np.float64],
        _t: int,
    ) -> NDArray[np.float64]:
        """Compute log p(x_t | x_{t-1})."""
        mu = self.params.mu
        phi = self.params.phi
        sigma = self.params.sigma_eta
        mean = mu + phi * (x_old - mu)
        diff = x_new - mean
        return -0.5 * np.sum(diff**2, axis=-1) / sigma**2 - 0.5 * np.log(
            2 * np.pi * sigma**2
        )

    def log_observation_likelihood(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        _t: int,
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t).

        y_t ~ N(0, exp(x_t)) => log p(y_t|x_t) = -0.5*x_t - 0.5*y_t^2/exp(x_t) - 0.5*log(2*pi)
        """
        # x_t is log-variance (h_t)
        h = particles.squeeze(-1) if particles.ndim > 1 else particles
        y = y_t.squeeze(-1) if y_t.ndim > 1 else y_t
        return -0.5 * h - 0.5 * y**2 * np.exp(-h) - 0.5 * np.log(2 * np.pi)

    def simulate(
        self,
        n_steps: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulate states (log-volatility) and observations.

        Returns
        -------
        tuple[NDArray, NDArray]
            (h_states, observations) each of shape (n_steps,)
        """
        if rng is None:
            rng = np.random.default_rng()

        mu = self.params.mu
        phi = self.params.phi
        sigma = self.params.sigma_eta

        h = np.zeros(n_steps)
        y = np.zeros(n_steps)

        h[0] = rng.normal(self.params.x0_mean, self.params.x0_std)
        y[0] = np.exp(h[0] / 2) * rng.standard_normal()

        for t in range(1, n_steps):
            h[t] = mu + phi * (h[t - 1] - mu) + sigma * rng.standard_normal()
            y[t] = np.exp(h[t] / 2) * rng.standard_normal()

        return h, y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_gaussian_model() -> LinearGaussianModel:
    """Linear Gaussian model with default parameters."""
    return LinearGaussianModel()


@pytest.fixture
def sv_model() -> StochasticVolatilityModel:
    """Stochastic volatility model with default parameters."""
    return StochasticVolatilityModel()


@pytest.fixture
def linear_gaussian_data() -> (
    tuple[LinearGaussianModel, NDArray[np.float64], NDArray[np.float64]]
):
    """Generate linear Gaussian data for testing.

    Returns (model, states, observations) with T=200.
    """
    rng = np.random.default_rng(42)
    model = LinearGaussianModel()
    states, obs = model.simulate(n_steps=200, rng=rng)
    return model, states, obs


@pytest.fixture
def sv_data() -> (
    tuple[StochasticVolatilityModel, NDArray[np.float64], NDArray[np.float64]]
):
    """Generate stochastic volatility data for testing.

    Returns (model, h_states, observations) with T=500.
    """
    rng = np.random.default_rng(123)
    model = StochasticVolatilityModel()
    h, y = model.simulate(n_steps=500, rng=rng)
    return model, h, y


@pytest.fixture
def pf_config():
    """Default particle filter configuration."""
    from particlefilterbox.core.config import PFConfig

    return PFConfig(n_particles=1000, seed=42, ess_threshold=0.5)


@pytest.fixture
def pf_config_large():
    """Large particle count configuration for convergence tests."""
    from particlefilterbox.core.config import PFConfig

    return PFConfig(n_particles=5000, seed=42, ess_threshold=0.5)
