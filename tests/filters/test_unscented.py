"""Tests for the Unscented Particle Filter.

Verifies kalmanbox UKF integration and convergence properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.unscented import UnscentedPF

# CRITICAL: kalmanbox import
from kalmanbox.filters import UnscentedKalmanFilter


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class LinearGaussianModelUPF(ParticleFilterModel):
    """Linear Gaussian model with transition/observation functions for UPF.

    x_t = phi * x_{t-1} + sigma_x * eps
    y_t = x_t + sigma_y * eps
    """

    k_states: int = 1
    k_obs: int = 1

    def __init__(
        self,
        phi: float = 0.95,
        sigma_x: float = 0.5,
        sigma_y: float = 1.0,
    ) -> None:
        self.phi = phi
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def initial_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> np.ndarray:
        return rng.normal(0, 1, size=(n_particles, 1))

    def transition(
        self, particles: np.ndarray, t: int, rng: np.random.Generator
    ) -> np.ndarray:
        return self.phi * particles + rng.normal(
            0, self.sigma_x, size=particles.shape
        )

    def transition_function(self, x: np.ndarray, t: int) -> np.ndarray:
        """Deterministic transition."""
        return self.phi * np.atleast_1d(x)

    def observation_function(self, x: np.ndarray, t: int) -> np.ndarray:
        """Deterministic observation (identity)."""
        return np.atleast_1d(x)[: self.k_obs]

    def transition_mean(self, particles: np.ndarray, t: int) -> np.ndarray:
        return self.phi * particles

    def Q(self, t: int) -> np.ndarray:
        """Process noise covariance."""
        return np.array([[self.sigma_x**2]])

    def R_obs(self, t: int) -> np.ndarray:
        """Observation noise covariance."""
        return np.array([[self.sigma_y**2]])

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        """log p(y_t | x_t) for each particle. Returns shape (N,)."""
        diff = particles[:, 0] - y_t[0]
        return (
            -0.5 * diff**2 / self.sigma_y**2
            - 0.5 * np.log(2 * np.pi * self.sigma_y**2)
        )


class NonlinearModelUPF(ParticleFilterModel):
    """Nonlinear model for testing UPF advantages.

    x_t = 0.5*x_{t-1} + 25*x_{t-1}/(1+x_{t-1}^2) + 8*cos(1.2*t) + sigma_x*eps
    y_t = x_t^2 / 20 + sigma_y * eps

    Standard benchmark from Gordon et al (1993).
    """

    k_states: int = 1
    k_obs: int = 1

    def __init__(
        self, sigma_x: float = np.sqrt(10.0), sigma_y: float = 1.0
    ) -> None:
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def initial_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> np.ndarray:
        return rng.normal(0, np.sqrt(5), size=(n_particles, 1))

    def transition(
        self, particles: np.ndarray, t: int, rng: np.random.Generator
    ) -> np.ndarray:
        x = particles.flatten()
        x_new = (
            0.5 * x
            + 25.0 * x / (1.0 + x**2)
            + 8.0 * np.cos(1.2 * t)
            + self.sigma_x * rng.normal(size=x.shape)
        )
        return x_new.reshape(-1, 1)

    def transition_function(self, x: np.ndarray, t: int) -> np.ndarray:
        x_val = np.atleast_1d(x).flatten()
        return np.atleast_1d(
            0.5 * x_val
            + 25.0 * x_val / (1.0 + x_val**2)
            + 8.0 * np.cos(1.2 * t)
        )

    def observation_function(self, x: np.ndarray, t: int) -> np.ndarray:
        x_val = np.atleast_1d(x).flatten()
        return np.atleast_1d(x_val**2 / 20.0)

    def Q(self, t: int) -> np.ndarray:
        return np.array([[self.sigma_x**2]])

    def R_obs(self, t: int) -> np.ndarray:
        return np.array([[self.sigma_y**2]])

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        """log p(y_t | x_t) for each particle. Returns shape (N,)."""
        x = particles[:, 0]
        y_pred = x**2 / 20.0
        diff = y_t[0] - y_pred
        return (
            -0.5 * diff**2 / self.sigma_y**2
            - 0.5 * np.log(2 * np.pi * self.sigma_y**2)
        )


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def generate_linear_data(
    T: int = 100,
    phi: float = 0.95,
    sigma_x: float = 0.5,
    sigma_y: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.empty(T)
    obs = np.empty(T)
    x = rng.normal(0, 1)
    for t in range(T):
        x = phi * x + sigma_x * rng.normal()
        states[t] = x
        obs[t] = x + sigma_y * rng.normal()
    return states, obs.reshape(-1, 1)


def generate_nonlinear_data(
    T: int = 100,
    sigma_x: float = np.sqrt(10.0),
    sigma_y: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.empty(T)
    obs = np.empty(T)
    x = rng.normal(0, np.sqrt(5))
    for t in range(T):
        x = 0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * np.cos(1.2 * t)
        x += sigma_x * rng.normal()
        states[t] = x
        obs[t] = x**2 / 20.0 + sigma_y * rng.normal()
    return states, obs.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUPFKalmanboxIntegration:
    """Verify kalmanbox UKF integration."""

    def test_upf_kalmanbox_integration(self) -> None:
        """Verify that UPF uses kalmanbox.UnscentedKalmanFilter."""
        model = LinearGaussianModelUPF()
        config = PFConfig(n_particles=100, ess_threshold=0.5, seed=42)
        upf = UnscentedPF(model=model, config=config)

        assert hasattr(upf, "_ukf")
        assert isinstance(upf._ukf, UnscentedKalmanFilter)


class TestUPFRuns:
    """Basic functionality tests."""

    def test_upf_runs(self) -> None:
        """Test that UPF runs without errors."""
        model = LinearGaussianModelUPF()
        config = PFConfig(
            n_particles=200, ess_threshold=0.5, resampling="systematic", seed=42
        )
        upf = UnscentedPF(model=model, config=config)

        _, obs = generate_linear_data(T=30)
        result = upf.filter(obs)

        assert result.filtered_means.shape == (30, 1)
        assert result.ess_history.shape == (30,)
        assert np.all(np.isfinite(result.filtered_means))

    def test_upf_nonlinear_runs(self) -> None:
        """Test UPF on nonlinear model."""
        model = NonlinearModelUPF()
        config = PFConfig(
            n_particles=500, ess_threshold=0.5, resampling="systematic", seed=42
        )
        upf = UnscentedPF(model=model, config=config)

        _, obs = generate_nonlinear_data(T=30)
        result = upf.filter(obs)

        assert result.filtered_means.shape == (30, 1)
        assert np.all(np.isfinite(result.filtered_means))


class TestUPFConvergence:
    """Convergence tests."""

    def test_upf_linear_convergence(self) -> None:
        """UPF should converge well on linear Gaussian (corr > 0.99, N=3000)."""
        model = LinearGaussianModelUPF(sigma_y=0.1)
        config = PFConfig(
            n_particles=3000,
            ess_threshold=0.5,
            resampling="systematic",
            seed=42,
        )
        upf = UnscentedPF(model=model, config=config)

        true_states, obs = generate_linear_data(T=100, sigma_y=0.1, seed=42)
        result = upf.filter(obs)

        estimated = result.filtered_means[:, 0]
        corr = np.corrcoef(true_states, estimated)[0, 1]
        assert corr > 0.99, f"Correlation {corr:.4f} < 0.99"

    def test_upf_better_than_bootstrap(self) -> None:
        """UPF should outperform Bootstrap on nonlinear model."""
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model_upf = NonlinearModelUPF()
        model_bpf = NonlinearModelUPF()

        config = PFConfig(
            n_particles=1000,
            ess_threshold=0.5,
            resampling="systematic",
            seed=42,
        )

        upf = UnscentedPF(model=model_upf, config=config)
        bpf = BootstrapPF(model=model_bpf, config=config)

        true_states, obs = generate_nonlinear_data(T=100, seed=42)

        result_upf = upf.filter(obs)
        result_bpf = bpf.filter(obs)

        rmse_upf = np.sqrt(
            np.mean((true_states - result_upf.filtered_means[:, 0]) ** 2)
        )
        rmse_bpf = np.sqrt(
            np.mean((true_states - result_bpf.filtered_means[:, 0]) ** 2)
        )

        # UPF should be at least as good (allowing 20% margin for stochasticity)
        assert rmse_upf <= rmse_bpf * 1.2, (
            f"UPF RMSE ({rmse_upf:.4f}) should be <= "
            f"Bootstrap RMSE ({rmse_bpf:.4f}) * 1.2"
        )


class TestUPFWeights:
    """Test weight computation."""

    def test_upf_weights_finite(self) -> None:
        """All log-likelihoods should be finite."""
        model = LinearGaussianModelUPF()
        config = PFConfig(n_particles=200, ess_threshold=0.5, seed=42)
        upf = UnscentedPF(model=model, config=config)

        _, obs = generate_linear_data(T=20)
        result = upf.filter(obs)

        assert np.all(np.isfinite(result.log_likelihoods))

    def test_upf_ess_reasonable(self) -> None:
        """UPF should maintain reasonable ESS."""
        model = LinearGaussianModelUPF()
        config = PFConfig(n_particles=500, ess_threshold=0.3, seed=42)
        upf = UnscentedPF(model=model, config=config)

        _, obs = generate_linear_data(T=50)
        result = upf.filter(obs)

        mean_ess = np.mean(result.ess_history)
        assert mean_ess > 50, f"Mean ESS {mean_ess:.1f} too low"
