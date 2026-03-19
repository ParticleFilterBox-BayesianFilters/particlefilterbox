"""Tests for the Ensemble Particle Filter."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.ensemble import EnsemblePF


# ---------------------------------------------------------------------------
# Test model: Linear Gaussian
# ---------------------------------------------------------------------------


class LinearGaussianModel(ParticleFilterModel):
    """Linear Gaussian model for testing EnsemblePF."""

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
        return rng.normal(0, 1, size=(n_particles, self.k_states))

    def transition(
        self, particles: np.ndarray, t: int, rng: np.random.Generator
    ) -> np.ndarray:
        return self.phi * particles + rng.normal(0, self.sigma_x, size=particles.shape)

    def observation_function(self, x: np.ndarray, t: int) -> np.ndarray:
        return np.atleast_1d(x)[:self.k_obs]

    def R(self, t: int) -> np.ndarray:
        return np.array([[self.sigma_y**2]])

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        diff = particles[:, 0] - y_t[0]
        return (
            -0.5 * diff**2 / self.sigma_y**2
            - 0.5 * np.log(2 * np.pi * self.sigma_y**2)
        )


def generate_data(
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


class TestEnsemblePFRuns:
    """Basic functionality tests."""

    def test_ensemble_runs(self) -> None:
        """Test that EnsemblePF runs without errors."""
        model = LinearGaussianModel()
        config = PFConfig(n_particles=500, ess_threshold=0.5, resampling="systematic", seed=42)
        epf = EnsemblePF(model=model, config=config)

        _, obs = generate_data(T=50)
        result = epf.filter(obs)

        assert result.filtered_means.shape == (50, 1)
        assert np.all(np.isfinite(result.filtered_means))
        assert np.all(np.isfinite(result.ess_history))

    def test_ensemble_with_fraction(self) -> None:
        """Test EnsemblePF with partial ensemble fraction."""
        model = LinearGaussianModel()
        config = PFConfig(n_particles=500, ess_threshold=0.5, seed=42)
        epf = EnsemblePF(model=model, config=config, ensemble_fraction=0.5)

        _, obs = generate_data(T=30)
        result = epf.filter(obs)
        assert np.all(np.isfinite(result.filtered_means))


class TestEnsemblePFConvergence:
    """Convergence tests."""

    def test_ensemble_converges(self) -> None:
        """EnsemblePF should converge on linear Gaussian (corr > 0.95)."""
        model = LinearGaussianModel(sigma_y=0.3)
        config = PFConfig(
            n_particles=2000, ess_threshold=0.5, resampling="systematic", seed=42
        )
        epf = EnsemblePF(model=model, config=config)

        true_states, obs = generate_data(T=100, sigma_y=0.3, seed=42)
        result = epf.filter(obs)

        estimated = result.filtered_means[:, 0]
        corr = np.corrcoef(true_states, estimated)[0, 1]
        assert corr > 0.95, f"Correlation {corr:.4f} < 0.95"


class TestEnsemblePFKalmanGain:
    """Tests for Kalman gain computation."""

    def test_kalman_gain_shape(self) -> None:
        """Kalman gain should have correct shape."""
        model = LinearGaussianModel()
        config = PFConfig(n_particles=100, ess_threshold=0.5, seed=42)
        epf = EnsemblePF(model=model, config=config)

        rng = np.random.default_rng(42)
        particles = rng.normal(0, 1, size=(100, 1))
        y_pred = particles.copy()
        R = np.array([[1.0]])

        K = epf._compute_kalman_gain(particles, y_pred, R)
        assert K.shape == (1, 1)

    def test_ensemble_update_modifies_particles(self) -> None:
        """Ensemble update should modify particle positions."""
        model = LinearGaussianModel()
        config = PFConfig(n_particles=100, ess_threshold=0.5, seed=42)
        epf = EnsemblePF(model=model, config=config)

        rng = np.random.default_rng(123)
        particles = rng.normal(0, 1, size=(100, 1))
        observation = np.array([5.0])

        updated = epf._ensemble_update(particles, observation, t=0)

        # Particles should have moved toward observation
        assert np.mean(updated) > np.mean(particles)
        assert not np.allclose(particles, updated)
