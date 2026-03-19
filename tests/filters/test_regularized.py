"""Tests for the Regularized Particle Filter."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.regularized import RegularizedPF


# ---------------------------------------------------------------------------
# Test model: Linear Gaussian
# ---------------------------------------------------------------------------

class LinearGaussianModel(ParticleFilterModel):
    """Simple linear Gaussian model for testing."""

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
    """Generate synthetic data from a linear Gaussian model."""
    rng = np.random.default_rng(seed)
    states = np.empty(T)
    obs = np.empty(T)
    x = rng.normal(0, 1)
    for t in range(T):
        x = phi * x + sigma_x * rng.normal()
        states[t] = x
        obs[t] = x + sigma_y * rng.normal()
    return states, obs.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------

class TestRegularizedPFRuns:
    """Basic functionality tests."""

    def test_regularized_runs_gaussian(self) -> None:
        """Test RegularizedPF runs with Gaussian kernel."""
        model = LinearGaussianModel()
        config = PFConfig(n_particles=500, ess_threshold=0.5, resampling="systematic", seed=42)
        rpf = RegularizedPF(
            model=model, config=config, bandwidth="silverman", kernel="gaussian"
        )

        _, obs = generate_data(T=50)
        result = rpf.filter(obs)

        assert result.filtered_means.shape == (50, 1)
        assert np.all(np.isfinite(result.filtered_means))

    def test_regularized_runs_epanechnikov(self) -> None:
        """Test RegularizedPF runs with Epanechnikov kernel."""
        model = LinearGaussianModel()
        config = PFConfig(n_particles=500, ess_threshold=0.5, resampling="systematic", seed=42)
        rpf = RegularizedPF(
            model=model, config=config, bandwidth="silverman", kernel="epanechnikov"
        )

        _, obs = generate_data(T=50)
        result = rpf.filter(obs)

        assert result.filtered_means.shape == (50, 1)
        assert np.all(np.isfinite(result.filtered_means))

    def test_regularized_fixed_bandwidth(self) -> None:
        """Test RegularizedPF with fixed bandwidth."""
        model = LinearGaussianModel()
        config = PFConfig(n_particles=500, ess_threshold=0.5, seed=42)
        rpf = RegularizedPF(
            model=model, config=config, bandwidth=0.1, kernel="gaussian"
        )

        _, obs = generate_data(T=30)
        result = rpf.filter(obs)
        assert np.all(np.isfinite(result.filtered_means))

    def test_invalid_kernel_raises(self) -> None:
        """Test that invalid kernel raises ValueError."""
        model = LinearGaussianModel()
        config = PFConfig(seed=42)
        with pytest.raises(ValueError, match="Unsupported kernel"):
            RegularizedPF(model=model, config=config, kernel="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

class TestRegularizedPFConvergence:
    """Convergence tests."""

    def test_regularized_converges(self) -> None:
        """RegularizedPF should converge on linear Gaussian (corr > 0.95)."""
        model = LinearGaussianModel(phi=0.95, sigma_x=1.0, sigma_y=0.2)
        config = PFConfig(n_particles=2000, ess_threshold=0.5, resampling="systematic", seed=42)
        rpf = RegularizedPF(
            model=model, config=config, bandwidth="silverman", kernel="gaussian"
        )

        true_states, obs = generate_data(T=100, sigma_x=1.0, sigma_y=0.2, seed=42)
        result = rpf.filter(obs)

        estimated = result.filtered_means[:, 0]
        corr = np.corrcoef(true_states, estimated)[0, 1]
        assert corr > 0.95, f"Correlation {corr:.4f} < 0.95"


# ---------------------------------------------------------------------------
# Silverman bandwidth tests
# ---------------------------------------------------------------------------

class TestRegularizedPFSilverman:
    """Tests for Silverman bandwidth."""

    def test_silverman_bandwidth_formula(self) -> None:
        """Test Silverman bandwidth computation."""
        model = LinearGaussianModel()
        config = PFConfig(seed=42)
        rpf = RegularizedPF(model=model, config=config)

        # h = (4 / (N * (k + 2)))^(1 / (k + 4))
        h = rpf._silverman_bandwidth(1000, 1)
        expected = (4.0 / (1000 * 3)) ** (1.0 / 5)
        assert abs(h - expected) < 1e-10

    def test_silverman_decreases_with_n(self) -> None:
        """Silverman bandwidth should decrease as N increases."""
        model = LinearGaussianModel()
        config = PFConfig(seed=42)
        rpf = RegularizedPF(model=model, config=config)

        h_100 = rpf._silverman_bandwidth(100, 1)
        h_1000 = rpf._silverman_bandwidth(1000, 1)
        h_10000 = rpf._silverman_bandwidth(10000, 1)

        assert h_100 > h_1000 > h_10000


# ---------------------------------------------------------------------------
# Jitter tests
# ---------------------------------------------------------------------------

class TestRegularizedPFJitter:
    """Tests for kernel jittering."""

    def test_jitter_changes_particles(self) -> None:
        """Jittering should modify particle positions."""
        model = LinearGaussianModel()
        config = PFConfig(seed=42)
        rpf = RegularizedPF(model=model, config=config, kernel="gaussian")
        rng = np.random.default_rng(42)

        particles = np.ones((100, 1))
        jittered = rpf._jitter(particles, bandwidth=0.1, rng=rng)

        assert not np.allclose(particles, jittered)
        assert np.all(np.abs(jittered - particles) < 1.0)

    def test_jitter_preserves_mean_approximately(self) -> None:
        """Jittering with zero-mean kernel should preserve mean approximately."""
        model = LinearGaussianModel()
        config = PFConfig(seed=42)
        rpf = RegularizedPF(model=model, config=config, kernel="gaussian")
        rng = np.random.default_rng(42)

        particles = rng.normal(3.0, 1.0, size=(10000, 1))
        original_mean = np.mean(particles, axis=0)

        jitter_rng = np.random.default_rng(123)
        jittered = rpf._jitter(particles, bandwidth=0.1, rng=jitter_rng)
        jittered_mean = np.mean(jittered, axis=0)

        np.testing.assert_allclose(original_mean, jittered_mean, atol=0.05)

    def test_epanechnikov_jitter_bounded(self) -> None:
        """Epanechnikov jitter should be bounded."""
        model = LinearGaussianModel()
        config = PFConfig(seed=42)
        rpf = RegularizedPF(model=model, config=config, kernel="epanechnikov")
        rng = np.random.default_rng(42)

        particles = np.zeros((1000, 1))
        jittered = rpf._jitter(particles, bandwidth=0.1, rng=rng)

        max_perturbation = np.max(np.abs(jittered - particles))
        # Should be bounded by bandwidth * sqrt(k+2)
        bound = 0.1 * np.sqrt(3)  # k=1, so sqrt(3)
        assert max_perturbation < bound * 2  # generous bound
