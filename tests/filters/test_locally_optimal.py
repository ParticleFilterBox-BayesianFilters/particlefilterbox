"""Tests for the Locally Optimal Particle Filter."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.locally_optimal import LocallyOptimalPF


# ---------------------------------------------------------------------------
# Test model: Linear Gaussian with analytic optimal proposal
# ---------------------------------------------------------------------------

class LinearGaussianOptimalModel(ParticleFilterModel):
    """Linear Gaussian model with analytic optimal proposal.

    x_t = phi * x_{t-1} + sigma_x * eps_x
    y_t = x_t + sigma_y * eps_y

    Optimal proposal: N(m, P) where
      P^{-1} = sigma_x^{-2} + sigma_y^{-2}
      m = P * (phi * x_{t-1} / sigma_x^2 + y_t / sigma_y^2)
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
        return rng.normal(0, 1, size=(n_particles, self.k_states))

    def transition(
        self, particles: np.ndarray, t: int, rng: np.random.Generator
    ) -> np.ndarray:
        noise = rng.normal(0, self.sigma_x, size=particles.shape)
        return self.phi * particles + noise

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        diff = particles[:, 0] - y_t[0]
        return (
            -0.5 * diff**2 / self.sigma_y**2
            - 0.5 * np.log(2 * np.pi * self.sigma_y**2)
        )

    def optimal_proposal_params(
        self, particles: np.ndarray, observation: np.ndarray, t: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Analytic optimal proposal parameters for linear Gaussian."""
        n = particles.shape[0]
        prec_x = 1.0 / self.sigma_x**2
        prec_y = 1.0 / self.sigma_y**2
        post_var = 1.0 / (prec_x + prec_y)

        pred_mean = self.phi * particles  # (N, 1)
        y = observation[0]

        means = post_var * (pred_mean * prec_x + y * prec_y)  # (N, 1)
        covs = np.full((n, 1, 1), post_var)

        return means, covs

    def predictive_log_likelihood(
        self, observation: np.ndarray, particles: np.ndarray, t: int
    ) -> float:
        """p(y_t | x_{t-1}) for linear Gaussian."""
        pred_mean = self.phi * particles.flatten()[0]
        pred_var = self.sigma_x**2 + self.sigma_y**2
        diff = observation[0] - pred_mean
        return float(
            -0.5 * diff**2 / pred_var
            - 0.5 * np.log(2 * np.pi * pred_var)
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

class TestLocallyOptimalPFRuns:
    """Basic tests for LocallyOptimalPF."""

    def test_locally_optimal_runs(self) -> None:
        """Test that LocallyOptimalPF runs without errors."""
        model = LinearGaussianOptimalModel()
        config = PFConfig(n_particles=500, ess_threshold=0.5, resampling="systematic", seed=42)
        lopf = LocallyOptimalPF(model=model, config=config)

        _, obs = generate_data(T=50)
        result = lopf.filter(obs)

        assert result.filtered_means.shape == (50, 1)
        assert result.ess_history.shape == (50,)
        assert np.all(np.isfinite(result.filtered_means))
        assert np.all(np.isfinite(result.ess_history))

    def test_locally_optimal_requires_proposal(self) -> None:
        """Test that LocallyOptimalPF raises error without optimal proposal."""

        class BadModel(ParticleFilterModel):
            k_states = 1
            k_obs = 1

            def initial_distribution(
                self, n: int, rng: np.random.Generator
            ) -> np.ndarray:
                return rng.normal(0, 1, (n, 1))

            def transition(
                self, p: np.ndarray, t: int, rng: np.random.Generator
            ) -> np.ndarray:
                return p

            def log_observation_likelihood(
                self, p: np.ndarray, y_t: np.ndarray, t: int
            ) -> np.ndarray:
                return np.zeros(p.shape[0])

        config = PFConfig(seed=42)
        with pytest.raises(ValueError, match="optimal_proposal"):
            LocallyOptimalPF(model=BadModel(), config=config)

    def test_locally_optimal_convergence(self) -> None:
        """Locally optimal PF should converge well (corr > 0.99)."""
        model = LinearGaussianOptimalModel(phi=0.95, sigma_x=1.0, sigma_y=0.2)
        config = PFConfig(n_particles=2000, ess_threshold=0.5, resampling="systematic", seed=42)
        lopf = LocallyOptimalPF(model=model, config=config)

        true_states, obs = generate_data(T=100, sigma_x=1.0, sigma_y=0.2, seed=42)
        result = lopf.filter(obs)

        estimated = result.filtered_means[:, 0]
        corr = np.corrcoef(true_states, estimated)[0, 1]
        assert corr > 0.99, f"Correlation {corr:.4f} < 0.99"


# ---------------------------------------------------------------------------
# Weight quality tests
# ---------------------------------------------------------------------------

class TestLocallyOptimalPFWeights:
    """Tests for weight quality."""

    def test_locally_optimal_high_ess(self) -> None:
        """Optimal proposal should yield high ESS."""
        model = LinearGaussianOptimalModel()
        config = PFConfig(n_particles=1000, ess_threshold=0.3, resampling="systematic", seed=42)
        lopf = LocallyOptimalPF(model=model, config=config)

        _, obs = generate_data(T=50, seed=42)
        result = lopf.filter(obs)

        mean_ess = float(np.mean(result.ess_history))
        max_ess = config.n_particles
        assert mean_ess > 0.5 * max_ess, (
            f"Mean ESS {mean_ess:.1f} should be > {0.5 * max_ess:.1f}"
        )
