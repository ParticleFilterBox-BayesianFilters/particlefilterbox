"""Tests for the Guided Particle Filter."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.guided import GuidedPF


# ---------------------------------------------------------------------------
# Test model: Linear Gaussian with full interface for GuidedPF
# ---------------------------------------------------------------------------


class LinearGaussianModelGuided(ParticleFilterModel):
    """Linear Gaussian model with full interface for GuidedPF."""

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

    def transition_function(self, x: np.ndarray, t: int) -> np.ndarray:
        return self.phi * np.atleast_1d(x)

    def transition_mean(self, particles: np.ndarray, t: int) -> np.ndarray:
        return self.phi * particles

    def observation_function(self, x: np.ndarray, t: int) -> np.ndarray:
        return np.atleast_1d(x)[:self.k_obs]

    def Q(self, t: int) -> np.ndarray:
        return np.array([[self.sigma_x**2]])

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


class TestGuidedPFRuns:
    """Basic functionality tests for all modes."""

    def test_guided_linearization_runs(self) -> None:
        """Test GuidedPF with linearization mode."""
        model = LinearGaussianModelGuided()
        config = PFConfig(
            n_particles=500, ess_threshold=0.5, resampling="systematic", seed=42
        )
        gpf = GuidedPF(model=model, config=config, guide_mode="linearization")

        assert gpf.guide_mode == "linearization"

        _, obs = generate_data(T=30)
        result = gpf.filter(obs)

        assert result.filtered_means.shape == (30, 1)
        assert np.all(np.isfinite(result.filtered_means))

    def test_guided_mode_finding_runs(self) -> None:
        """Test GuidedPF with mode_finding mode."""
        model = LinearGaussianModelGuided()
        config = PFConfig(
            n_particles=500, ess_threshold=0.5, resampling="systematic", seed=42
        )
        gpf = GuidedPF(model=model, config=config, guide_mode="mode_finding")

        _, obs = generate_data(T=30)
        result = gpf.filter(obs)

        assert result.filtered_means.shape == (30, 1)
        assert np.all(np.isfinite(result.filtered_means))

    def test_guided_nudging_runs(self) -> None:
        """Test GuidedPF with nudging mode."""
        model = LinearGaussianModelGuided()
        config = PFConfig(
            n_particles=500, ess_threshold=0.5, resampling="systematic", seed=42
        )
        gpf = GuidedPF(
            model=model,
            config=config,
            guide_mode="nudging",
            nudge_factor=0.3,
        )

        _, obs = generate_data(T=30)
        result = gpf.filter(obs)

        assert result.filtered_means.shape == (30, 1)
        assert np.all(np.isfinite(result.filtered_means))

    def test_invalid_mode_raises(self) -> None:
        """Invalid guide mode should raise ValueError."""
        model = LinearGaussianModelGuided()
        config = PFConfig(n_particles=100, seed=42)
        with pytest.raises(ValueError, match="Unsupported guide_mode"):
            GuidedPF(model=model, config=config, guide_mode="invalid")  # type: ignore[arg-type]


class TestGuidedPFConvergence:
    """Convergence tests."""

    def test_guided_linearization_converges(self) -> None:
        """Linearization mode should converge (corr > 0.95)."""
        model = LinearGaussianModelGuided(sigma_y=0.3)
        config = PFConfig(
            n_particles=2000, ess_threshold=0.5, resampling="systematic", seed=42
        )
        gpf = GuidedPF(model=model, config=config, guide_mode="linearization")

        true_states, obs = generate_data(T=100, sigma_y=0.3, seed=42)
        result = gpf.filter(obs)

        estimated = result.filtered_means[:, 0]
        corr = np.corrcoef(true_states, estimated)[0, 1]
        assert corr > 0.95, f"Correlation {corr:.4f} < 0.95"

    def test_guided_nudging_converges(self) -> None:
        """Nudging mode should converge (corr > 0.90)."""
        model = LinearGaussianModelGuided(sigma_y=0.3)
        config = PFConfig(
            n_particles=2000, ess_threshold=0.5, resampling="systematic", seed=42
        )
        gpf = GuidedPF(
            model=model,
            config=config,
            guide_mode="nudging",
            nudge_factor=0.3,
        )

        true_states, obs = generate_data(T=100, sigma_y=0.3, seed=42)
        result = gpf.filter(obs)

        estimated = result.filtered_means[:, 0]
        corr = np.corrcoef(true_states, estimated)[0, 1]
        assert corr > 0.90, f"Correlation {corr:.4f} < 0.90"


class TestGuidedPFProposal:
    """Tests for proposal guidance."""

    def test_linearization_shifts_toward_obs(self) -> None:
        """Linearization should shift proposal toward observation."""
        model = LinearGaussianModelGuided()
        config = PFConfig(n_particles=100, ess_threshold=0.5, seed=42)
        gpf = GuidedPF(model=model, config=config, guide_mode="linearization")

        x_pred = np.array([0.0])
        observation = np.array([5.0])

        mean, cov = gpf._guide_linearization(x_pred, observation, t=0)

        # Guided mean should be shifted toward observation (y=5)
        assert mean[0] > x_pred[0], "Guided mean should shift toward observation"
        assert cov.shape == (1, 1)
        assert cov[0, 0] > 0

    def test_nudging_shifts_proportionally(self) -> None:
        """Nudging should shift by nudge_factor * innovation."""
        model = LinearGaussianModelGuided()
        config = PFConfig(n_particles=100, ess_threshold=0.5, seed=42)

        gpf_weak = GuidedPF(
            model=model,
            config=config,
            guide_mode="nudging",
            nudge_factor=0.1,
        )
        gpf_strong = GuidedPF(
            model=model,
            config=config,
            guide_mode="nudging",
            nudge_factor=0.9,
        )

        x_pred = np.array([0.0])
        observation = np.array([5.0])

        mean_weak, _ = gpf_weak._guide_nudging(x_pred, observation, t=0)
        mean_strong, _ = gpf_strong._guide_nudging(x_pred, observation, t=0)

        # Stronger nudge should shift more
        assert abs(mean_strong[0]) > abs(mean_weak[0])
