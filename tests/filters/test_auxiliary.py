"""Tests for the Auxiliary Particle Filter."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.auxiliary import AuxiliaryPF


# ---------------------------------------------------------------------------
# Test model: Linear Gaussian
# ---------------------------------------------------------------------------

class LinearGaussianModel(ParticleFilterModel):
    """Simple linear Gaussian model for testing.

    x_t = phi * x_{t-1} + sigma_x * eps_x
    y_t = x_t + sigma_y * eps_y
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

    def transition_mean(self, particles: np.ndarray, t: int) -> np.ndarray:
        """Deterministic transition mean (no noise)."""
        return self.phi * particles

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        """log p(y_t | x_t) for each particle. Returns shape (N,)."""
        # particles: (N, 1), y_t: (1,) or (k_obs,)
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

class TestAuxiliaryPFRuns:
    """Basic functionality tests."""

    def test_auxiliary_runs(self) -> None:
        """Test that AuxiliaryPF runs without errors."""
        model = LinearGaussianModel()
        config = PFConfig(n_particles=500, ess_threshold=0.5, resampling="systematic", seed=42)
        apf = AuxiliaryPF(model=model, config=config)

        _, obs = generate_data(T=50)
        result = apf.filter(obs)

        assert result.filtered_means.shape == (50, 1)
        assert result.ess_history.shape == (50,)
        assert result.log_likelihoods.shape == (50,)
        assert np.all(np.isfinite(result.filtered_means))
        assert np.all(np.isfinite(result.ess_history))

    def test_auxiliary_with_transition_mean(self) -> None:
        """Test APF with model providing transition_mean."""
        model = LinearGaussianModel()
        assert hasattr(model, "transition_mean")

        config = PFConfig(n_particles=500, ess_threshold=0.5, seed=42)
        apf = AuxiliaryPF(model=model, config=config)
        assert apf._has_transition_mean is True

        _, obs = generate_data(T=20)
        result = apf.filter(obs)
        assert np.all(np.isfinite(result.filtered_means))

    def test_auxiliary_without_transition_mean(self) -> None:
        """Test APF without transition_mean (fallback)."""

        class ModelNoMean(LinearGaussianModel):
            # Override transition_mean to hide it
            transition_mean = None  # type: ignore[assignment]

        # Make it non-callable so APF fallback triggers
        model = ModelNoMean()
        assert not callable(getattr(model, "transition_mean", None))

        config = PFConfig(n_particles=500, ess_threshold=0.5, seed=42)
        apf = AuxiliaryPF(model=model, config=config)
        assert apf._has_transition_mean is False

        _, obs = generate_data(T=20)
        result = apf.filter(obs)
        assert np.all(np.isfinite(result.filtered_means))


# ---------------------------------------------------------------------------
# Convergence and quality tests
# ---------------------------------------------------------------------------

class TestAuxiliaryPFConvergence:
    """Convergence and quality tests."""

    def test_auxiliary_linear_convergence(self) -> None:
        """APF should converge well on linear Gaussian model (corr > 0.99)."""
        model = LinearGaussianModel(phi=0.95, sigma_x=1.0, sigma_y=0.2)
        config = PFConfig(n_particles=2000, ess_threshold=0.5, resampling="systematic", seed=42)
        apf = AuxiliaryPF(model=model, config=config)

        true_states, obs = generate_data(T=100, sigma_x=1.0, sigma_y=0.2, seed=42)
        result = apf.filter(obs)

        estimated = result.filtered_means[:, 0]
        correlation = np.corrcoef(true_states, estimated)[0, 1]
        assert correlation > 0.99, f"Correlation {correlation:.4f} < 0.99"

    def test_auxiliary_better_weights(self) -> None:
        """APF should have better ESS than Bootstrap (on average)."""
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model = LinearGaussianModel(phi=0.95, sigma_x=0.5, sigma_y=1.0)
        config_apf = PFConfig(n_particles=1000, ess_threshold=0.5, resampling="systematic", seed=42)
        config_bpf = PFConfig(n_particles=1000, ess_threshold=0.5, resampling="systematic", seed=42)

        apf = AuxiliaryPF(model=model, config=config_apf)
        bpf = BootstrapPF(model=model, config=config_bpf)

        _, obs = generate_data(T=100, seed=42)

        result_apf = apf.filter(obs)
        result_bpf = bpf.filter(obs)

        mean_ess_apf = np.mean(result_apf.ess_history)
        mean_ess_bpf = np.mean(result_bpf.ess_history)

        # APF should generally have higher ESS due to pre-selection
        assert mean_ess_apf > mean_ess_bpf, (
            f"APF ESS ({mean_ess_apf:.1f}) should be > Bootstrap ESS ({mean_ess_bpf:.1f})"
        )


# ---------------------------------------------------------------------------
# First stage weight tests
# ---------------------------------------------------------------------------

class TestAuxiliaryPFFirstStage:
    """Tests for the first-stage weight computation."""

    def test_first_stage_weights_shape(self) -> None:
        """First stage weights should have correct shape."""
        model = LinearGaussianModel()
        config = PFConfig(n_particles=100, ess_threshold=0.5, seed=42)
        apf = AuxiliaryPF(model=model, config=config)

        particles = np.random.default_rng(42).normal(0, 1, size=(100, 1))
        log_weights = np.full(100, -np.log(100))
        observation = np.array([1.0])

        first_stage, log_lambdas = apf._first_stage_weights(
            log_weights, particles, observation, t=0
        )

        assert first_stage.shape == (100,)
        assert log_lambdas.shape == (100,)
        assert np.all(np.isfinite(first_stage))
        assert np.all(np.isfinite(log_lambdas))

    def test_first_stage_prefers_close_particles(self) -> None:
        """First stage should assign higher weight to particles closer to observation."""
        model = LinearGaussianModel(phi=1.0, sigma_y=0.1)
        config = PFConfig(n_particles=100, ess_threshold=0.5, seed=42)
        apf = AuxiliaryPF(model=model, config=config)

        # Particles: half near observation, half far away
        particles = np.zeros((100, 1))
        particles[:50] = 5.0   # Close to observation
        particles[50:] = -5.0  # Far from observation

        log_weights = np.full(100, -np.log(100))
        observation = np.array([5.0])

        _, log_lambdas = apf._first_stage_weights(
            log_weights, particles, observation, t=0
        )

        # Particles near 5.0 should have higher lambda
        mean_near = np.mean(log_lambdas[:50])
        mean_far = np.mean(log_lambdas[50:])
        assert mean_near > mean_far, (
            f"Near particles ({mean_near:.2f}) should have higher lambda "
            f"than far particles ({mean_far:.2f})"
        )
