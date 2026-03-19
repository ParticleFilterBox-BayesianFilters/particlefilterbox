"""Tests for ParticleFilterModel ABC."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.core.model import ParticleFilterModel


class TestParticleFilterModel:
    def test_abc_enforcement(self) -> None:
        """Cannot instantiate ParticleFilterModel directly."""
        with pytest.raises(TypeError):
            ParticleFilterModel()  # type: ignore[abstract]

    def test_subclass_without_methods(self) -> None:
        """Subclass without all abstract methods -> TypeError."""

        class IncompleteModel(ParticleFilterModel):
            k_states = 1
            k_obs = 1

            def transition(self, particles, t, rng):
                return particles

            # Missing: log_observation_likelihood, initial_distribution

        with pytest.raises(TypeError):
            IncompleteModel()  # type: ignore[abstract]

    def test_complete_subclass(self) -> None:
        """Complete subclass can be instantiated."""

        class SimpleModel(ParticleFilterModel):
            k_states = 1
            k_obs = 1

            def transition(self, particles, t, rng):
                return particles + rng.standard_normal(particles.shape)

            def log_observation_likelihood(self, particles, y_t, t):
                return -0.5 * np.sum((y_t - particles) ** 2, axis=1)

            def initial_distribution(self, n_particles, rng):
                return rng.standard_normal((n_particles, 1))

        model = SimpleModel()
        assert model.k_states == 1
        assert model.k_obs == 1
        assert model.has_linear_substate() is False

    def test_default_proposal_uses_transition(self) -> None:
        """Default proposal delegates to transition."""

        class SimpleModel(ParticleFilterModel):
            k_states = 1
            k_obs = 1

            def transition(self, particles, t, rng):
                return particles + 0.1 * rng.standard_normal(particles.shape)

            def log_observation_likelihood(self, particles, y_t, t):
                return -0.5 * np.sum((y_t - particles) ** 2, axis=1)

            def initial_distribution(self, n_particles, rng):
                return rng.standard_normal((n_particles, 1))

        model = SimpleModel()
        rng = np.random.default_rng(42)
        particles = np.zeros((10, 1))
        new_p, log_q = model.proposal(particles, np.array([0.0]), 0, rng)
        assert new_p.shape == (10, 1)
        assert log_q.shape == (10,)
        np.testing.assert_array_equal(log_q, np.zeros(10))

    def test_log_transition_density_not_implemented(self) -> None:
        """Default log_transition_density raises NotImplementedError."""

        class SimpleModel(ParticleFilterModel):
            k_states = 1
            k_obs = 1

            def transition(self, particles, t, rng):
                return particles

            def log_observation_likelihood(self, particles, y_t, t):
                return np.zeros(particles.shape[0])

            def initial_distribution(self, n_particles, rng):
                return np.zeros((n_particles, 1))

        model = SimpleModel()
        with pytest.raises(NotImplementedError):
            model.log_transition_density(np.zeros((10, 1)), np.zeros((10, 1)), 0)

    def test_param_names_and_params(self) -> None:
        """Test params property."""

        class ModelWithParams(ParticleFilterModel):
            k_states = 1
            k_obs = 1

            def __init__(self, mu: float = 0.0, sigma: float = 1.0):
                self.mu = mu
                self.sigma = sigma

            @property
            def params(self) -> dict:
                return {"mu": self.mu, "sigma": self.sigma}

            def transition(self, particles, t, rng):
                return particles

            def log_observation_likelihood(self, particles, y_t, t):
                return np.zeros(particles.shape[0])

            def initial_distribution(self, n_particles, rng):
                return np.zeros((n_particles, 1))

        model = ModelWithParams(mu=1.0, sigma=2.0)
        assert model.param_names == ["mu", "sigma"]
        assert model.params == {"mu": 1.0, "sigma": 2.0}
