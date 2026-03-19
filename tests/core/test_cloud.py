"""Tests for ParticleCloud."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.core.cloud import ParticleCloud


class TestParticleCloud:
    def test_uniform_weights_ess(self) -> None:
        """Uniform weights -> ESS = N exactly."""
        cloud = ParticleCloud(n_particles=1000, k_states=1)
        cloud.set_uniform_weights()
        assert cloud.ess == 1000.0

    def test_degenerate_weights_ess(self) -> None:
        """One weight = 1, rest = 0 -> ESS = 1."""
        cloud = ParticleCloud(n_particles=100, k_states=1)
        log_w = np.full(100, -np.inf)
        log_w[0] = 0.0
        cloud.set_log_weights(log_w)
        assert pytest.approx(cloud.ess, abs=0.01) == 1.0

    def test_log_sum_exp_stability(self) -> None:
        """Very negative log-weights should not produce nan/inf."""
        cloud = ParticleCloud(n_particles=3, k_states=1)
        log_w = np.array([-1000.0, -1001.0, -999.0])
        cloud.set_log_weights(log_w)
        assert np.isfinite(cloud.ess)
        assert np.all(np.isfinite(cloud.normalized_weights))

    def test_weighted_mean_uniform(self) -> None:
        """Uniform weights -> weighted_mean = arithmetic mean."""
        cloud = ParticleCloud(n_particles=100, k_states=2)
        rng = np.random.default_rng(42)
        cloud.particles = rng.standard_normal((100, 2))
        cloud.set_uniform_weights()
        np.testing.assert_allclose(
            cloud.weighted_mean(), cloud.particles.mean(axis=0), atol=1e-10
        )

    def test_weighted_cov(self) -> None:
        """Covariance shape and positive semi-definiteness."""
        cloud = ParticleCloud(n_particles=100, k_states=3)
        rng = np.random.default_rng(42)
        cloud.particles = rng.standard_normal((100, 3))
        cloud.set_uniform_weights()
        cov = cloud.weighted_cov()
        assert cov.shape == (3, 3)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_resample_resets_weights(self) -> None:
        """After resample, all log_weights = 0."""
        cloud = ParticleCloud(n_particles=10, k_states=1)
        log_w = np.random.default_rng(42).standard_normal(10)
        cloud.set_log_weights(log_w)
        indices = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        cloud.resample(indices)
        np.testing.assert_array_equal(cloud.log_weights, np.zeros(10))

    def test_resample_preserves_particles(self) -> None:
        """Particles after resample are subset of originals."""
        cloud = ParticleCloud(n_particles=10, k_states=2)
        rng = np.random.default_rng(42)
        original = rng.standard_normal((10, 2))
        cloud.particles = original.copy()
        indices = np.array([0, 0, 3, 3, 5, 5, 7, 7, 9, 9])
        cloud.resample(indices)
        for i in range(10):
            assert np.any(np.all(cloud.particles[i] == original, axis=1))

    def test_add_log_weights(self) -> None:
        """Increments are added correctly."""
        cloud = ParticleCloud(n_particles=5, k_states=1)
        cloud.set_uniform_weights()  # log_weights = 0
        increments = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cloud.add_log_weights(increments)
        np.testing.assert_array_equal(cloud.log_weights, increments)

    def test_clone_independence(self) -> None:
        """Modifying clone does not affect original."""
        cloud = ParticleCloud(n_particles=10, k_states=2)
        rng = np.random.default_rng(42)
        cloud.particles = rng.standard_normal((10, 2))
        clone = cloud.clone()
        clone.particles[0, 0] = 999.0
        assert cloud.particles[0, 0] != 999.0

    def test_repr(self) -> None:
        cloud = ParticleCloud(n_particles=100, k_states=2)
        cloud.set_uniform_weights()
        r = repr(cloud)
        assert "N=100" in r
        assert "k=2" in r
        assert "ESS=" in r

    def test_set_log_weights_wrong_shape(self) -> None:
        cloud = ParticleCloud(n_particles=10, k_states=1)
        with pytest.raises(ValueError):
            cloud.set_log_weights(np.zeros(5))

    def test_add_log_weights_wrong_shape(self) -> None:
        cloud = ParticleCloud(n_particles=10, k_states=1)
        with pytest.raises(ValueError):
            cloud.add_log_weights(np.zeros(5))

    def test_resample_wrong_size(self) -> None:
        cloud = ParticleCloud(n_particles=10, k_states=1)
        with pytest.raises(ValueError):
            cloud.resample(np.array([0, 1, 2]))
