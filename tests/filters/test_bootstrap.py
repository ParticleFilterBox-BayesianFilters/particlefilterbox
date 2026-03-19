"""Tests for Bootstrap Particle Filter.

Tests cover:
- Basic functionality (runs without error)
- Output shapes
- Log-likelihood computation
- ESS monitoring
- Missing data handling
- Reproducibility with seed
- Step-by-step vs batch consistency
"""

from __future__ import annotations

import numpy as np


class TestBootstrapPFBasic:
    """Basic functionality tests for BootstrapPF."""

    def test_runs_without_error(self, linear_gaussian_data, pf_config):
        """BootstrapPF should run on linear Gaussian data without errors."""
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model, _, obs = linear_gaussian_data
        pf = BootstrapPF(model, pf_config)
        results = pf.filter(obs)
        assert results is not None

    def test_returns_correct_shapes(self, linear_gaussian_data, pf_config):
        """Output arrays should have correct shapes."""
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model, _, obs = linear_gaussian_data
        n_obs = len(obs)
        pf = BootstrapPF(model, pf_config)
        results = pf.filter(obs)

        assert results.filtered_means.shape == (n_obs, model.state_dim)
        assert results.filtered_covs.shape == (n_obs, model.state_dim, model.state_dim)
        assert results.log_likelihoods.shape == (n_obs,)
        assert results.ess_history.shape == (n_obs,)
        assert results.resampled.shape == (n_obs,)
        assert results.n_particles == pf_config.n_particles

    def test_loglikelihood_finite(self, linear_gaussian_data, pf_config):
        """Log-likelihood should be finite."""
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model, _, obs = linear_gaussian_data
        pf = BootstrapPF(model, pf_config)
        results = pf.filter(obs)

        assert np.isfinite(results.log_likelihood)
        assert np.all(np.isfinite(results.log_likelihoods))

    def test_ess_history(self, linear_gaussian_data, pf_config):
        """ESS should be tracked and within valid range."""
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model, _, obs = linear_gaussian_data
        pf = BootstrapPF(model, pf_config)
        results = pf.filter(obs)

        n_part = pf_config.n_particles
        assert np.all(results.ess_history > 0)
        assert np.all(results.ess_history <= n_part + 1)  # +1 for numerical tolerance


class TestBootstrapPFMissingData:
    """Tests for missing data handling."""

    def test_missing_data_nan(self, linear_gaussian_data, pf_config):
        """Bootstrap PF should handle NaN observations gracefully."""
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model, _, obs = linear_gaussian_data
        obs_with_missing = obs.copy()
        # Set some observations to NaN
        missing_idx = [10, 20, 30, 50, 75]
        obs_with_missing[missing_idx] = np.nan

        pf = BootstrapPF(model, pf_config)
        results = pf.filter(obs_with_missing)

        # Should still produce finite results
        assert np.isfinite(results.log_likelihood)
        # Missing steps should contribute 0 to log-likelihood
        assert np.all(results.log_likelihoods[missing_idx] == 0.0)
        # Filtered means should still be finite
        assert np.all(np.isfinite(results.filtered_means))

    def test_missing_data_mask(self, linear_gaussian_data, pf_config):
        """Bootstrap PF should handle explicit mask."""
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model, _, obs = linear_gaussian_data
        mask = np.zeros(len(obs), dtype=bool)
        mask[[10, 20, 30]] = True

        pf = BootstrapPF(model, pf_config)
        results = pf.filter(obs, mask=mask)

        assert np.isfinite(results.log_likelihood)
        assert np.all(results.log_likelihoods[mask] == 0.0)


class TestBootstrapPFReproducibility:
    """Tests for reproducibility with fixed seed."""

    def test_reproducibility(self, linear_gaussian_data):
        """Same seed should produce identical results."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model, _, obs = linear_gaussian_data

        config1 = PFConfig(n_particles=500, seed=12345, ess_threshold=0.5)
        config2 = PFConfig(n_particles=500, seed=12345, ess_threshold=0.5)

        pf1 = BootstrapPF(model, config1)
        pf2 = BootstrapPF(model, config2)

        results1 = pf1.filter(obs)
        results2 = pf2.filter(obs)

        np.testing.assert_array_equal(results1.filtered_means, results2.filtered_means)
        np.testing.assert_equal(results1.log_likelihood, results2.log_likelihood)

    def test_different_seeds_differ(self, linear_gaussian_data):
        """Different seeds should produce different results."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model, _, obs = linear_gaussian_data

        config1 = PFConfig(n_particles=500, seed=111, ess_threshold=0.5)
        config2 = PFConfig(n_particles=500, seed=222, ess_threshold=0.5)

        pf1 = BootstrapPF(model, config1)
        pf2 = BootstrapPF(model, config2)

        results1 = pf1.filter(obs)
        results2 = pf2.filter(obs)

        # Results should differ (very unlikely to be exactly equal)
        assert not np.allclose(results1.filtered_means, results2.filtered_means)


class TestBootstrapPFStepByStep:
    """Tests for step-by-step (online) mode consistency."""

    def test_step_by_step_equals_batch(self, linear_gaussian_data):
        """Step-by-step filtering should match batch filtering."""
        from scipy.special import logsumexp

        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model, _, obs = linear_gaussian_data

        # Batch mode
        config_batch = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)
        pf_batch = BootstrapPF(model, config_batch)
        results_batch = pf_batch.filter(obs)

        # Step-by-step mode (same seed)
        config_step = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)
        pf_step = BootstrapPF(model, config_step)

        rng = pf_step._get_rng()
        cloud = pf_step.initialize(rng)

        step_ll = 0.0
        step_means = []

        for t in range(len(obs)):
            y_t = np.atleast_1d(obs[t])
            cloud, ll_t = pf_step.filter_step(cloud, y_t, t)
            step_ll += ll_t
            # Compute mean from cloud
            weights = np.exp(cloud.log_weights - logsumexp(cloud.log_weights))
            mean = np.average(cloud.particles, weights=weights, axis=0)
            step_means.append(mean)

        step_means_arr = np.array(step_means)

        # Should match batch results
        np.testing.assert_allclose(
            step_means_arr, results_batch.filtered_means, rtol=1e-10
        )
        np.testing.assert_allclose(step_ll, results_batch.log_likelihood, rtol=1e-10)


class TestBootstrapPFSV:
    """Test Bootstrap PF on Stochastic Volatility model."""

    def test_runs_on_sv_model(self, sv_data, pf_config):
        """BootstrapPF should work with SV model."""
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model, _, obs = sv_data
        pf = BootstrapPF(model, pf_config)
        results = pf.filter(obs)

        assert np.isfinite(results.log_likelihood)
        assert np.all(np.isfinite(results.filtered_means))
