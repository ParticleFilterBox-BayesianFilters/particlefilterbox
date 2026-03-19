"""CRITICAL convergence tests: Particle Filter vs Kalman Filter.

These tests validate that the particle filter converges to the exact
Kalman filter solution for linear Gaussian models. This is the primary
correctness check for the particle filter implementation.

Model:
    x_t = 0.9 * x_{t-1} + N(0, 1)
    y_t = x_t + N(0, 0.5)

References:
    - Gordon et al. (1993): Bootstrap filter
    - Chopin & Papaspiliopoulos (2020): Chapter 10, convergence results
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.filters.conftest import (
    LinearGaussianModel,
    kalman_filter,
)


class TestFilteredMeanCorrelation:
    """Test that PF filtered means correlate highly with Kalman means."""

    @pytest.mark.slow
    def test_filtered_mean_correlation_bootstrap(self) -> None:
        """CRITICAL: corr(pf_mean, kf_mean) > 0.99 with N=5000."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF

        # Generate data
        rng = np.random.default_rng(42)
        model = LinearGaussianModel()
        states, obs = model.simulate(n_steps=200, rng=rng)

        # Run Kalman filter (exact)
        kf_means, kf_vars, kf_ll = kalman_filter(
            obs,
            phi=model.phi,
            sigma_eta=model.sigma_eta,
            sigma_eps=model.sigma_eps,
        )

        # Run Bootstrap PF with N=5000
        config = PFConfig(n_particles=5000, seed=123, ess_threshold=0.5)
        pf = BootstrapPF(model, config)  # type: ignore[arg-type]
        results = pf.filter(obs)

        # Compute correlation
        pf_means = results.filtered_means[:, 0]
        correlation = np.corrcoef(pf_means, kf_means)[0, 1]

        assert correlation > 0.99, (
            f"CRITICAL: Correlation between PF and Kalman means is {correlation:.4f}, "
            f"expected > 0.99"
        )

    @pytest.mark.slow
    def test_filtered_mean_correlation_sir(self) -> None:
        """CRITICAL: SIR (bootstrap fallback) should also correlate > 0.99."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.sir import SIR

        rng = np.random.default_rng(42)
        model = LinearGaussianModel()
        states, obs = model.simulate(n_steps=200, rng=rng)

        kf_means, kf_vars, kf_ll = kalman_filter(
            obs,
            phi=model.phi,
            sigma_eta=model.sigma_eta,
            sigma_eps=model.sigma_eps,
        )

        config = PFConfig(n_particles=5000, seed=123, ess_threshold=0.5)
        pf = SIR(model, config)  # type: ignore[arg-type]
        results = pf.filter(obs)

        pf_means = results.filtered_means[:, 0]
        correlation = np.corrcoef(pf_means, kf_means)[0, 1]

        assert correlation > 0.99, (
            f"CRITICAL: SIR correlation with Kalman is {correlation:.4f}, "
            f"expected > 0.99"
        )


class TestLoglikelihoodConvergence:
    """Test that PF log-likelihood converges to Kalman log-likelihood."""

    @pytest.mark.slow
    def test_loglikelihood_convergence_bootstrap(self) -> None:
        """CRITICAL: |pf_loglike - kf_loglike| < 2.0 with N=5000."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF

        rng = np.random.default_rng(42)
        model = LinearGaussianModel()
        states, obs = model.simulate(n_steps=200, rng=rng)

        kf_means, kf_vars, kf_ll = kalman_filter(
            obs,
            phi=model.phi,
            sigma_eta=model.sigma_eta,
            sigma_eps=model.sigma_eps,
        )

        config = PFConfig(n_particles=5000, seed=123, ess_threshold=0.5)
        pf = BootstrapPF(model, config)  # type: ignore[arg-type]
        results = pf.filter(obs)

        ll_diff = abs(results.log_likelihood - kf_ll)

        assert ll_diff < 2.0, (
            f"CRITICAL: |PF_ll - KF_ll| = {ll_diff:.4f}, expected < 2.0. "
            f"PF_ll={results.log_likelihood:.4f}, KF_ll={kf_ll:.4f}"
        )

    @pytest.mark.slow
    def test_loglikelihood_convergence_sir(self) -> None:
        """CRITICAL: SIR log-likelihood also converges."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.sir import SIR

        rng = np.random.default_rng(42)
        model = LinearGaussianModel()
        states, obs = model.simulate(n_steps=200, rng=rng)

        kf_means, kf_vars, kf_ll = kalman_filter(
            obs,
            phi=model.phi,
            sigma_eta=model.sigma_eta,
            sigma_eps=model.sigma_eps,
        )

        config = PFConfig(n_particles=5000, seed=123, ess_threshold=0.5)
        pf = SIR(model, config)  # type: ignore[arg-type]
        results = pf.filter(obs)

        ll_diff = abs(results.log_likelihood - kf_ll)

        assert ll_diff < 2.0, (
            f"CRITICAL: SIR |PF_ll - KF_ll| = {ll_diff:.4f}, expected < 2.0. "
            f"PF_ll={results.log_likelihood:.4f}, KF_ll={kf_ll:.4f}"
        )


class TestConvergenceWithN:
    """Test that estimation error decreases with more particles."""

    @pytest.mark.slow
    def test_convergence_with_n(self) -> None:
        """Error should decrease monotonically with N."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF

        rng = np.random.default_rng(42)
        model = LinearGaussianModel()
        states, obs = model.simulate(n_steps=200, rng=rng)

        kf_means, kf_vars, kf_ll = kalman_filter(
            obs,
            phi=model.phi,
            sigma_eta=model.sigma_eta,
            sigma_eps=model.sigma_eps,
        )

        n_particles_list = [100, 500, 1000, 5000]
        errors: list[float] = []

        for n_particles in n_particles_list:
            config = PFConfig(
                n_particles=n_particles, seed=42, ess_threshold=0.5
            )
            pf = BootstrapPF(model, config)  # type: ignore[arg-type]
            results = pf.filter(obs)

            pf_means = results.filtered_means[:, 0]
            rmse = float(np.sqrt(np.mean((pf_means - kf_means) ** 2)))
            errors.append(rmse)

        # Error should generally decrease with N
        # Allow some non-monotonicity but overall trend should be clear
        assert errors[-1] < errors[0], (
            f"Error with N={n_particles_list[-1]} ({errors[-1]:.4f}) should be "
            f"less than error with N={n_particles_list[0]} ({errors[0]:.4f})"
        )

        # The largest N should give quite small error
        assert errors[-1] < 0.2, (
            f"RMSE with N=5000 is {errors[-1]:.4f}, expected < 0.2"
        )

    @pytest.mark.slow
    def test_loglikelihood_convergence_with_n(self) -> None:
        """Log-likelihood error should decrease with N."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF

        rng = np.random.default_rng(42)
        model = LinearGaussianModel()
        states, obs = model.simulate(n_steps=200, rng=rng)

        kf_means, kf_vars, kf_ll = kalman_filter(
            obs,
            phi=model.phi,
            sigma_eta=model.sigma_eta,
            sigma_eps=model.sigma_eps,
        )

        n_particles_list = [100, 500, 1000, 5000]
        ll_errors: list[float] = []

        for n_particles in n_particles_list:
            config = PFConfig(
                n_particles=n_particles, seed=42, ess_threshold=0.5
            )
            pf = BootstrapPF(model, config)  # type: ignore[arg-type]
            results = pf.filter(obs)
            ll_errors.append(abs(results.log_likelihood - kf_ll))

        # Error should decrease overall
        assert ll_errors[-1] < ll_errors[0], (
            f"LL error with N={n_particles_list[-1]} ({ll_errors[-1]:.4f}) "
            f"should be less than with N={n_particles_list[0]} ({ll_errors[0]:.4f})"
        )


class TestCoverage:
    """Test that credible intervals have correct coverage."""

    @pytest.mark.slow
    def test_coverage_95(self) -> None:
        """95% CI should contain true state approximately 95% of the time."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF

        rng = np.random.default_rng(42)
        model = LinearGaussianModel()
        states, obs = model.simulate(n_steps=500, rng=rng)

        config = PFConfig(n_particles=5000, seed=123, ess_threshold=0.5)
        pf = BootstrapPF(model, config)  # type: ignore[arg-type]
        results = pf.filter(obs)

        pf_means = results.filtered_means[:, 0]
        pf_vars = results.filtered_covs[:, 0, 0]
        pf_stds = np.sqrt(np.maximum(pf_vars, 1e-10))

        # 95% CI: mean +/- 1.96 * std
        lower = pf_means - 1.96 * pf_stds
        upper = pf_means + 1.96 * pf_stds

        # Check how often true state falls within CI
        in_ci = (states >= lower) & (states <= upper)
        coverage = float(np.mean(in_ci))

        # Allow some tolerance: coverage should be roughly 90-100%
        # (particle filter CIs may be slightly conservative or liberal)
        assert coverage > 0.85, (
            f"Coverage is {coverage:.2%}, expected > 85% for a 95% CI"
        )
        assert coverage < 1.0, (
            f"Coverage is {coverage:.2%}, which is suspicious (too high)"
        )
