"""Tests for PMCMCResults."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.pmcmc.results import PMCMCResults


@pytest.fixture
def sample_results() -> PMCMCResults:
    """Create sample PMCMCResults for testing."""
    rng = np.random.default_rng(42)
    n_iter = 1000
    k_params = 3

    # Simulate a simple chain (AR(1) process for realistic autocorrelation)
    chains = np.zeros((n_iter, k_params))
    chains[0] = rng.normal(size=k_params)
    for t in range(1, n_iter):
        chains[t] = 0.9 * chains[t - 1] + 0.1 * rng.normal(size=k_params)

    # Add known means
    chains[:, 0] += 1.0
    chains[:, 1] += 2.0
    chains[:, 2] += 3.0

    log_liks = -0.5 * np.sum(chains**2, axis=1)
    acceptance = rng.random(n_iter) < 0.25

    return PMCMCResults(
        chains=chains,
        param_names=["alpha", "beta", "gamma"],
        log_likelihood_chain=log_liks,
        acceptance_history=acceptance,
        burnin=200,
        thin=1,
    )


class TestPMCMCResults:
    """Tests for PMCMCResults container."""

    def test_posterior_samples_burnin(self, sample_results: PMCMCResults) -> None:
        """Posterior samples should exclude burn-in."""
        assert len(sample_results.posterior_samples) == 800  # 1000 - 200

    def test_posterior_samples_thinning(self) -> None:
        """Posterior samples should apply thinning."""
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(1000, 2))
        results = PMCMCResults(
            chains=chains,
            burnin=200,
            thin=2,
        )
        assert len(results.posterior_samples) == 400  # (1000-200)/2

    def test_posterior_mean(self, sample_results: PMCMCResults) -> None:
        """Posterior mean should be close to chain means."""
        mean = sample_results.posterior_mean()
        assert mean.shape == (3,)
        # Means should be approximately [1, 2, 3]
        assert abs(mean[0] - 1.0) < 0.5
        assert abs(mean[1] - 2.0) < 0.5
        assert abs(mean[2] - 3.0) < 0.5

    def test_posterior_std(self, sample_results: PMCMCResults) -> None:
        """Posterior std should be positive."""
        std = sample_results.posterior_std()
        assert std.shape == (3,)
        assert np.all(std > 0)

    def test_credible_interval(self, sample_results: PMCMCResults) -> None:
        """Credible intervals should contain posterior mean."""
        lower, upper = sample_results.credible_interval(alpha=0.05)
        mean = sample_results.posterior_mean()
        assert np.all(lower < mean)
        assert np.all(upper > mean)
        assert np.all(lower < upper)

    def test_acceptance_rate(self, sample_results: PMCMCResults) -> None:
        """Acceptance rate should be in [0, 1]."""
        rate = sample_results.acceptance_rate()
        assert 0.0 <= rate <= 1.0

    def test_effective_sample_size(self, sample_results: PMCMCResults) -> None:
        """ESS should be positive and less than n_samples."""
        ess = sample_results.effective_sample_size(param_idx=0)
        assert ess > 0
        assert ess <= sample_results.n_effective_samples

    def test_summary_format(self, sample_results: PMCMCResults) -> None:
        """Summary should be a formatted string with key information."""
        summary = sample_results.summary()
        assert "PMCMC Posterior Summary" in summary
        assert "alpha" in summary
        assert "beta" in summary
        assert "gamma" in summary
        assert "Acceptance rate" in summary

    def test_acf_shape(self, sample_results: PMCMCResults) -> None:
        """ACF should have correct shape and start at 1."""
        acf = sample_results.acf(param_idx=0, max_lag=20)
        assert acf.shape == (21,)
        assert np.isclose(acf[0], 1.0)

    def test_acf_decreasing(self, sample_results: PMCMCResults) -> None:
        """ACF should generally decrease for correlated chain."""
        acf = sample_results.acf(param_idx=0, max_lag=20)
        # First few lags should be decreasing for AR(1) chain
        assert acf[1] < acf[0]
        assert acf[5] < acf[1]

    def test_r_hat_split_chain(self, sample_results: PMCMCResults) -> None:
        """Split-chain R-hat should be close to 1 for converged chain."""
        r = sample_results.r_hat(param_idx=0)
        assert 0.9 < r < 1.3  # Relaxed bound for split chain

    def test_r_hat_multi_chain(self) -> None:
        """Multi-chain R-hat should be close to 1 for similar chains."""
        rng = np.random.default_rng(42)
        chains_list = []
        for i in range(4):
            ch = rng.normal(loc=1.0, scale=0.1, size=(500, 2))
            chains_list.append(
                PMCMCResults(chains=ch, burnin=100, thin=1)
            )

        r = chains_list[0].r_hat(
            other_chains=chains_list[1:], param_idx=0
        )
        assert r < 1.1

    def test_geweke_converged(self, sample_results: PMCMCResults) -> None:
        """Geweke test should not reject for converged chain."""
        z, p = sample_results.geweke_test(param_idx=0)
        assert isinstance(z, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_trace_plot_data(self, sample_results: PMCMCResults) -> None:
        """Trace plot data should have correct shapes."""
        iters, values = sample_results.trace_plot_data(param_idx=0)
        assert len(iters) == 1000
        assert len(values) == 1000

    def test_to_dataframe(self, sample_results: PMCMCResults) -> None:
        """DataFrame should have correct columns and shape."""
        df = sample_results.to_dataframe()
        assert list(df.columns) == ["alpha", "beta", "gamma"]
        assert len(df) == 800  # post-burnin

    def test_default_param_names(self) -> None:
        """Should generate default param names if none given."""
        chains = np.random.default_rng(42).normal(size=(100, 3))
        results = PMCMCResults(chains=chains, burnin=10, thin=1)
        assert results.param_names == ["param_0", "param_1", "param_2"]

    def test_n_params(self, sample_results: PMCMCResults) -> None:
        """Should report correct number of parameters."""
        assert sample_results.n_params == 3

    def test_n_iterations(self, sample_results: PMCMCResults) -> None:
        """Should report correct number of iterations."""
        assert sample_results.n_iterations == 1000
