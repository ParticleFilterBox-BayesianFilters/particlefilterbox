"""Tests for PMCMC diagnostics."""

import numpy as np
import pytest

from particlefilterbox.diagnostics.pmcmc_diagnostics import PMCMCDiagnostics


class TestPMCMCDiagnostics:
    """Tests for PMCMCDiagnostics."""

    def test_r_hat_converged_chains(self) -> None:
        """Independent chains from same distribution should have R-hat ~ 1."""
        rng = np.random.default_rng(42)
        # 4 chains, 2000 iterations, 2 parameters - all from N(0,1)
        chains = rng.normal(0, 1, size=(4, 2000, 2))
        diag = PMCMCDiagnostics(chains)

        r_hat_vals = diag.r_hat()
        assert isinstance(r_hat_vals, np.ndarray)
        for i, rh in enumerate(r_hat_vals):
            assert rh < 1.1, f"R-hat[{i}] = {rh:.3f} > 1.1"

    def test_r_hat_divergent_chains(self) -> None:
        """Chains from different distributions should have R-hat >> 1."""
        rng = np.random.default_rng(42)
        chains = np.zeros((4, 1000, 1))
        for m in range(4):
            chains[m, :, 0] = rng.normal(m * 10, 1, size=1000)  # different means
        diag = PMCMCDiagnostics(chains)
        rh = diag.r_hat(param=0)
        assert rh > 1.5, f"R-hat = {rh:.3f} should be >> 1 for divergent chains"

    def test_r_hat_requires_multiple_chains(self) -> None:
        """R-hat with single chain should raise."""
        chains = np.random.randn(1000, 2)  # single chain
        diag = PMCMCDiagnostics(chains)
        with pytest.raises(ValueError, match="at least 2 chains"):
            diag.r_hat()

    def test_ess_chain(self) -> None:
        """ESS of iid samples should be close to chain length."""
        rng = np.random.default_rng(42)
        # IID samples -> ESS should be close to L
        chains = rng.normal(0, 1, size=(1, 5000, 1))
        diag = PMCMCDiagnostics(chains)
        ess_val = diag.ess(param=0)
        assert isinstance(ess_val, float)
        # For IID, ESS ~ L
        assert ess_val > 2000, f"ESS = {ess_val:.0f} too low for IID chain"

    def test_ess_autocorrelated(self) -> None:
        """ESS of autocorrelated chain should be much less than L."""
        rng = np.random.default_rng(42)
        # AR(1) with high autocorrelation
        L = 5000
        x = np.zeros(L)
        for t in range(1, L):
            x[t] = 0.99 * x[t - 1] + rng.normal(0, 0.1)
        chains = x.reshape(1, L, 1)
        diag = PMCMCDiagnostics(chains)
        ess_val = diag.ess(param=0)
        assert isinstance(ess_val, float)
        assert ess_val < L / 5, f"ESS = {ess_val:.0f} too high for autocorrelated chain"

    def test_acf_decreasing(self) -> None:
        """ACF of IID chain should be close to 0 for lag > 0."""
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, size=(1, 5000, 1))
        diag = PMCMCDiagnostics(chains)
        acf_vals = diag.acf(param=0, chain=0, max_lag=20)
        # lag 0 should be 1
        assert abs(acf_vals[0] - 1.0) < 1e-10
        # lags > 0 should be close to 0 for IID
        for k in range(1, len(acf_vals)):
            assert abs(acf_vals[k]) < 0.1, f"ACF[{k}] = {acf_vals[k]:.3f} should be ~0"

    def test_geweke(self) -> None:
        """Geweke test on IID chain should indicate convergence."""
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, size=(1, 5000, 1))
        diag = PMCMCDiagnostics(chains)
        g = diag.geweke(param=0, chain=0)
        assert abs(g["z_score"]) < 3.0, f"Geweke z = {g['z_score']:.3f}"
        assert g["converged"] is True

    def test_geweke_non_stationary(self) -> None:
        """Geweke test on trending chain should detect non-convergence."""
        x = np.linspace(0, 100, 5000)  # trending chain
        chains = x.reshape(1, 5000, 1)
        diag = PMCMCDiagnostics(chains)
        g = diag.geweke(param=0, chain=0)
        assert abs(g["z_score"]) > 2.0, "Should detect non-stationarity"

    def test_acceptance_rate(self) -> None:
        """Test acceptance rate computation."""
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, size=(1, 1000, 1))
        diag = PMCMCDiagnostics(chains)
        ar = diag.acceptance_rate(chain=0)
        # For IID, almost all values differ -> rate ~ 1
        assert ar > 0.9

    def test_is_converged(self) -> None:
        """IID chains should be flagged as converged."""
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, size=(4, 5000, 2))
        diag = PMCMCDiagnostics(chains)
        assert diag.is_converged()

    def test_summary(self) -> None:
        """Test summary output."""
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, size=(4, 2000, 2))
        diag = PMCMCDiagnostics(chains)
        s = diag.summary()
        assert "ess" in s
        assert "r_hat" in s
        assert "acceptance_rates" in s
        assert "is_converged" in s
        assert "geweke_all_converged" in s

    def test_trace(self) -> None:
        """Test trace retrieval."""
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, size=(2, 100, 3))
        diag = PMCMCDiagnostics(chains)

        # Single chain trace
        t = diag.trace(param=1, chain=0)
        assert t.shape == (100,)

        # All chains trace
        t_all = diag.trace(param=1)
        assert t_all.shape == (2, 100)

    def test_1d_input(self) -> None:
        """Test with 1D input (single parameter, single chain)."""
        x = np.random.randn(500)
        diag = PMCMCDiagnostics(x)
        assert diag.n_chains == 1
        assert diag.n_params == 1
        assert diag.chain_length == 500

    def test_ess_all_params(self) -> None:
        """Test ESS for all parameters."""
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, size=(1, 2000, 3))
        diag = PMCMCDiagnostics(chains)
        ess_vals = diag.ess()
        assert isinstance(ess_vals, np.ndarray)
        assert len(ess_vals) == 3
