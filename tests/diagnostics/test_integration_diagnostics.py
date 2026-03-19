"""Integration tests for diagnostics and acceleration modules.

Tests end-to-end workflows combining multiple diagnostic tools.
"""

import numpy as np
import pytest

from particlefilterbox.diagnostics import (
    ESSMonitor,
    WeightAnalysis,
    DegeneracyDetector,
    PMCMCDiagnostics,
)
from particlefilterbox.acceleration.numba_kernels import (
    _NUMBA_AVAILABLE,
    log_sum_exp_python,
    log_sum_exp_numba,
    normalize_log_weights_python,
    normalize_log_weights_numba,
    systematic_resample_python,
    systematic_resample_numba,
    enable_numba,
    disable_numba,
)
from particlefilterbox.acceleration.adaptive_n import AdaptiveN


class TestFullDiagnosticsWorkflow:
    """Test complete diagnostics workflow."""

    def test_full_diagnostics_workflow(self) -> None:
        """Simulate a full filtering run and analyze with all diagnostics."""
        rng = np.random.default_rng(42)
        n_particles = 200
        n_time_steps = 50

        # Initialize monitors
        ess_monitor = ESSMonitor()
        weight_analysis = WeightAnalysis()

        # Simulate filter-like weight evolution
        for t in range(n_time_steps):
            # Generate weights that gradually become less uniform
            alpha = np.ones(n_particles)
            alpha[0] = 1.0 + t * 0.1  # slowly increasing dominance
            weights = rng.dirichlet(alpha)

            ess_monitor.update_from_weights(weights, time_step=t)
            weight_analysis.update_from_weights(weights)

        # Check ESS monitor
        ess_summary = ess_monitor.summary()
        assert ess_summary["n_time_steps"] == n_time_steps
        assert ess_summary["ess_min"] > 0
        assert ess_summary["ess_mean"] > 0

        # Check weight analysis
        wa_summary = weight_analysis.summary()
        assert wa_summary["n_time_steps"] == n_time_steps
        assert wa_summary["entropy_current"] > 0
        assert 0 <= wa_summary["gini_current"] <= 1

        # Entropy should be below max (weights not perfectly uniform)
        assert wa_summary["entropy_ratio"] <= 1.0

        # Histories should have correct length
        assert len(weight_analysis.entropy_history()) == n_time_steps
        assert len(weight_analysis.gini_history()) == n_time_steps

    def test_diagnostics_with_degenerate_filter(self) -> None:
        """Test diagnostics detect degenerate filter correctly."""
        n_particles = 100
        n_time_steps = 20

        ess_monitor = ESSMonitor()
        weight_analysis = WeightAnalysis()

        for t in range(n_time_steps):
            # Increasingly degenerate weights
            weights = np.zeros(n_particles)
            weights[0] = 1.0 - 0.001 * (n_time_steps - t)
            weights[1:] = (0.001 * (n_time_steps - t)) / (n_particles - 1)
            weights = weights / weights.sum()

            ess_monitor.update_from_weights(weights, time_step=t)
            weight_analysis.update_from_weights(weights)

        # Should have alerts
        assert not ess_monitor.is_healthy() or len(ess_monitor.alerts) > 0

        # Gini should be high
        wa_summary = weight_analysis.summary()
        assert wa_summary["gini_current"] > 0.5

    def test_pmcmc_diagnostics_integration(self) -> None:
        """Test PMCMC diagnostics with synthetic chains."""
        rng = np.random.default_rng(42)

        # 4 converged chains
        chains = rng.normal(5.0, 1.0, size=(4, 3000, 2))
        diag = PMCMCDiagnostics(chains)

        summary = diag.summary()
        assert summary["n_chains"] == 4
        assert summary["chain_length"] == 3000
        assert summary["n_params"] == 2
        assert summary["is_converged"] is True

        # R-hat should be close to 1
        assert summary["r_hat_max"] < 1.1

        # ESS should be reasonable
        assert summary["ess_min"] > 100

    def test_adaptive_n_integration(self) -> None:
        """Test AdaptiveN with simulated ESS trajectory."""
        adaptive = AdaptiveN(n_min=100, n_max=5000, growth_factor=1.5, shrink_factor=0.7)

        current_n = 500
        rng = np.random.default_rng(42)

        n_steps = 30
        for _ in range(n_steps):
            # Simulate ESS
            ess_ratio = rng.uniform(0.05, 0.95)
            ess = ess_ratio * current_n
            current_n = adaptive.adapt(current_n=current_n, ess=ess)

        assert len(adaptive.n_history) == n_steps
        # All values should be within bounds
        assert all(adaptive.n_min <= n <= adaptive.n_max for n in adaptive.n_history)


class TestNumbaEndToEnd:
    """Test Numba acceleration end-to-end."""

    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not installed")
    def test_numba_end_to_end_loglik(self) -> None:
        """Numba-accelerated operations should give same results as Python."""
        rng = np.random.default_rng(42)

        # Generate test data
        n = 5000
        log_weights = rng.normal(-5, 2, size=n)
        weights = normalize_log_weights_python(log_weights)
        u = rng.uniform(0, 1.0 / n)

        # Compare log-sum-exp
        lse_py = log_sum_exp_python(log_weights)
        lse_nb = log_sum_exp_numba(log_weights)
        assert abs(lse_py - lse_nb) < 1e-10, (
            f"log_sum_exp: Python={lse_py}, Numba={lse_nb}"
        )

        # Compare normalization
        w_py = normalize_log_weights_python(log_weights)
        w_nb = normalize_log_weights_numba(log_weights)
        np.testing.assert_allclose(w_py, w_nb, atol=1e-10)

        # Compare resampling
        idx_py = systematic_resample_python(weights, u)
        idx_nb = systematic_resample_numba(weights, u)
        np.testing.assert_array_equal(idx_py, idx_nb)

    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not installed")
    def test_enable_disable_numba(self) -> None:
        """Enable/disable cycle should not break anything."""
        assert enable_numba() is True
        disable_numba()

        # Operations should still work after disable
        rng = np.random.default_rng(42)
        log_w = rng.normal(-3, 1, size=100)
        w = normalize_log_weights_python(log_w)
        assert abs(np.sum(w) - 1.0) < 1e-10

    def test_imports_work(self) -> None:
        """All public imports should work."""
        from particlefilterbox.diagnostics import (  # noqa: F401
            ESSMonitor,
            WeightAnalysis,
            ConvergenceStudy,
            DegeneracyDetector,
            ModelComparison,
            PMCMCDiagnostics,
        )
        from particlefilterbox.acceleration import (  # noqa: F401
            enable_numba,
            disable_numba,
            GPUBackend,
            ParallelRunner,
            AdaptiveN,
        )

        # Verify they are callable/instantiable
        assert callable(ESSMonitor)
        assert callable(WeightAnalysis)
        assert callable(ConvergenceStudy)
        assert callable(DegeneracyDetector)
        assert callable(ModelComparison)
        assert callable(PMCMCDiagnostics)
        assert callable(enable_numba)
        assert callable(disable_numba)
        assert callable(GPUBackend)
        assert callable(ParallelRunner)
        assert callable(AdaptiveN)
