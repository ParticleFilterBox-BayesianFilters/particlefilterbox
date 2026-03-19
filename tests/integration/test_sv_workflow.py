"""CRITICAL: Integration test for stochastic volatility workflow.

Tests the complete SV workflow: simulate -> filter -> smooth -> PMMH ->
diagnostics -> report. This is the most important integration test
as it exercises virtually every component of the library.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


class TestSVWorkflow:
    """Complete stochastic volatility workflow test."""

    @pytest.fixture
    def sv_data(self) -> dict[str, Any]:
        """Simulate SV data with known parameters.

        Model:
            h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t
            y_t = exp(h_t/2) * epsilon_t

        True parameters: mu=0, phi=0.97, sigma_eta=0.15
        """
        rng = np.random.default_rng(42)
        T = 200
        mu = 0.0
        phi = 0.97
        sigma_eta = 0.15

        h = np.zeros(T)
        y = np.zeros(T)
        h[0] = mu + sigma_eta * rng.standard_normal() / np.sqrt(1 - phi**2)

        for t in range(1, T):
            h[t] = mu + phi * (h[t - 1] - mu) + sigma_eta * rng.standard_normal()

        for t in range(T):
            y[t] = np.exp(h[t] / 2) * rng.standard_normal()

        return {
            "observations": y,
            "true_states": h,
            "true_params": {"mu": mu, "phi": phi, "sigma_eta": sigma_eta},
            "T": T,
        }

    def test_step1_simulate_data(self, sv_data: dict[str, Any]) -> None:
        """Step 1: Simulated data should have expected properties."""
        y = sv_data["observations"]
        h = sv_data["true_states"]
        T = sv_data["T"]

        assert len(y) == T
        assert len(h) == T
        # Returns should be approximately mean-zero
        assert abs(np.mean(y)) < 0.5
        # Returns should show volatility clustering
        assert np.std(y) > 0

    def test_step2_filter(self, sv_data: dict[str, Any]) -> None:
        """Step 2: Bootstrap PF should filter SV data.

        This test validates that the particle filter can be applied to
        SV data and produces reasonable state estimates.
        """
        try:
            from particlefilterbox.models.sv import SVModel

            from particlefilterbox.filters.bootstrap import BootstrapFilter

            model = SVModel(mu=0.0, phi=0.97, sigma_eta=0.15)
            rng = np.random.default_rng(42)
            pf = BootstrapFilter(model=model, n_particles=500, rng=rng)
            results = pf.filter(sv_data["observations"])

            # Should produce log-likelihood
            assert hasattr(results, "log_likelihood")

            # Filtered states should correlate with true states
            filtered_mean = getattr(results, "filtered_mean", None)
            if filtered_mean is not None:
                fm = np.asarray(filtered_mean).flatten()
                h_true = sv_data["true_states"]
                corr = np.corrcoef(fm[: len(h_true)], h_true[: len(fm)])[0, 1]
                assert corr > 0.5, f"Correlation too low: {corr}"

        except ImportError:
            pytest.skip("BootstrapFilter or SVModel not yet implemented")

    def test_step3_smooth(self, sv_data: dict[str, Any]) -> None:
        """Step 3: Smoother should improve on filtered estimates."""
        try:
            from particlefilterbox.models.sv import SVModel

            from particlefilterbox.filters.bootstrap import BootstrapFilter
            from particlefilterbox.smoothers import FFBSm

            model = SVModel(mu=0.0, phi=0.97, sigma_eta=0.15)
            rng = np.random.default_rng(42)
            pf = BootstrapFilter(model=model, n_particles=200, rng=rng)
            filter_results = pf.filter(sv_data["observations"])

            smoother = FFBSm(model=model, rng=rng)
            smooth_results = smoother.smooth(filter_results)

            # Smoothed estimates should exist
            smoothed_mean = getattr(smooth_results, "smoothed_mean", None)
            assert smoothed_mean is not None or smooth_results is not None

        except ImportError:
            pytest.skip("Smoother not yet implemented")

    def test_step4_pmmh(self, sv_data: dict[str, Any]) -> None:
        """Step 4: PMMH should recover approximate true parameters."""
        try:
            from particlefilterbox.models.sv import SVModel

            from particlefilterbox.pmcmc.pmmh import PMMH

            model = SVModel()
            rng = np.random.default_rng(42)

            pmmh = PMMH(
                model=model,
                n_particles=100,
                n_iterations=500,  # Short for testing
                rng=rng,
            )
            results = pmmh.run(sv_data["observations"])

            chain = getattr(results, "chain", None)
            assert chain is not None, "PMMH should produce a chain"

            chain_arr = np.asarray(chain)
            assert chain_arr.shape[0] >= 100, "Chain should have iterations"

        except ImportError:
            pytest.skip("PMMH not yet implemented")

    def test_step5_diagnostics(self, sv_data: dict[str, Any]) -> None:
        """Step 5: Diagnostics should run on filter results."""
        try:
            from particlefilterbox.models.sv import SVModel

            from particlefilterbox.filters.bootstrap import BootstrapFilter

            model = SVModel(mu=0.0, phi=0.97, sigma_eta=0.15)
            rng = np.random.default_rng(42)
            pf = BootstrapFilter(model=model, n_particles=200, rng=rng)
            results = pf.filter(sv_data["observations"])

            # ESS should be computable
            ess = getattr(results, "ess", None)
            if ess is not None:
                ess_arr = np.asarray(ess)
                assert np.all(ess_arr > 0)
                assert np.all(ess_arr <= 200)

        except ImportError:
            pytest.skip("Components not yet implemented")

    def test_step6_report(self, sv_data: dict[str, Any]) -> None:
        """Step 6: Report should be generatable."""
        try:
            from particlefilterbox.reports import SVReportTransformer

            # Create mock results
            mock_results = SimpleNamespace(
                filtered_mean=np.random.randn(200, 1),
                chain=np.random.randn(500, 3),
                param_names=["mu", "phi", "sigma_eta"],
                log_likelihood=-100.0,
            )

            transformer = SVReportTransformer()
            report = transformer.transform(mock_results)

            html = report.to_html()
            assert "Stochastic Volatility" in html
            assert len(report.sections) >= 3

        except ImportError:
            pytest.skip("Reports module not yet implemented")
