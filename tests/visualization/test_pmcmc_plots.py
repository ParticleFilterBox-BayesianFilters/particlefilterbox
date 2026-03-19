"""Tests for PMCMC diagnostic plots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from particlefilterbox.visualization.pmcmc_plots import (
    plot_acf,
    plot_pairplot,
    plot_posterior,
    plot_posterior_predictive,
    plot_running_mean,
    plot_trace,
)


@pytest.fixture
def mock_pmcmc_results() -> SimpleNamespace:
    """Create mock PMCMCResults for testing."""
    rng = np.random.default_rng(42)
    n_iter = 1000
    k_params = 3
    chain = rng.standard_normal((n_iter, k_params))
    # Add trend to make trace interesting
    chain[:, 0] += np.linspace(0, 2, n_iter)
    return SimpleNamespace(
        chain=chain,
        param_names=["mu", "phi", "sigma"],
        observations=rng.standard_normal(100),
        posterior_predictive=rng.standard_normal((50, 100)),
    )


class TestPMCMCPlotsNoError:
    """Test that PMCMC plot functions run without errors."""

    def test_plot_trace(self, mock_pmcmc_results: Any) -> None:
        """plot_trace should produce a figure without error."""
        fig, ax = plot_trace(mock_pmcmc_results, param="mu")
        assert fig is not None
        plt.close(fig)

    def test_plot_trace_by_index(self, mock_pmcmc_results: Any) -> None:
        """plot_trace with integer index."""
        fig, ax = plot_trace(mock_pmcmc_results, param=1)
        assert fig is not None
        plt.close(fig)

    def test_plot_posterior(self, mock_pmcmc_results: Any) -> None:
        """plot_posterior should produce a figure without error."""
        fig, ax = plot_posterior(mock_pmcmc_results, param="phi")
        assert fig is not None
        plt.close(fig)

    def test_plot_posterior_with_prior(self, mock_pmcmc_results: Any) -> None:
        """plot_posterior with prior overlay."""
        from scipy.stats import norm

        prior = norm(0, 1).pdf
        fig, ax = plot_posterior(mock_pmcmc_results, param="mu", prior=prior)
        assert fig is not None
        plt.close(fig)

    def test_plot_acf(self, mock_pmcmc_results: Any) -> None:
        """plot_acf should produce a figure without error."""
        fig, ax = plot_acf(mock_pmcmc_results, param="sigma")
        assert fig is not None
        plt.close(fig)

    def test_plot_pairplot(self, mock_pmcmc_results: Any) -> None:
        """plot_pairplot should produce a figure without error."""
        fig, axes = plot_pairplot(mock_pmcmc_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_running_mean(self, mock_pmcmc_results: Any) -> None:
        """plot_running_mean should produce a figure without error."""
        fig, ax = plot_running_mean(mock_pmcmc_results, param="mu")
        assert fig is not None
        plt.close(fig)

    def test_plot_posterior_predictive(self, mock_pmcmc_results: Any) -> None:
        """plot_posterior_predictive should produce a figure without error."""
        fig, ax = plot_posterior_predictive(mock_pmcmc_results)
        assert fig is not None
        plt.close(fig)

    def test_with_existing_ax(self, mock_pmcmc_results: Any) -> None:
        """PMCMC plots should accept an existing ax parameter."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_trace(mock_pmcmc_results, param=0, ax=ax)
        assert ax2 is ax
        plt.close(fig)
