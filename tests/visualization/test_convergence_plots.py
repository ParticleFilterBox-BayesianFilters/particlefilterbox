"""Tests for convergence diagnostic plots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from particlefilterbox.visualization.convergence_plots import (
    plot_convergence_rate,
    plot_loglike_distribution,
    plot_qq_weights,
)


@pytest.fixture
def mock_conv_results() -> dict[int, list[float]]:
    """Create mock convergence results."""
    rng = np.random.default_rng(42)
    results: dict[int, list[float]] = {}
    true_val = 1.0
    for n in [100, 500, 1000, 5000]:
        noise = rng.standard_normal(20) / np.sqrt(n)
        results[n] = list(true_val + noise)
    return results


@pytest.fixture
def mock_filter_results() -> SimpleNamespace:
    """Create mock filter results with weights."""
    rng = np.random.default_rng(42)
    T, N = 50, 100
    weights = rng.dirichlet(np.ones(N), size=T)
    return SimpleNamespace(weights=weights)


class TestConvergencePlotsNoError:
    """Test that convergence plot functions run without errors."""

    def test_plot_convergence_rate(
        self, mock_conv_results: dict[int, list[float]]
    ) -> None:
        """plot_convergence_rate should produce a figure without error."""
        fig, ax = plot_convergence_rate(mock_conv_results, true_value=1.0)
        assert fig is not None
        plt.close(fig)

    def test_plot_convergence_rate_no_true(
        self, mock_conv_results: dict[int, list[float]]
    ) -> None:
        """plot_convergence_rate without true value."""
        fig, ax = plot_convergence_rate(mock_conv_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_loglike_distribution(
        self, mock_conv_results: dict[int, list[float]]
    ) -> None:
        """plot_loglike_distribution should produce a figure without error."""
        fig, ax = plot_loglike_distribution(mock_conv_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_qq_weights(self, mock_filter_results: Any) -> None:
        """plot_qq_weights should produce a figure without error."""
        fig, ax = plot_qq_weights(mock_filter_results, t=10)
        assert fig is not None
        plt.close(fig)

    def test_with_existing_ax(self, mock_filter_results: Any) -> None:
        """Convergence plots should accept an existing ax parameter."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_qq_weights(mock_filter_results, ax=ax)
        assert ax2 is ax
        plt.close(fig)
