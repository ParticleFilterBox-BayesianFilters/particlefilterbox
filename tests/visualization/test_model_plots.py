"""Tests for model-specific visualization plots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from particlefilterbox.visualization.model_plots import (
    plot_irf,
    plot_jump_indicators,
    plot_regime_probabilities,
    plot_volatility,
)


@pytest.fixture
def mock_sv_results() -> SimpleNamespace:
    """Create mock SV filter results."""
    rng = np.random.default_rng(42)
    T, N, d = 100, 200, 1
    particles = rng.standard_normal((T, N, d))
    weights = np.ones((T, N)) / N
    filtered_mean = rng.standard_normal((T, d))
    return SimpleNamespace(
        particles=particles,
        weights=weights,
        filtered_mean=filtered_mean,
    )


@pytest.fixture
def mock_regime_results() -> SimpleNamespace:
    """Create mock regime-switching results."""
    rng = np.random.default_rng(42)
    T, K = 100, 3
    regime_probs = rng.dirichlet(np.ones(K), size=T)
    jump_probs = rng.uniform(0, 1, size=T)
    return SimpleNamespace(
        regime_probs=regime_probs,
        jump_probs=jump_probs,
    )


@pytest.fixture
def mock_irf_results() -> SimpleNamespace:
    """Create mock IRF results."""
    rng = np.random.default_rng(42)
    n_draws, n_periods, n_vars, n_shocks = 100, 20, 2, 2
    irf = rng.standard_normal((n_draws, n_periods, n_vars, n_shocks))
    # Make IRF decay to zero
    for t in range(n_periods):
        irf[:, t, :, :] *= np.exp(-0.1 * t)
    return SimpleNamespace(irf=irf)


class TestModelPlotsNoError:
    """Test that model plot functions run without errors."""

    def test_plot_volatility(self, mock_sv_results: Any) -> None:
        """plot_volatility should produce a figure without error."""
        fig, ax = plot_volatility(mock_sv_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_volatility_with_true(self, mock_sv_results: Any) -> None:
        """plot_volatility with true volatility overlay."""
        true_vol = np.random.default_rng(42).standard_normal(100)
        fig, ax = plot_volatility(mock_sv_results, true_vol=true_vol)
        assert fig is not None
        plt.close(fig)

    def test_plot_jump_indicators(self, mock_regime_results: Any) -> None:
        """plot_jump_indicators should produce a figure without error."""
        fig, ax = plot_jump_indicators(mock_regime_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_regime_probabilities(self, mock_regime_results: Any) -> None:
        """plot_regime_probabilities should produce a figure without error."""
        fig, ax = plot_regime_probabilities(mock_regime_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_irf(self, mock_irf_results: Any) -> None:
        """plot_irf should produce a figure without error."""
        fig, ax = plot_irf(mock_irf_results, shock_idx=0, response_idx=0)
        assert fig is not None
        plt.close(fig)

    def test_with_existing_ax(self, mock_sv_results: Any) -> None:
        """Model plots should accept an existing ax parameter."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_volatility(mock_sv_results, ax=ax)
        assert ax2 is ax
        plt.close(fig)
