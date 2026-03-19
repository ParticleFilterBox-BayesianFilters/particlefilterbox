"""Tests for weight diagnostic plots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from particlefilterbox.visualization.weight_plots import (
    plot_ess_timeline,
    plot_weight_entropy,
    plot_weight_histogram,
    plot_weight_max,
)


@pytest.fixture
def mock_results() -> SimpleNamespace:
    """Create mock FilterResults with weights."""
    rng = np.random.default_rng(42)
    T, N = 50, 100
    weights = rng.dirichlet(np.ones(N), size=T)
    return SimpleNamespace(
        weights=weights,
        n_particles=N,
    )


class TestWeightPlotsNoError:
    """Test that weight plot functions run without errors."""

    def test_plot_ess_timeline(self, mock_results: Any) -> None:
        """plot_ess_timeline should produce a figure without error."""
        fig, ax = plot_ess_timeline(mock_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_ess_timeline_with_threshold(self, mock_results: Any) -> None:
        """plot_ess_timeline with explicit threshold."""
        fig, ax = plot_ess_timeline(mock_results, threshold=30.0)
        assert fig is not None
        plt.close(fig)

    def test_plot_weight_histogram(self, mock_results: Any) -> None:
        """plot_weight_histogram should produce a figure without error."""
        fig, ax = plot_weight_histogram(mock_results, t=10)
        assert fig is not None
        plt.close(fig)

    def test_plot_weight_entropy(self, mock_results: Any) -> None:
        """plot_weight_entropy should produce a figure without error."""
        fig, ax = plot_weight_entropy(mock_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_weight_max(self, mock_results: Any) -> None:
        """plot_weight_max should produce a figure without error."""
        fig, ax = plot_weight_max(mock_results)
        assert fig is not None
        plt.close(fig)

    def test_with_existing_ax(self, mock_results: Any) -> None:
        """Weight plots should accept an existing ax parameter."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_ess_timeline(mock_results, ax=ax)
        assert ax2 is ax
        plt.close(fig)
