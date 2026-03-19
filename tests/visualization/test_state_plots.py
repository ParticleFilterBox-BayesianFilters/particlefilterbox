"""Tests for state estimation plots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from particlefilterbox.visualization.state_plots import (
    plot_filtered_state,
    plot_filtered_vs_smoothed,
    plot_forecast,
    plot_observation_fit,
    plot_smoothed_state,
)


@pytest.fixture
def mock_results() -> SimpleNamespace:
    """Create mock FilterResults with states."""
    rng = np.random.default_rng(42)
    T, N, d = 50, 100, 2
    particles = rng.standard_normal((T, N, d))
    weights = np.ones((T, N)) / N
    filtered_mean = rng.standard_normal((T, d))
    smoothed_mean = rng.standard_normal((T, d))
    observations = rng.standard_normal((T, 1))
    return SimpleNamespace(
        particles=particles,
        weights=weights,
        filtered_mean=filtered_mean,
        smoothed_mean=smoothed_mean,
        observations=observations,
        n_particles=N,
    )


class TestStatePlotsNoError:
    """Test that state plot functions run without errors."""

    def test_plot_filtered_state(self, mock_results: Any) -> None:
        """plot_filtered_state should produce a figure without error."""
        fig, ax = plot_filtered_state(mock_results, state_idx=0)
        assert fig is not None
        plt.close(fig)

    def test_plot_filtered_state_with_true(self, mock_results: Any) -> None:
        """plot_filtered_state with true state overlay."""
        true_state = np.random.default_rng(42).standard_normal(50)
        fig, ax = plot_filtered_state(mock_results, true_state=true_state)
        assert fig is not None
        plt.close(fig)

    def test_plot_smoothed_state(self, mock_results: Any) -> None:
        """plot_smoothed_state should produce a figure without error."""
        fig, ax = plot_smoothed_state(mock_results, state_idx=0)
        assert fig is not None
        plt.close(fig)

    def test_plot_filtered_vs_smoothed(self, mock_results: Any) -> None:
        """plot_filtered_vs_smoothed should produce a figure without error."""
        fig, ax = plot_filtered_vs_smoothed(mock_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_observation_fit(self, mock_results: Any) -> None:
        """plot_observation_fit should produce a figure without error."""
        obs = np.random.default_rng(42).standard_normal(50)
        fig, ax = plot_observation_fit(mock_results, observations=obs)
        assert fig is not None
        plt.close(fig)

    def test_plot_forecast(self, mock_results: Any) -> None:
        """plot_forecast should produce a figure without error."""
        fig, ax = plot_forecast(mock_results)
        assert fig is not None
        plt.close(fig)

    def test_with_existing_ax(self, mock_results: Any) -> None:
        """State plots should accept an existing ax parameter."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_filtered_state(mock_results, ax=ax)
        assert ax2 is ax
        plt.close(fig)
