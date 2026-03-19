"""Tests for particle visualization plots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from particlefilterbox.visualization.particle_plots import (
    plot_ancestral_tree,
    plot_particle_cloud,
    plot_particle_evolution,
    plot_particle_trajectories,
)


@pytest.fixture
def mock_results() -> SimpleNamespace:
    """Create mock FilterResults for testing."""
    rng = np.random.default_rng(42)
    T, N, d = 50, 100, 2
    particles = rng.standard_normal((T, N, d))
    weights = np.ones((T, N)) / N
    ancestors = np.tile(np.arange(N), (T, 1))
    return SimpleNamespace(
        particles=particles,
        weights=weights,
        ancestors=ancestors,
        n_particles=N,
    )


class TestParticlePlotsNoError:
    """Test that particle plot functions run without errors."""

    def test_plot_particle_cloud(self, mock_results: Any) -> None:
        """plot_particle_cloud should produce a figure without error."""
        fig, ax = plot_particle_cloud(mock_results, t=10)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_particle_trajectories(self, mock_results: Any) -> None:
        """plot_particle_trajectories should produce a figure without error."""
        fig, ax = plot_particle_trajectories(mock_results, n=10)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_ancestral_tree(self, mock_results: Any) -> None:
        """plot_ancestral_tree should produce a figure without error."""
        fig, ax = plot_ancestral_tree(mock_results, n=5)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_particle_evolution(self, mock_results: Any) -> None:
        """plot_particle_evolution should produce a figure without error."""
        fig, ax = plot_particle_evolution(mock_results)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_with_existing_ax(self, mock_results: Any) -> None:
        """All functions should accept an existing ax parameter."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_particle_cloud(mock_results, ax=ax)
        assert ax2 is ax
        plt.close(fig)
