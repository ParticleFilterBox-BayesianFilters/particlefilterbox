"""Tests for ParticleSmootherResults."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from particlefilterbox.core.smooth_results import ParticleSmootherResults


class TestParticleSmootherResults:
    """Tests for ParticleSmootherResults dataclass."""

    def _make_results(
        self,
        T: int = 10,
        k: int = 1,
        N: int = 50,
        with_trajectories: bool = False,
        M: int = 20,
    ) -> ParticleSmootherResults:
        """Helper to create a ParticleSmootherResults instance."""
        rng = np.random.default_rng(42)
        smoothed_mean = rng.standard_normal((T, k))
        smoothed_cov = np.zeros((T, k, k))
        for t in range(T):
            A = rng.standard_normal((k, k))
            smoothed_cov[t] = A @ A.T + 0.1 * np.eye(k)

        smoothed_weights = np.abs(rng.standard_normal((T, N)))
        smoothed_weights /= smoothed_weights.sum(axis=1, keepdims=True)

        quantiles = {
            0.025: rng.standard_normal((T, k)),
            0.5: smoothed_mean.copy(),
            0.975: rng.standard_normal((T, k)),
        }

        trajectories = None
        if with_trajectories:
            trajectories = rng.standard_normal((M, T, k))

        return ParticleSmootherResults(
            smoothed_mean=smoothed_mean,
            smoothed_cov=smoothed_cov,
            smoothed_quantiles=quantiles,
            smoothed_weights=smoothed_weights,
            trajectories=trajectories,
            method="TestSmoother",
            computation_time_seconds=1.23,
        )

    def test_smooth_results_shapes(self) -> None:
        """Test that shapes are correctly set."""
        results = self._make_results(T=10, k=2, N=50)
        assert results.n_timesteps == 10
        assert results.state_dim == 2
        assert results.n_particles == 50
        assert results.smoothed_mean.shape == (10, 2)
        assert results.smoothed_cov.shape == (10, 2, 2)
        assert results.smoothed_weights.shape == (10, 50)

    def test_smooth_results_1d_reshaping(self) -> None:
        """Test that 1D smoothed_mean is reshaped to (T, 1)."""
        mean_1d = np.random.randn(10)
        results = ParticleSmootherResults(
            smoothed_mean=mean_1d,
            smoothed_cov=np.zeros((10, 1, 1)),
        )
        assert results.smoothed_mean.shape == (10, 1)
        assert results.n_timesteps == 10
        assert results.state_dim == 1

    def test_smooth_results_summary(self) -> None:
        """Test summary() returns correct information."""
        results = self._make_results(T=10, k=1, N=50, with_trajectories=True, M=20)
        summary = results.summary()

        assert summary["method"] == "TestSmoother"
        assert summary["n_timesteps"] == 10
        assert summary["n_particles"] == 50
        assert summary["state_dim"] == 1
        assert summary["has_trajectories"] is True
        assert summary["n_trajectories"] == 20
        assert isinstance(summary["smoothed_mean_range"], tuple)
        assert len(summary["smoothed_mean_range"]) == 2
        assert isinstance(summary["smoothed_std_range"], tuple)
        assert summary["computation_time_seconds"] == pytest.approx(1.23)

    def test_smooth_results_summary_no_trajectories(self) -> None:
        """Test summary() when no trajectories are available."""
        results = self._make_results(with_trajectories=False)
        summary = results.summary()
        assert summary["has_trajectories"] is False
        assert summary["n_trajectories"] == 0

    def test_to_dataframe(self) -> None:
        """Test to_dataframe() returns correct DataFrame."""
        results = self._make_results(T=10, k=2, N=50)
        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "t" in df.columns
        assert "state_0_mean" in df.columns
        assert "state_1_mean" in df.columns
        assert "state_0_std" in df.columns
        assert "state_1_std" in df.columns
        # Check quantile columns
        assert "state_0_q0.025" in df.columns
        assert "state_0_q0.975" in df.columns
        assert "state_1_q0.500" in df.columns

    def test_to_dataframe_1d(self) -> None:
        """Test to_dataframe() with 1D state."""
        results = self._make_results(T=5, k=1, N=20)
        df = results.to_dataframe()

        assert len(df) == 5
        assert "state_0_mean" in df.columns
        assert "state_1_mean" not in df.columns

    def test_to_dataframe_t_column(self) -> None:
        """Test that t column contains correct indices."""
        results = self._make_results(T=15, k=1)
        df = results.to_dataframe()
        assert list(df["t"]) == list(range(15))

    def test_functional_estimate_requires_weights(self) -> None:
        """Test that functional_estimate raises error without weights."""
        results = ParticleSmootherResults(
            smoothed_mean=np.zeros((5, 1)),
            smoothed_cov=np.zeros((5, 1, 1)),
            smoothed_weights=np.array([]),
        )
        with pytest.raises(ValueError, match="Smoothed weights not available"):
            results.functional_estimate(lambda x: x[:, 0])

    def test_functional_estimate_requires_filter_results(self) -> None:
        """Test that functional_estimate raises error without filter results."""
        results = ParticleSmootherResults(
            smoothed_mean=np.zeros((5, 1)),
            smoothed_cov=np.zeros((5, 1, 1)),
            smoothed_weights=np.ones((5, 10)) / 10,
            filter_results=None,
        )
        with pytest.raises(ValueError, match="Filter results reference"):
            results.functional_estimate(lambda x: x[:, 0])
