"""Tests for FFBSi (Forward Filtering Backward Simulation) particle smoother."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.core.smooth_results import ParticleSmootherResults
from particlefilterbox.smoothers.ffbsi import FFBSi
from particlefilterbox.smoothers.ffbsm import FFBSm


class TestFFBSiRuns:
    """Basic functionality tests for FFBSi."""

    def test_ffbsi_runs(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that FFBSi runs without error on valid input."""
        filter_results, model = linear_gaussian_data
        smoother = FFBSi(seed=42)
        result = smoother.smooth(filter_results, model, n_trajectories=50)

        assert isinstance(result, ParticleSmootherResults)
        assert result.method == "FFBSi"
        assert result.computation_time_seconds > 0

    def test_ffbsi_default_trajectories(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test FFBSi with default n_trajectories=100."""
        filter_results, model = linear_gaussian_data
        smoother = FFBSi(seed=42)
        result = smoother.smooth(filter_results, model)

        assert result.trajectories is not None
        assert result.trajectories.shape[0] == 100

    def test_trajectory_shapes(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that trajectories have correct shape (M, T, k)."""
        filter_results, model = linear_gaussian_data
        M = 50
        smoother = FFBSi(seed=42)
        result = smoother.smooth(filter_results, model, n_trajectories=M)

        T = len(filter_results.particles_history)
        k = filter_results.particles_history[0].shape[1]

        assert result.trajectories is not None
        assert result.trajectories.shape == (M, T, k)
        assert result.smoothed_mean.shape == (T, k)
        assert result.smoothed_cov.shape == (T, k, k)

    def test_ffbsi_2d(
        self, linear_gaussian_data_2d: tuple,
    ) -> None:
        """Test FFBSi with 2D state."""
        filter_results, model = linear_gaussian_data_2d
        smoother = FFBSi(seed=42)
        result = smoother.smooth(filter_results, model, n_trajectories=30)

        T = len(filter_results.particles_history)
        k = filter_results.particles_history[0].shape[1]

        assert result.trajectories is not None
        assert result.trajectories.shape == (30, T, k)


class TestFFBSiVarianceProperty:
    """Tests for the smoothing variance property with FFBSi."""

    def test_smoothed_leq_filtered_variance(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that Var[x_t | y_{1:T}] <= Var[x_t | y_{1:t}].

        FFBSi variance is estimated from trajectories. With enough trajectories,
        the smoothed variance should be less than or equal to filtered variance.
        """
        filter_results, model = linear_gaussian_data
        smoother = FFBSi(seed=42)
        result = smoother.smooth(filter_results, model, n_trajectories=200)

        T = result.n_timesteps
        violations = 0
        for t in range(T):
            smooth_var = np.trace(result.smoothed_cov[t])
            filter_var = np.trace(filter_results.filtered_cov[t])

            if smooth_var > filter_var + 0.5:
                violations += 1

        # Allow some violations due to Monte Carlo noise
        max_violations = int(T * 0.15)
        assert violations <= max_violations, (
            f"Variance property violated {violations}/{T} times "
            f"(max allowed: {max_violations})"
        )


class TestFFBSiConsistency:
    """Tests for consistency between FFBSi and FFBSm."""

    def test_ffbsi_mean_close_to_ffbsm(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that FFBSi and FFBSm produce similar smoothed means.

        Both methods estimate the same distribution, so with enough
        trajectories/particles, their means should be close.
        """
        filter_results, model = linear_gaussian_data

        # Run FFBSm (exact)
        ffbsm = FFBSm()
        result_m = ffbsm.smooth(filter_results, model)

        # Run FFBSi with many trajectories
        ffbsi = FFBSi(seed=42)
        result_i = ffbsi.smooth(filter_results, model, n_trajectories=500)

        # Compare means (allow for Monte Carlo noise)
        mean_diff = np.abs(result_m.smoothed_mean - result_i.smoothed_mean)
        mean_mean_diff = np.mean(mean_diff)

        assert mean_mean_diff < 0.5, (
            f"Average absolute difference between FFBSm and FFBSi means: "
            f"{mean_mean_diff:.4f} (should be < 0.5)"
        )


class TestFFBSiReproducibility:
    """Tests for FFBSi reproducibility with seeds."""

    def test_seed_reproducibility(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that the same seed produces the same results."""
        filter_results, model = linear_gaussian_data

        smoother1 = FFBSi(seed=123)
        result1 = smoother1.smooth(filter_results, model, n_trajectories=20)

        smoother2 = FFBSi(seed=123)
        result2 = smoother2.smooth(filter_results, model, n_trajectories=20)

        assert result1.trajectories is not None
        assert result2.trajectories is not None
        np.testing.assert_array_equal(result1.trajectories, result2.trajectories)

    def test_different_seeds_differ(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that different seeds produce different results."""
        filter_results, model = linear_gaussian_data

        smoother1 = FFBSi(seed=1)
        result1 = smoother1.smooth(filter_results, model, n_trajectories=20)

        smoother2 = FFBSi(seed=2)
        result2 = smoother2.smooth(filter_results, model, n_trajectories=20)

        assert result1.trajectories is not None
        assert result2.trajectories is not None
        assert not np.array_equal(result1.trajectories, result2.trajectories)


class TestFFBSiValidation:
    """Tests for FFBSi input validation."""

    def test_requires_transition_density(self) -> None:
        """Test that FFBSi raises error if model lacks log_transition_density."""

        class NoTransitionModel:
            pass

        smoother = FFBSi()
        from tests.smoothers.conftest import _generate_linear_gaussian_data

        filter_results, _ = _generate_linear_gaussian_data(T=5, N=10)

        with pytest.raises(AttributeError, match="log_transition_density"):
            smoother.smooth(filter_results, NoTransitionModel())

    def test_quantiles_computed(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that quantiles are computed from trajectories."""
        filter_results, model = linear_gaussian_data
        smoother = FFBSi(seed=42, quantiles=[0.1, 0.5, 0.9])
        result = smoother.smooth(filter_results, model, n_trajectories=50)

        assert 0.1 in result.smoothed_quantiles
        assert 0.5 in result.smoothed_quantiles
        assert 0.9 in result.smoothed_quantiles

        T = result.n_timesteps
        k = result.state_dim
        for q_arr in result.smoothed_quantiles.values():
            assert q_arr.shape == (T, k)
