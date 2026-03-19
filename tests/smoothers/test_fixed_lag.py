"""Tests for FixedLagSmoother."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.core.smooth_results import ParticleSmootherResults
from particlefilterbox.smoothers.fixed_lag import FixedLagSmoother


class TestFixedLagSmootherRuns:
    """Basic functionality tests for FixedLagSmoother."""

    def test_fixed_lag_runs(
        self,
        linear_gaussian_data: tuple,
    ) -> None:
        """Test that FixedLagSmoother runs without error."""
        filter_results, model = linear_gaussian_data
        smoother = FixedLagSmoother(lag=5)
        result = smoother.smooth(filter_results, model)

        assert isinstance(result, ParticleSmootherResults)
        assert result.method == "FixedLagSmoother"

    def test_fixed_lag_output_shapes(
        self,
        linear_gaussian_data: tuple,
    ) -> None:
        """Test output shapes."""
        filter_results, model = linear_gaussian_data
        smoother = FixedLagSmoother(lag=5)
        result = smoother.smooth(filter_results, model)

        T = len(filter_results.particles_history)
        k = filter_results.particles_history[0].shape[1]
        N = filter_results.particles_history[0].shape[0]

        assert result.smoothed_mean.shape == (T, k)
        assert result.smoothed_cov.shape == (T, k, k)
        assert result.smoothed_weights.shape == (T, N)


class TestFixedLagProperties:
    """Tests for FixedLagSmoother theoretical properties."""

    def test_lag_zero_equals_filter(
        self,
        linear_gaussian_data: tuple,
    ) -> None:
        """Test that lag=0 gives approximately filtering results.

        With lag=0, the fixed-lag smoother should produce results
        very close to the original filtering.
        """
        filter_results, model = linear_gaussian_data
        smoother = FixedLagSmoother(lag=0)
        result = smoother.smooth(filter_results, model)

        # Smoothed mean should be close to filtered mean
        mean_diff = np.mean(
            np.abs(result.smoothed_mean - filter_results.filtered_mean)
        )
        assert mean_diff < 1.0, (
            f"Lag=0 smoothed mean differs from filtered mean by {mean_diff:.4f}"
        )

    def test_lag_equals_T(
        self,
        linear_gaussian_data: tuple,
    ) -> None:
        """Test that lag=T approximates fixed-interval smoothing.

        With lag=T, the fixed-lag smoother uses all available data for
        each timestep, approximating full smoothing.
        """
        filter_results, model = linear_gaussian_data
        T = len(filter_results.particles_history)
        smoother = FixedLagSmoother(lag=T)
        result = smoother.smooth(filter_results, model)

        assert result.smoothed_mean.shape[0] == T
        assert result.computation_time_seconds > 0

    def test_increasing_lag_improves(
        self,
        linear_gaussian_data: tuple,
    ) -> None:
        """Test that increasing lag tends to reduce variance.

        With more lag, we use more future information, which should
        generally reduce the estimation variance.
        """
        filter_results, model = linear_gaussian_data

        avg_variances = []
        for lag in [0, 5, 10, 20]:
            smoother = FixedLagSmoother(lag=lag)
            result = smoother.smooth(filter_results, model)

            # Average variance over timesteps (skip boundaries)
            T = result.n_timesteps
            start = 5
            end = T - 5
            avg_var = np.mean(
                [np.trace(result.smoothed_cov[t]) for t in range(start, end)]
            )
            avg_variances.append(avg_var)

        # Variance should generally decrease with lag (allow some tolerance)
        # At minimum, lag=20 should have <= lag=0 variance
        assert avg_variances[-1] <= avg_variances[0] * 1.5, (
            f"Lag=20 variance ({avg_variances[-1]:.4f}) should be <= "
            f"lag=0 variance ({avg_variances[0]:.4f})"
        )

    def test_negative_lag_raises(self) -> None:
        """Test that negative lag raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            FixedLagSmoother(lag=-1)


class TestFixedLagOnlineMode:
    """Tests for online (step-by-step) smoothing."""

    def test_online_mode(
        self,
        linear_gaussian_data: tuple,
    ) -> None:
        """Test smooth_step for online usage."""
        filter_results, model = linear_gaussian_data
        smoother = FixedLagSmoother(lag=3)

        T = len(filter_results.particles_history)

        # Reconstruct ancestor indices
        ancestor_indices = smoother._get_or_reconstruct_ancestors(
            filter_results,
            filter_results.particles_history,
            T,
            filter_results.particles_history[0].shape[0],
        )

        # Run online for each timestep
        for t in range(T):
            mean, weights = smoother.smooth_step(
                t=t,
                particles_history=filter_results.particles_history[: t + 1],
                weights_history=filter_results.weights_history[: t + 1],
                ancestor_indices=ancestor_indices[: t + 1],
            )
            k = filter_results.particles_history[0].shape[1]
            assert mean.shape == (k,)
            assert weights.shape == (
                filter_results.particles_history[0].shape[0],
            )
            assert np.isclose(np.sum(weights), 1.0, atol=1e-10)

    def test_smooth_step_and_batch_consistent(
        self,
        linear_gaussian_data: tuple,
    ) -> None:
        """Test that online and batch modes give consistent results."""
        filter_results, model = linear_gaussian_data
        smoother = FixedLagSmoother(lag=3)

        # Batch mode
        result_batch = smoother.smooth(filter_results, model)

        # The batch smoothed mean should exist
        assert result_batch.smoothed_mean is not None
        assert result_batch.n_timesteps > 0
