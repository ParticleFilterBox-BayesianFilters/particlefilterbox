"""Tests for FFBSm (Forward Filtering Backward Smoothing) particle smoother."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.smoothers.ffbsm import FFBSm
from particlefilterbox.core.smooth_results import ParticleSmootherResults

# Fixtures from conftest.py: linear_gaussian_data, mock_model, rng


class TestFFBSmRuns:
    """Basic functionality tests for FFBSm."""

    def test_ffbsm_runs(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that FFBSm runs without error on valid input."""
        filter_results, model = linear_gaussian_data
        smoother = FFBSm()
        result = smoother.smooth(filter_results, model)

        assert isinstance(result, ParticleSmootherResults)
        assert result.method == "FFBSm"
        assert result.computation_time_seconds > 0

    def test_ffbsm_output_shapes(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that FFBSm output has correct shapes."""
        filter_results, model = linear_gaussian_data
        smoother = FFBSm()
        result = smoother.smooth(filter_results, model)

        T = len(filter_results.particles_history)
        k = filter_results.particles_history[0].shape[1]
        N = filter_results.particles_history[0].shape[0]

        assert result.smoothed_mean.shape == (T, k)
        assert result.smoothed_cov.shape == (T, k, k)
        assert result.smoothed_weights.shape == (T, N)

    def test_ffbsm_2d(
        self, linear_gaussian_data_2d: tuple,
    ) -> None:
        """Test FFBSm with 2D state."""
        filter_results, model = linear_gaussian_data_2d
        smoother = FFBSm()
        result = smoother.smooth(filter_results, model)

        T = len(filter_results.particles_history)
        k = filter_results.particles_history[0].shape[1]

        assert result.smoothed_mean.shape == (T, k)
        assert result.smoothed_cov.shape == (T, k, k)


class TestFFBSmWeights:
    """Tests for FFBSm smoothed weight properties."""

    def test_smoothed_weights_sum_one(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that smoothed weights sum to 1 at each timestep."""
        filter_results, model = linear_gaussian_data
        smoother = FFBSm()
        result = smoother.smooth(filter_results, model)

        for t in range(result.n_timesteps):
            assert np.isclose(
                np.sum(result.smoothed_weights[t]), 1.0, atol=1e-10
            ), f"Weights at t={t} sum to {np.sum(result.smoothed_weights[t])}"

    def test_smoothed_weights_non_negative(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that smoothed weights are non-negative."""
        filter_results, model = linear_gaussian_data
        smoother = FFBSm()
        result = smoother.smooth(filter_results, model)

        assert np.all(result.smoothed_weights >= 0), "Negative smoothed weights found"

    def test_last_step_equals_filter(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that smoothed weights at T equal filtered weights at T.

        At the last timestep, smoothed = filtered because there is no future data.
        """
        filter_results, model = linear_gaussian_data
        smoother = FFBSm()
        result = smoother.smooth(filter_results, model)

        T = result.n_timesteps
        last_filtered = filter_results.weights_history[T - 1]
        last_smoothed = result.smoothed_weights[T - 1]

        np.testing.assert_allclose(
            last_smoothed,
            last_filtered,
            atol=1e-8,
            err_msg="Last step smoothed weights should equal filtered weights",
        )


class TestFFBSmVarianceProperty:
    """Tests for the fundamental smoothing variance property."""

    def test_smoothed_leq_filtered_variance(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that Var[x_t | y_{1:T}] <= Var[x_t | y_{1:t}] for all t.

        This is the fundamental property of smoothing: using future data
        should never increase the variance.
        """
        filter_results, model = linear_gaussian_data
        smoother = FFBSm()
        result = smoother.smooth(filter_results, model)

        T = result.n_timesteps
        for t in range(T):
            smooth_var = np.trace(result.smoothed_cov[t])
            filter_var = np.trace(filter_results.filtered_cov[t])

            assert smooth_var <= filter_var + 1e-6, (
                f"Variance property violated at t={t}: "
                f"smooth_var={smooth_var:.6f} > filter_var={filter_var:.6f}"
            )


class TestFFBSmAccuracy:
    """Tests for FFBSm estimation accuracy."""

    def test_smoothed_closer_to_true(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that smoothed estimates have lower MSE than filtered estimates.

        On average, the smoothed MSE should be lower because smoothing uses
        all data.
        """
        filter_results, model = linear_gaussian_data
        smoother = FFBSm()
        result = smoother.smooth(filter_results, model)

        true_states = filter_results.true_states
        T = result.n_timesteps

        # Compute MSE for filtered and smoothed (skip first few for burn-in)
        burn_in = 5
        filtered_mse = np.mean(
            (filter_results.filtered_mean[burn_in:T] - true_states[burn_in:T]) ** 2
        )
        smoothed_mse = np.mean(
            (result.smoothed_mean[burn_in:T] - true_states[burn_in:T]) ** 2
        )

        assert smoothed_mse < filtered_mse * 1.1, (
            f"Smoothed MSE ({smoothed_mse:.4f}) should be less than "
            f"filtered MSE ({filtered_mse:.4f})"
        )

    def test_smoothed_quantiles_exist(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that smoothed quantiles are computed."""
        filter_results, model = linear_gaussian_data
        smoother = FFBSm()
        result = smoother.smooth(filter_results, model)

        assert len(result.smoothed_quantiles) > 0
        for q, arr in result.smoothed_quantiles.items():
            assert arr.shape == result.smoothed_mean.shape, (
                f"Quantile {q} shape mismatch"
            )


class TestFFBSmValidation:
    """Tests for FFBSm input validation."""

    def test_requires_transition_density(self) -> None:
        """Test that FFBSm raises error if model lacks log_transition_density."""

        class NoTransitionModel:
            pass

        smoother = FFBSm()
        # We need a mock filter_results that passes validation
        from tests.smoothers.conftest import _generate_linear_gaussian_data

        filter_results, _ = _generate_linear_gaussian_data(T=5, N=10)

        with pytest.raises(AttributeError, match="log_transition_density"):
            smoother.smooth(filter_results, NoTransitionModel())

    def test_rejects_no_particles(
        self, mock_filter_results_no_particles: object, mock_model: object,
    ) -> None:
        """Test that FFBSm rejects filter results without particles."""
        smoother = FFBSm()
        with pytest.raises(ValueError, match="particles_history"):
            smoother.smooth(mock_filter_results_no_particles, mock_model)

    def test_rejects_no_weights(
        self, mock_filter_results_no_weights: object, mock_model: object,
    ) -> None:
        """Test that FFBSm rejects filter results without weights."""
        smoother = FFBSm()
        with pytest.raises(ValueError, match="weights_history"):
            smoother.smooth(mock_filter_results_no_weights, mock_model)
