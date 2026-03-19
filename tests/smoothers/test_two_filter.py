"""Tests for TwoFilterSmoother."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.core.smooth_results import ParticleSmootherResults
from particlefilterbox.smoothers.two_filter import TwoFilterSmoother


class TestTwoFilterSmootherRuns:
    """Basic functionality tests for TwoFilterSmoother."""

    def test_two_filter_runs(
        self,
        linear_gaussian_data: tuple,
    ) -> None:
        """Test that TwoFilterSmoother runs without error."""
        filter_results, model = linear_gaussian_data
        smoother = TwoFilterSmoother(seed=42)
        result = smoother.smooth(filter_results, model)

        assert isinstance(result, ParticleSmootherResults)
        assert result.method == "TwoFilterSmoother"

    def test_two_filter_output_shapes(
        self,
        linear_gaussian_data: tuple,
    ) -> None:
        """Test output shapes."""
        filter_results, model = linear_gaussian_data
        smoother = TwoFilterSmoother()
        result = smoother.smooth(filter_results, model)

        T = len(filter_results.particles_history)
        k = filter_results.particles_history[0].shape[1]
        N = filter_results.particles_history[0].shape[0]

        assert result.smoothed_mean.shape == (T, k)
        assert result.smoothed_cov.shape == (T, k, k)
        assert result.smoothed_weights.shape == (T, N)


class TestTwoFilterSmootherVariance:
    """Tests for variance property of TwoFilterSmoother."""

    def test_two_filter_smoothed_variance(
        self,
        linear_gaussian_data: tuple,
    ) -> None:
        """Test that smoothed variance <= filtered variance for most timesteps.

        The two-filter smoother may have some numerical issues, so we allow
        a small fraction of violations.
        """
        filter_results, model = linear_gaussian_data
        smoother = TwoFilterSmoother(seed=42)
        result = smoother.smooth(filter_results, model)

        T = result.n_timesteps
        violations = 0
        for t in range(T):
            smooth_var = np.trace(result.smoothed_cov[t])
            filter_var = np.trace(filter_results.filtered_cov[t])
            if smooth_var > filter_var + 0.5:
                violations += 1

        max_violations = int(T * 0.2)
        assert violations <= max_violations, (
            f"Variance property violated {violations}/{T} times "
            f"(max allowed: {max_violations})"
        )

    def test_two_filter_weights_sum_one(
        self,
        linear_gaussian_data: tuple,
    ) -> None:
        """Test that smoothed weights sum to 1."""
        filter_results, model = linear_gaussian_data
        smoother = TwoFilterSmoother()
        result = smoother.smooth(filter_results, model)

        for t in range(result.n_timesteps):
            assert np.isclose(
                np.sum(result.smoothed_weights[t]), 1.0, atol=1e-10
            )


class TestTwoFilterSmootherValidation:
    """Tests for TwoFilterSmoother input validation."""

    def test_requires_observation_density(self) -> None:
        """Test error when model lacks log_observation_density."""

        class BadModel:
            def log_transition_density(
                self, x_new: np.ndarray, x_old: np.ndarray, t: int
            ) -> np.ndarray:
                return np.zeros(1)

        smoother = TwoFilterSmoother()
        from tests.smoothers.conftest import _generate_linear_gaussian_data

        filter_results, _ = _generate_linear_gaussian_data(T=5, N=10)

        with pytest.raises(AttributeError, match="log_observation_density"):
            smoother.smooth(filter_results, BadModel())
