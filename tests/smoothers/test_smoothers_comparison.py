"""Systematic comparison tests for all particle smoothers.

These tests verify that all smoothers produce consistent results and
satisfy the fundamental smoothing properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.smoothers.ffbsm import FFBSm
from particlefilterbox.smoothers.ffbsi import FFBSi
from particlefilterbox.smoothers.two_filter import TwoFilterSmoother
from particlefilterbox.smoothers.fixed_lag import FixedLagSmoother
from particlefilterbox.core.smooth_results import ParticleSmootherResults


class TestAllSmoothersOnLinearGaussian:
    """Test all smoothers on a linear Gaussian model.

    For a linear Gaussian model, the exact smoothed distribution is known
    (from the RTS smoother / Kalman smoother). All particle smoothers should
    converge to this distribution with enough particles.
    """

    def test_ffbsm_on_linear_gaussian(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test FFBSm on linear Gaussian produces reasonable results."""
        filter_results, model = linear_gaussian_data
        smoother = FFBSm()
        result = smoother.smooth(filter_results, model)

        true_states = filter_results.true_states
        smoothed_mse = np.mean(
            (result.smoothed_mean - true_states) ** 2
        )
        # MSE should be reasonable (depends on N, Q, R)
        assert smoothed_mse < 5.0, f"FFBSm MSE too high: {smoothed_mse:.4f}"

    def test_ffbsi_on_linear_gaussian(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test FFBSi on linear Gaussian produces reasonable results."""
        filter_results, model = linear_gaussian_data
        smoother = FFBSi(seed=42)
        result = smoother.smooth(filter_results, model, n_trajectories=200)

        true_states = filter_results.true_states
        smoothed_mse = np.mean(
            (result.smoothed_mean - true_states) ** 2
        )
        assert smoothed_mse < 5.0, f"FFBSi MSE too high: {smoothed_mse:.4f}"

    def test_two_filter_on_linear_gaussian(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test TwoFilterSmoother on linear Gaussian produces reasonable results."""
        filter_results, model = linear_gaussian_data
        smoother = TwoFilterSmoother(seed=42)
        result = smoother.smooth(filter_results, model)

        true_states = filter_results.true_states
        smoothed_mse = np.mean(
            (result.smoothed_mean - true_states) ** 2
        )
        assert smoothed_mse < 10.0, (
            f"TwoFilterSmoother MSE too high: {smoothed_mse:.4f}"
        )

    def test_fixed_lag_on_linear_gaussian(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test FixedLagSmoother on linear Gaussian produces reasonable results."""
        filter_results, model = linear_gaussian_data
        smoother = FixedLagSmoother(lag=10)
        result = smoother.smooth(filter_results, model)

        true_states = filter_results.true_states
        smoothed_mse = np.mean(
            (result.smoothed_mean - true_states) ** 2
        )
        assert smoothed_mse < 10.0, (
            f"FixedLagSmoother MSE too high: {smoothed_mse:.4f}"
        )

    def test_all_smoothers_consistent_means(
        self, linear_gaussian_data: tuple,
    ) -> None:
        """Test that FFBSm and FFBSi produce similar means.

        Both methods estimate the same distribution, so their means
        should be close (up to Monte Carlo noise).
        """
        filter_results, model = linear_gaussian_data

        ffbsm = FFBSm()
        result_m = ffbsm.smooth(filter_results, model)

        ffbsi = FFBSi(seed=42)
        result_i = ffbsi.smooth(filter_results, model, n_trajectories=500)

        # Average absolute difference
        mean_diff = np.mean(np.abs(result_m.smoothed_mean - result_i.smoothed_mean))
        assert mean_diff < 1.0, (
            f"FFBSm and FFBSi means differ by {mean_diff:.4f} on average"
        )


class TestSmoothedVarianceProperty:
    """Test the fundamental property: Var[x_t | y_{1:T}] <= Var[x_t | y_{1:t}].

    This property must hold for ALL smoothers and ALL timesteps (with
    some tolerance for Monte Carlo noise).
    """

    @pytest.mark.parametrize("smoother_class,kwargs", [
        (FFBSm, {}),
        (FFBSi, {"n_trajectories": 300}),
        (FixedLagSmoother, {}),
    ])
    def test_smoothed_variance_leq_filtered(
        self,
        linear_gaussian_data: tuple,
        smoother_class: type,
        kwargs: dict,
    ) -> None:
        """Test variance property for each smoother.

        Parameters
        ----------
        linear_gaussian_data : tuple
            Filter results and model.
        smoother_class : type
            Smoother class to test.
        kwargs : dict
            Additional kwargs for smooth().
        """
        filter_results, model = linear_gaussian_data

        if smoother_class == FFBSi:
            smoother = smoother_class(seed=42)  # type: ignore[call-arg]
        elif smoother_class == FixedLagSmoother:
            smoother = smoother_class(lag=10)  # type: ignore[call-arg]
        else:
            smoother = smoother_class()

        result = smoother.smooth(filter_results, model, **kwargs)

        T = result.n_timesteps
        violations = 0

        for t in range(T):
            smooth_var = np.trace(result.smoothed_cov[t])
            filter_var = np.trace(filter_results.filtered_cov[t])

            if smooth_var > filter_var + 0.5:
                violations += 1

        # Allow up to 15% violations due to Monte Carlo noise
        max_violations = max(int(T * 0.15), 3)
        assert violations <= max_violations, (
            f"{smoother_class.__name__}: Variance property violated "
            f"{violations}/{T} times (max: {max_violations})"
        )

    @pytest.mark.parametrize("smoother_class,kwargs", [
        (FFBSm, {}),
        (FFBSi, {"n_trajectories": 300}),
    ])
    def test_smoothed_mse_less_than_filtered(
        self,
        linear_gaussian_data: tuple,
        smoother_class: type,
        kwargs: dict,
    ) -> None:
        """Test that smoothed MSE is less than filtered MSE.

        On average, smoothed estimates should be more accurate than
        filtered estimates because they use all data.
        """
        filter_results, model = linear_gaussian_data

        if smoother_class == FFBSi:
            smoother = smoother_class(seed=42)  # type: ignore[call-arg]
        else:
            smoother = smoother_class()

        result = smoother.smooth(filter_results, model, **kwargs)

        true_states = filter_results.true_states
        T = result.n_timesteps

        # Skip first and last few timesteps for burn-in/edge effects
        burn = 5
        filtered_mse = np.mean(
            (filter_results.filtered_mean[burn:T - burn] - true_states[burn:T - burn]) ** 2
        )
        smoothed_mse = np.mean(
            (result.smoothed_mean[burn:T - burn] - true_states[burn:T - burn]) ** 2
        )

        # Smoothed should be at most slightly worse than filtered
        assert smoothed_mse < filtered_mse * 1.2, (
            f"{smoother_class.__name__}: Smoothed MSE ({smoothed_mse:.4f}) "
            f"should be <= filtered MSE ({filtered_mse:.4f})"
        )


class TestSmootherOutputFormats:
    """Test that all smoothers produce well-formed output."""

    @pytest.mark.parametrize("smoother_class,smoother_kwargs,smooth_kwargs", [
        (FFBSm, {}, {}),
        (FFBSi, {"seed": 42}, {"n_trajectories": 50}),
        (TwoFilterSmoother, {"seed": 42}, {}),
        (FixedLagSmoother, {"lag": 5}, {}),
    ])
    def test_output_is_particle_smoother_results(
        self,
        linear_gaussian_data: tuple,
        smoother_class: type,
        smoother_kwargs: dict,
        smooth_kwargs: dict,
    ) -> None:
        """Test that all smoothers return ParticleSmootherResults."""
        filter_results, model = linear_gaussian_data
        smoother = smoother_class(**smoother_kwargs)
        result = smoother.smooth(filter_results, model, **smooth_kwargs)

        assert isinstance(result, ParticleSmootherResults)
        assert result.n_timesteps > 0
        assert result.state_dim > 0
        assert result.method == smoother_class.__name__
        assert result.computation_time_seconds > 0

    @pytest.mark.parametrize("smoother_class,smoother_kwargs,smooth_kwargs", [
        (FFBSm, {}, {}),
        (FFBSi, {"seed": 42}, {"n_trajectories": 50}),
        (TwoFilterSmoother, {"seed": 42}, {}),
        (FixedLagSmoother, {"lag": 5}, {}),
    ])
    def test_summary_works(
        self,
        linear_gaussian_data: tuple,
        smoother_class: type,
        smoother_kwargs: dict,
        smooth_kwargs: dict,
    ) -> None:
        """Test that summary() works for all smoothers."""
        filter_results, model = linear_gaussian_data
        smoother = smoother_class(**smoother_kwargs)
        result = smoother.smooth(filter_results, model, **smooth_kwargs)

        summary = result.summary()
        assert isinstance(summary, dict)
        assert "method" in summary
        assert "n_timesteps" in summary

    @pytest.mark.parametrize("smoother_class,smoother_kwargs,smooth_kwargs", [
        (FFBSm, {}, {}),
        (FFBSi, {"seed": 42}, {"n_trajectories": 50}),
        (TwoFilterSmoother, {"seed": 42}, {}),
        (FixedLagSmoother, {"lag": 5}, {}),
    ])
    def test_to_dataframe_works(
        self,
        linear_gaussian_data: tuple,
        smoother_class: type,
        smoother_kwargs: dict,
        smooth_kwargs: dict,
    ) -> None:
        """Test that to_dataframe() works for all smoothers."""
        import pandas as pd

        filter_results, model = linear_gaussian_data
        smoother = smoother_class(**smoother_kwargs)
        result = smoother.smooth(filter_results, model, **smooth_kwargs)

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == result.n_timesteps


class TestSmootherImports:
    """Test that all smoothers are importable from the package."""

    def test_import_base(self) -> None:
        """Test BaseParticleSmoother import."""
        from particlefilterbox.smoothers import BaseParticleSmoother
        assert BaseParticleSmoother is not None

    def test_import_ffbsm(self) -> None:
        """Test FFBSm import."""
        from particlefilterbox.smoothers import FFBSm
        assert FFBSm is not None

    def test_import_ffbsi(self) -> None:
        """Test FFBSi import."""
        from particlefilterbox.smoothers import FFBSi
        assert FFBSi is not None

    def test_import_two_filter(self) -> None:
        """Test TwoFilterSmoother import."""
        from particlefilterbox.smoothers import TwoFilterSmoother
        assert TwoFilterSmoother is not None

    def test_import_fixed_lag(self) -> None:
        """Test FixedLagSmoother import."""
        from particlefilterbox.smoothers import FixedLagSmoother
        assert FixedLagSmoother is not None

    def test_all_exports(self) -> None:
        """Test that __all__ contains all expected exports."""
        from particlefilterbox import smoothers

        expected = [
            "BaseParticleSmoother",
            "FFBSm",
            "FFBSi",
            "TwoFilterSmoother",
            "FixedLagSmoother",
        ]
        for name in expected:
            assert name in smoothers.__all__, f"{name} not in smoothers.__all__"

    def test_smooth_results_import(self) -> None:
        """Test ParticleSmootherResults import."""
        from particlefilterbox.core.smooth_results import ParticleSmootherResults
        assert ParticleSmootherResults is not None
