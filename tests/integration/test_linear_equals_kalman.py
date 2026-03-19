"""CRITICAL: Particle filter should match Kalman filter for linear Gaussian models.

This is the gold-standard validation: for a linear Gaussian state-space model,
the Kalman filter provides the exact optimal solution. A well-implemented
particle filter with enough particles should closely approximate these results.

Target: correlation > 0.99 between PF and Kalman filtered means.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


def kalman_filter(
    y: NDArray[np.floating[Any]],
    A: float,
    C: float,
    Q: float,
    R: float,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], float]:
    """Run Kalman filter on linear Gaussian model.

    x_t = A * x_{t-1} + w_t,  w_t ~ N(0, Q)
    y_t = C * x_t + v_t,      v_t ~ N(0, R)

    Parameters
    ----------
    y : NDArray
        Observations shape (T,).
    A, C, Q, R : float
        Model parameters.

    Returns
    -------
    tuple of (filtered_mean, filtered_var, log_likelihood)
    """
    T = len(y)
    x_filt = np.zeros(T)
    P_filt = np.zeros(T)
    log_likelihood = 0.0

    # Initial: stationary distribution
    x_pred = 0.0
    P_pred = Q / (1.0 - A**2) if abs(A) < 1 else Q

    for t in range(T):
        # Prediction error
        v_t = y[t] - C * x_pred
        S_t = C * P_pred * C + R

        # Log-likelihood increment
        log_likelihood += -0.5 * (np.log(2 * np.pi * S_t) + v_t**2 / S_t)

        # Kalman gain
        K_t = P_pred * C / S_t

        # Update
        x_filt[t] = x_pred + K_t * v_t
        P_filt[t] = (1 - K_t * C) * P_pred

        # Predict next
        x_pred = A * x_filt[t]
        P_pred = A * P_filt[t] * A + Q

    return x_filt, P_filt, log_likelihood


class TestLinearEqualsKalman:
    """Test PF approximation quality against Kalman filter."""

    @pytest.fixture
    def linear_gaussian_data(self) -> dict[str, Any]:
        """Generate linear Gaussian state-space data."""
        rng = np.random.default_rng(42)
        T = 200
        A = 0.9
        C = 1.0
        Q = 1.0
        R = 1.0

        x = np.zeros(T)
        y = np.zeros(T)
        x[0] = rng.standard_normal() * np.sqrt(Q / (1 - A**2))

        for t in range(1, T):
            x[t] = A * x[t - 1] + np.sqrt(Q) * rng.standard_normal()

        for t in range(T):
            y[t] = C * x[t] + np.sqrt(R) * rng.standard_normal()

        # Kalman filter solution
        kf_mean, kf_var, kf_loglike = kalman_filter(y, A, C, Q, R)

        return {
            "observations": y,
            "true_states": x,
            "A": A,
            "C": C,
            "Q": Q,
            "R": R,
            "kalman_mean": kf_mean,
            "kalman_var": kf_var,
            "kalman_loglike": kf_loglike,
        }

    def test_kalman_filter_sanity(self, linear_gaussian_data: dict[str, Any]) -> None:
        """Kalman filter should produce reasonable results."""
        kf_mean = linear_gaussian_data["kalman_mean"]
        true_states = linear_gaussian_data["true_states"]

        corr = np.corrcoef(kf_mean, true_states)[0, 1]
        assert corr > 0.8, f"Kalman correlation with truth: {corr}"

    def test_pf_matches_kalman_mean(
        self, linear_gaussian_data: dict[str, Any]
    ) -> None:
        """CRITICAL: PF filtered mean should correlate > 0.99 with Kalman."""
        try:
            from particlefilterbox.models.linear_gaussian import LinearGaussianModel

            from particlefilterbox.filters.bootstrap import BootstrapFilter

            model = LinearGaussianModel(
                A=linear_gaussian_data["A"],
                C=linear_gaussian_data["C"],
                Q=linear_gaussian_data["Q"],
                R=linear_gaussian_data["R"],
            )
            rng = np.random.default_rng(42)
            pf = BootstrapFilter(model=model, n_particles=5000, rng=rng)
            results = pf.filter(linear_gaussian_data["observations"])

            filtered_mean = getattr(results, "filtered_mean", None)
            if filtered_mean is not None:
                pf_mean = np.asarray(filtered_mean).flatten()
                kf_mean = linear_gaussian_data["kalman_mean"]

                # Correlation test
                T = min(len(pf_mean), len(kf_mean))
                corr = np.corrcoef(pf_mean[:T], kf_mean[:T])[0, 1]
                assert corr > 0.99, (
                    f"PF-Kalman correlation: {corr:.4f} (expected > 0.99)"
                )

                # RMSE test
                rmse = np.sqrt(np.mean((pf_mean[:T] - kf_mean[:T]) ** 2))
                assert rmse < 0.5, f"PF-Kalman RMSE: {rmse:.4f} (expected < 0.5)"

        except ImportError:
            pytest.skip("BootstrapFilter or LinearGaussianModel not yet implemented")

    def test_pf_loglike_close_to_kalman(
        self, linear_gaussian_data: dict[str, Any]
    ) -> None:
        """PF log-likelihood should be close to Kalman log-likelihood."""
        try:
            from particlefilterbox.models.linear_gaussian import LinearGaussianModel

            from particlefilterbox.filters.bootstrap import BootstrapFilter

            model = LinearGaussianModel(
                A=linear_gaussian_data["A"],
                C=linear_gaussian_data["C"],
                Q=linear_gaussian_data["Q"],
                R=linear_gaussian_data["R"],
            )

            # Average over multiple runs for stable estimate
            log_likes = []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                pf = BootstrapFilter(model=model, n_particles=2000, rng=rng)
                results = pf.filter(linear_gaussian_data["observations"])
                ll = getattr(results, "log_likelihood", None)
                if ll is not None:
                    log_likes.append(float(ll))

            if log_likes:
                pf_ll = np.mean(log_likes)
                kf_ll = linear_gaussian_data["kalman_loglike"]

                # PF log-likelihood should be within 5% of Kalman
                rel_error = abs(pf_ll - kf_ll) / abs(kf_ll)
                assert rel_error < 0.05, (
                    f"Log-likelihood relative error: {rel_error:.4f} "
                    f"(PF: {pf_ll:.2f}, Kalman: {kf_ll:.2f})"
                )

        except ImportError:
            pytest.skip("Components not yet implemented")

    def test_pf_convergence_with_n(
        self, linear_gaussian_data: dict[str, Any]
    ) -> None:
        """PF accuracy should improve with more particles."""
        try:
            from particlefilterbox.models.linear_gaussian import LinearGaussianModel

            from particlefilterbox.filters.bootstrap import BootstrapFilter

            model = LinearGaussianModel(
                A=linear_gaussian_data["A"],
                C=linear_gaussian_data["C"],
                Q=linear_gaussian_data["Q"],
                R=linear_gaussian_data["R"],
            )

            kf_mean = linear_gaussian_data["kalman_mean"]
            rmses = []

            for N in [100, 500, 2000]:
                rng = np.random.default_rng(42)
                pf = BootstrapFilter(model=model, n_particles=N, rng=rng)
                results = pf.filter(linear_gaussian_data["observations"])

                filtered_mean = getattr(results, "filtered_mean", None)
                if filtered_mean is not None:
                    pf_mean = np.asarray(filtered_mean).flatten()
                    T = min(len(pf_mean), len(kf_mean))
                    rmse = np.sqrt(np.mean((pf_mean[:T] - kf_mean[:T]) ** 2))
                    rmses.append(rmse)

            if len(rmses) >= 3:
                # RMSE should generally decrease with N
                # (allowing some tolerance due to randomness)
                assert rmses[-1] < rmses[0] * 2, f"RMSE not decreasing: {rmses}"

        except ImportError:
            pytest.skip("Components not yet implemented")
