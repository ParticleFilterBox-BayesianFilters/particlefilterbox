"""Tests for DSGE model."""

import numpy as np
import pytest

from particlefilterbox.models.dsge import DSGE


class TestDSGEFirstOrder:
    """Test first-order (linear) DSGE."""

    def test_dsge_first_order_linear(self) -> None:
        A = np.array([[0.8, 0.1], [-0.2, 0.9]])
        B = np.eye(2) * 0.1
        C = np.eye(2)
        model = DSGE.from_matrices(A, B, C, order=1)

        result = model.simulate(T=200, seed=42)
        assert result["states"].shape == (200, 2)
        assert result["observations"].shape == (200, 2)
        assert np.all(np.isfinite(result["states"]))

    def test_dsge_from_matrices(self) -> None:
        A = np.array([[0.9]])
        B = np.array([[0.1]])
        C = np.array([[1.0]])
        model = DSGE.from_matrices(A, B, C)
        assert model.k_states == 1
        assert model.k_obs == 1


class TestDSGESecondOrder:
    """Test second-order (nonlinear) DSGE."""

    def test_dsge_second_order(self) -> None:
        A = np.array([[0.8, 0.1], [-0.2, 0.9]])
        B = np.eye(2) * 0.05
        C = np.eye(2)
        model = DSGE.from_matrices(A, B, C, order=2, sigma_scale=0.5)

        result = model.simulate(T=200, seed=42)
        assert result["states"].shape == (200, 2)
        assert np.all(np.isfinite(result["states"]))

    def test_dsge_second_order_with_tensor(self) -> None:
        k = 2
        A = np.eye(k) * 0.8
        B = np.eye(k) * 0.05
        C = np.eye(k)
        Q = np.zeros((k, k, k))
        Q[0, 0, 0] = 0.1
        Q[1, 1, 1] = 0.1
        model = DSGE.from_matrices(
            A, B, C, order=2, quadratic_terms=Q
        )
        result = model.simulate(T=100, seed=42)
        assert np.all(np.isfinite(result["states"]))


class TestDSGEZLB:
    """Test Zero Lower Bound."""

    def test_dsge_zlb(self) -> None:
        """Interest rate should always be >= 0 with ZLB."""
        A = np.array([
            [0.8, 0.1, 0.0],
            [-0.2, 0.9, 0.3],
            [0.0, -0.5, 0.7],
        ])
        B = np.eye(3) * 0.3  # Large shocks to hit ZLB
        C = np.eye(3)
        model = DSGE.from_matrices(
            A, B, C, order=1, zlb=True, zlb_index=2
        )

        result = model.simulate(T=500, seed=42)
        interest_rates = result["observations"][:, 2]
        assert np.all(interest_rates >= 0), (
            f"ZLB violated: min rate = {interest_rates.min():.6f}"
        )
        # Should have some zero observations (hitting ZLB)
        n_at_zlb = np.sum(interest_rates == 0.0)
        assert n_at_zlb > 0, "ZLB never binding - increase shock size"


class TestDSGEIRF:
    """Test impulse response functions."""

    def test_dsge_irf(self) -> None:
        A = np.array([[0.9, 0.0], [0.1, 0.8]])
        B = np.eye(2) * 0.1
        C = np.eye(2)
        model = DSGE.from_matrices(A, B, C, order=1)

        irf = model.impulse_response(
            shock=0, periods=20, n_particles=500, seed=42
        )
        assert irf.shape == (20, 2)
        # IRF should decay
        assert abs(irf[0, 0]) > abs(irf[-1, 0])

    def test_dsge_default_prior(self) -> None:
        model = DSGE()
        priors = model.default_prior()
        assert isinstance(priors, dict)
        assert len(priors) > 0
