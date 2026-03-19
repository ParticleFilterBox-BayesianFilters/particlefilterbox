"""Tests for BoundedStates, Mixture, and ContinuousTime models."""

import numpy as np
import pytest

from particlefilterbox.models.bounded_states import BoundedStates
from particlefilterbox.models.mixture import Mixture
from particlefilterbox.models.continuous_time import ContinuousTime


class TestBoundedStates:
    """Test bounded state models."""

    def test_bounded_zlb(self) -> None:
        """State should always be >= 0 (ZLB)."""
        model = BoundedStates(
            bounds=[(0.0, None)],
            base_phi=0.95,
            base_sigma=0.5,  # Large sigma to test bound
        )
        result = model.simulate(T=500, seed=42)
        states = result["states"]
        assert np.all(states >= 0), f"ZLB violated: min = {states.min()}"

    def test_bounded_two_sided(self) -> None:
        """Test two-sided bounds."""
        model = BoundedStates(
            bounds=[(0.0, 1.0)],
            base_mu=np.array([0.5]),
            base_sigma=0.3,
        )
        result = model.simulate(T=500, seed=42)
        states = result["states"]
        assert np.all(states >= 0), f"Lower bound violated: min = {states.min()}"
        assert np.all(states <= 1), f"Upper bound violated: max = {states.max()}"

    def test_bounded_apply_bounds(self) -> None:
        model = BoundedStates(bounds=[(0.0, 10.0)])
        state = np.array([[-1.0], [5.0], [15.0]])
        bounded = model._apply_bounds(state)
        assert np.all(bounded >= 0)
        assert np.all(bounded <= 10)

    def test_bounded_default_prior(self) -> None:
        model = BoundedStates()
        priors = model.default_prior()
        assert "phi" in priors
        assert "sigma" in priors


class TestMixture:
    """Test mixture observation models."""

    def test_mixture_simulation(self) -> None:
        model = Mixture(
            n_components=3,
            weights=np.array([0.5, 0.3, 0.2]),
            component_means=np.array([0.0, 2.0, -2.0]),
            component_stds=np.array([0.5, 0.5, 0.5]),
        )
        result = model.simulate(T=500, seed=42)
        assert result["observations"].shape == (500, 1)
        assert np.all(np.isfinite(result["observations"]))

    def test_mixture_log_obs_density(self) -> None:
        model = Mixture(n_components=2)
        rng = np.random.default_rng(42)
        state = rng.standard_normal((100, 1))
        log_dens = model.log_observation_density(0.5, state)
        assert log_dens.shape == (100,)
        assert np.all(np.isfinite(log_dens))
        assert np.all(log_dens <= 0)

    def test_mixture_logsumexp_stability(self) -> None:
        """Log-sum-exp should handle large/small values."""
        model = Mixture(n_components=2)
        log_vals = np.array([[-1000, -999], [-1, -2]])
        result = model._logsumexp(log_vals)
        assert np.all(np.isfinite(result))

    def test_mixture_default_prior(self) -> None:
        model = Mixture()
        priors = model.default_prior()
        assert "phi" in priors


class TestContinuousTimeCIR:
    """Test CIR model."""

    def test_cir_positive(self) -> None:
        """CIR rate should stay positive."""
        model = ContinuousTime("cir")
        result = model.simulate(T=500, seed=42)
        states = result["states"]
        assert np.all(states > 0), f"CIR went negative: min = {states.min()}"

    def test_cir_mean_reversion(self) -> None:
        model = ContinuousTime("cir", params={
            "kappa": 2.0, "theta": 0.05, "sigma": 0.1
        })
        result = model.simulate(T=1000, seed=42)
        r = result["states"][:, 0]
        assert abs(np.mean(r) - 0.05) < 0.02

    def test_cir_default_prior(self) -> None:
        model = ContinuousTime("cir")
        priors = model.default_prior()
        assert "kappa" in priors


class TestContinuousTimeVasicek:
    """Test Vasicek model."""

    def test_vasicek_exact(self) -> None:
        """Vasicek should use exact Gaussian discretization."""
        model = ContinuousTime("vasicek")
        result = model.simulate(T=500, seed=42)
        assert result["states"].shape == (500, 1)
        assert np.all(np.isfinite(result["states"]))

    def test_vasicek_mean_reversion(self) -> None:
        model = ContinuousTime("vasicek", params={
            "kappa": 1.0, "theta": 0.03, "sigma": 0.02
        })
        result = model.simulate(T=1000, seed=42)
        r = result["states"][:, 0]
        assert abs(np.mean(r) - 0.03) < 0.01


class TestContinuousTimeHeston:
    """Test Heston model."""

    def test_heston_2d_state(self) -> None:
        model = ContinuousTime("heston")
        result = model.simulate(T=500, seed=42)
        assert result["states"].shape == (500, 2)
        v = result["states"][:, 1]
        assert np.all(v > 0), f"Heston variance went negative: min = {v.min()}"

    def test_heston_simulate(self) -> None:
        model = ContinuousTime("heston")
        result = model.simulate(T=252, seed=42)
        assert "prices" in result
        assert np.all(result["prices"] > 0)
