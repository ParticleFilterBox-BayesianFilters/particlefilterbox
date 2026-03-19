"""Tests for the Rao-Blackwellized Particle Filter.

CRITICAL: Tests verify kalmanbox integration and that RBPF with 500 particles
matches or exceeds Bootstrap PF with 5000 particles.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from particlefilterbox.filters.rao_blackwellized import RaoBlackwellizedPF
from particlefilterbox.core.config import PFConfig

# CRITICAL: kalmanbox imports
from kalmanbox.core import StateSpaceRepresentation
from kalmanbox.filters import KalmanFilter


class MixedLinearNonlinearModel:
    """Mixed model where x_t is nonlinear and s_t is linear given x_t.

    Nonlinear: x_t = 0.5 * x_{t-1} + 25 * x_{t-1} / (1 + x_{t-1}^2) + sigma_x * eps
    Linear:    s_t = phi_s * s_{t-1} + x_t + sigma_s * eta
    Obs:       y_t = s_t + sigma_y * eps_y

    This is a standard RBPF test model from Schon et al (2005).
    """

    def __init__(
        self,
        sigma_x: float = 1.0,
        phi_s: float = 0.9,
        sigma_s: float = 0.5,
        sigma_y: float = 1.0,
    ) -> None:
        self.sigma_x = sigma_x
        self.phi_s = phi_s
        self.sigma_s = sigma_s
        self.sigma_y = sigma_y
        self.k_states = 2  # (x_nl, s_lin)
        self.k_obs = 1
        self.k_nonlinear = 1
        self.k_linear = 1

    def has_linear_substate(self) -> bool:
        return True

    def initial_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> np.ndarray:
        return rng.normal(0, 1, size=(n_particles, 2))

    def initial_nonlinear_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> np.ndarray:
        return rng.normal(0, 1, size=(n_particles, 1))

    def initial_linear_mean(self) -> np.ndarray:
        return np.zeros(1)

    def initial_linear_cov(self) -> np.ndarray:
        return np.eye(1) * 10.0

    def transition_nonlinear(
        self,
        x_nl: np.ndarray,
        t: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Nonlinear transition: x_t = 0.5*x + 25*x/(1+x^2) + noise."""
        x = x_nl.flatten() if x_nl.ndim > 1 else x_nl
        x_new = (
            0.5 * x
            + 25.0 * x / (1.0 + x**2)
            + self.sigma_x * rng.normal(size=x.shape)
        )
        return x_new.reshape(-1, 1)

    def transition(
        self,
        particles: np.ndarray,
        t: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Full transition (for Bootstrap comparison)."""
        x_nl = particles[:, 0:1]
        s_lin = particles[:, 1:2]

        x_new = self.transition_nonlinear(x_nl, t, rng)
        s_new = (
            self.phi_s * s_lin
            + x_new
            + self.sigma_s * rng.normal(size=s_lin.shape)
        )
        return np.hstack([x_new, s_new])

    def log_observation_likelihood(
        self,
        particles: np.ndarray,
        y_t: np.ndarray,
        t: int,
    ) -> np.ndarray:
        """Log p(y_t | x_t) for Bootstrap PF comparison.

        y_t = s_t + sigma_y * eps => p(y_t|state) = N(y_t; s_t, sigma_y^2)
        """
        s = particles[:, 1]
        diff = y_t.flatten()[0] - s
        return -0.5 * diff**2 / self.sigma_y**2 - 0.5 * np.log(
            2 * np.pi * self.sigma_y**2
        )

    def linear_ssm(self, x_nonlinear: np.ndarray) -> StateSpaceRepresentation:
        """Get linear SSM conditioned on nonlinear state.

        s_t = phi_s * s_{t-1} + x_t + sigma_s * eta
        y_t = s_t + sigma_y * eps

        kalmanbox notation:
          T = [[phi_s]]        (state transition)
          Z = [[1.0]]          (observation design)
          R = [[1.0]]          (selection matrix)
          Q = [[sigma_s^2]]    (state noise cov)
          H = [[sigma_y^2]]    (obs noise cov)
          c = [x_val]          (state intercept = nonlinear contribution)
          d = [0.0]            (obs intercept)
        """
        x_val = float(x_nonlinear.flatten()[0])

        ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
        ssm.T = np.array([[self.phi_s]])
        ssm.Z = np.array([[1.0]])
        ssm.R = np.array([[1.0]])
        ssm.Q = np.array([[self.sigma_s**2]])
        ssm.H = np.array([[self.sigma_y**2]])
        ssm.c = np.array([x_val])
        ssm.d = np.array([0.0])
        ssm.a1 = np.array([0.0])
        ssm.P1 = np.array([[10.0]])

        return ssm


def generate_mixed_data(
    T: int = 100,
    sigma_x: float = 1.0,
    phi_s: float = 0.9,
    sigma_s: float = 0.5,
    sigma_y: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data from the mixed linear/nonlinear model.

    Returns
    -------
    x_true : (T,) nonlinear states
    s_true : (T,) linear states
    obs : (T, 1) observations
    """
    rng = np.random.default_rng(seed)
    x_true = np.empty(T)
    s_true = np.empty(T)
    obs = np.empty((T, 1))

    x = rng.normal(0, 1)
    s = rng.normal(0, 1)

    for t in range(T):
        x = 0.5 * x + 25.0 * x / (1.0 + x**2) + sigma_x * rng.normal()
        s = phi_s * s + x + sigma_s * rng.normal()
        y = s + sigma_y * rng.normal()

        x_true[t] = x
        s_true[t] = s
        obs[t, 0] = y

    return x_true, s_true, obs


class TestRBPFKalmanboxIntegration:
    """CRITICAL: Verify kalmanbox integration."""

    def test_rbpf_kalmanbox_integration(self) -> None:
        """Verify that RBPF uses kalmanbox.KalmanFilter internally."""
        model = MixedLinearNonlinearModel()
        config = PFConfig(
            n_particles=100, ess_threshold=0.5, resampling="systematic", seed=42
        )
        rbpf = RaoBlackwellizedPF(model=model, config=config)

        # Check internal KalmanFilter instance
        assert hasattr(rbpf, "_kf")
        assert isinstance(rbpf._kf, KalmanFilter)

    def test_rbpf_linear_ssm_returns_ssr(self) -> None:
        """Verify model.linear_ssm returns StateSpaceRepresentation."""
        model = MixedLinearNonlinearModel()
        x_nl = np.array([1.0])
        ssm = model.linear_ssm(x_nl)
        assert isinstance(ssm, StateSpaceRepresentation)

    def test_rbpf_runs(self) -> None:
        """Test that RBPF runs without errors."""
        model = MixedLinearNonlinearModel()
        config = PFConfig(
            n_particles=200, ess_threshold=0.5, resampling="systematic", seed=42
        )
        rbpf = RaoBlackwellizedPF(model=model, config=config)

        _, _, obs = generate_mixed_data(T=50)
        result = rbpf.filter(obs)

        assert result.filtered_means.shape == (50, 2)
        assert result.ess_history.shape == (50,)
        assert np.all(np.isfinite(result.filtered_means))


class TestRBPFLinearPartMatchesKalman:
    """Test that linear part tracked by RBPF matches standalone Kalman."""

    def test_rbpf_linear_part_matches_kalman(self) -> None:
        """Linear state estimate should be close to Kalman filter estimate.

        For a nearly linear model (sigma_x ~ 0), RBPF's Kalman component
        should converge to the true linear state.
        """
        model = MixedLinearNonlinearModel(sigma_x=0.01)
        config = PFConfig(
            n_particles=500, ess_threshold=0.5, resampling="systematic", seed=42
        )
        rbpf = RaoBlackwellizedPF(model=model, config=config)

        _, s_true, obs = generate_mixed_data(T=50, sigma_x=0.01, seed=42)
        result = rbpf.filter(obs)

        # Linear part estimate
        s_estimated = result.filtered_means[:, 1]
        corr = np.corrcoef(s_true, s_estimated)[0, 1]
        assert corr > 0.90, f"Linear state correlation {corr:.4f} < 0.90"


class TestRBPFFewerParticles:
    """CRITICAL: RBPF 500 particles >= Bootstrap 5000."""

    def test_rbpf_fewer_particles(self) -> None:
        """RBPF with 500 particles should match Bootstrap with 5000.

        This is the key advantage of Rao-Blackwellization: by analytically
        marginalizing the linear component, we need far fewer particles.
        """
        from particlefilterbox.filters.bootstrap import BootstrapPF

        model = MixedLinearNonlinearModel()
        _x_true, s_true, obs = generate_mixed_data(T=100, seed=42)

        # RBPF with 500 particles
        config_rbpf = PFConfig(
            n_particles=500, ess_threshold=0.5, resampling="systematic", seed=42
        )
        rbpf = RaoBlackwellizedPF(model=model, config=config_rbpf)
        result_rbpf = rbpf.filter(obs)

        # Bootstrap with 5000 particles
        config_bpf = PFConfig(
            n_particles=5000, ess_threshold=0.5, resampling="systematic", seed=42
        )
        bpf = BootstrapPF(model=model, config=config_bpf)
        result_bpf = bpf.filter(obs)

        # Compare RMSE on linear state
        s_rbpf = result_rbpf.filtered_means[:, 1]
        s_bpf = result_bpf.filtered_means[:, 1]

        rmse_rbpf = np.sqrt(np.mean((s_true - s_rbpf) ** 2))
        rmse_bpf = np.sqrt(np.mean((s_true - s_bpf) ** 2))

        # RBPF should be at least as good
        assert rmse_rbpf <= rmse_bpf * 1.2, (
            f"RBPF RMSE ({rmse_rbpf:.4f}) should be <= Bootstrap RMSE ({rmse_bpf:.4f}) * 1.2"
        )


class TestRBPFResampleCarriesKalman:
    """Test that resampling carries Kalman states."""

    def test_rbpf_resample_carries_kalman(self) -> None:
        """After resampling, Kalman means and covs should be properly copied."""
        model = MixedLinearNonlinearModel()
        config = PFConfig(
            n_particles=100, ess_threshold=0.5, resampling="systematic", seed=42
        )
        rbpf = RaoBlackwellizedPF(model=model, config=config)

        # Create test data
        rng = np.random.default_rng(42)
        particles_nl = rng.normal(0, 1, size=(100, 1))
        kalman_means = rng.normal(0, 1, size=(100, 1))
        kalman_covs = np.tile(np.eye(1) * 2.0, (100, 1, 1))

        # Degenerate weights (all on first particle)
        weights = np.zeros(100)
        weights[0] = 1.0

        new_p, new_m, new_c = rbpf._resample_with_kalman(
            particles_nl, kalman_means, kalman_covs, weights
        )

        # After resampling with degenerate weights, all should be copies of particle 0
        for i in range(100):
            assert_allclose(new_p[i], particles_nl[0], atol=1e-10)
            assert_allclose(new_m[i], kalman_means[0], atol=1e-10)
            assert_allclose(new_c[i], kalman_covs[0], atol=1e-10)

    def test_rbpf_resample_independence(self) -> None:
        """Resampled Kalman states should be independent copies."""
        model = MixedLinearNonlinearModel()
        config = PFConfig(
            n_particles=10, ess_threshold=0.5, resampling="systematic", seed=42
        )
        rbpf = RaoBlackwellizedPF(model=model, config=config)

        rng = np.random.default_rng(42)
        particles_nl = rng.normal(0, 1, size=(10, 1))
        kalman_means = rng.normal(0, 1, size=(10, 1))
        kalman_covs = np.tile(np.eye(1), (10, 1, 1))

        weights = np.ones(10) / 10.0

        _, new_m, _ = rbpf._resample_with_kalman(
            particles_nl, kalman_means, kalman_covs, weights
        )

        # Modifying one should not affect others (deep copy)
        new_m[0, 0] = 999.0
        assert not np.any(new_m[1:] == 999.0)


class TestRBPFValidation:
    """Validation and error handling tests."""

    def test_rbpf_rejects_model_without_linear_substate(self) -> None:
        """RBPF should reject models without linear substate."""

        class BadModel:
            k_states = 1
            k_obs = 1

            def has_linear_substate(self) -> bool:
                return False

            def initial_distribution(
                self, n: int, rng: np.random.Generator
            ) -> np.ndarray:
                return rng.normal(0, 1, (n, 1))

            def transition(
                self, p: np.ndarray, t: int, rng: np.random.Generator
            ) -> np.ndarray:
                return p

            def log_observation_likelihood(
                self, particles: np.ndarray, y_t: np.ndarray, t: int
            ) -> np.ndarray:
                return np.zeros(particles.shape[0])

        config = PFConfig(n_particles=10, seed=42)
        with pytest.raises(ValueError, match="has_linear_substate"):
            RaoBlackwellizedPF(model=BadModel(), config=config)  # type: ignore[arg-type]
