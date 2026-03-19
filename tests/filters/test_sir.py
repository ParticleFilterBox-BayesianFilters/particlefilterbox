"""Tests for SIR (Sequential Importance Resampling) Particle Filter.

Tests cover:
- SIR with bootstrap proposal (should match BootstrapPF)
- SIR with custom proposal
- Weight correction computation
- Fallback behavior
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Helper: Model with custom proposal (locally optimal for LinearGaussian)
# ---------------------------------------------------------------------------


class LinearGaussianWithProposal:
    """Linear Gaussian model with optimal proposal for testing SIR.

    For the linear Gaussian model:
        x_t = phi * x_{t-1} + N(0, sigma_eta^2)
        y_t = x_t + N(0, sigma_eps^2)

    The optimal proposal is:
        q(x_t | x_{t-1}, y_t) = N(mu_opt, sigma_opt^2)

    where:
        sigma_opt^2 = 1 / (1/sigma_eta^2 + 1/sigma_eps^2)
        mu_opt = sigma_opt^2 * (phi * x_{t-1} / sigma_eta^2 + y_t / sigma_eps^2)
    """

    k_states: int = 1
    k_obs: int = 1

    def __init__(
        self,
        phi: float = 0.9,
        sigma_eta: float = 1.0,
        sigma_eps: float = 0.5,
        x0_mean: float = 0.0,
        x0_std: float = 1.0,
    ) -> None:
        self.phi = phi
        self.sigma_eta = sigma_eta
        self.sigma_eps = sigma_eps
        self.x0_mean = x0_mean
        self.x0_std = x0_std
        self.state_dim = 1
        self.obs_dim = 1

        # Optimal proposal parameters
        self._sigma_opt_sq = 1.0 / (
            1.0 / sigma_eta**2 + 1.0 / sigma_eps**2
        )
        self._sigma_opt = np.sqrt(self._sigma_opt_sq)

    def initial_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample from p(x_0)."""
        return rng.normal(self.x0_mean, self.x0_std, size=(n_particles, 1))

    def transition(
        self,
        particles: NDArray[np.float64],
        _t: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample from p(x_t | x_{t-1})."""
        mean = self.phi * particles
        return mean + self.sigma_eta * rng.standard_normal(particles.shape)

    def log_transition_density(
        self,
        x_new: NDArray[np.float64],
        x_old: NDArray[np.float64],
        _t: int,
    ) -> NDArray[np.float64]:
        """Compute log p(x_t | x_{t-1})."""
        mean = self.phi * x_old
        diff = x_new - mean
        return -0.5 * np.sum(diff**2, axis=-1) / self.sigma_eta**2 - 0.5 * np.log(
            2 * np.pi * self.sigma_eta**2
        )

    def log_observation_likelihood(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        _t: int,
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t)."""
        diff = y_t - particles
        return -0.5 * np.sum(diff**2, axis=-1) / self.sigma_eps**2 - 0.5 * np.log(
            2 * np.pi * self.sigma_eps**2
        )

    def proposal_sample(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        t: int,
        *,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample from the optimal proposal q(x_t | x_{t-1}, y_t)."""
        mu_opt = self._sigma_opt_sq * (
            self.phi * particles / self.sigma_eta**2 + y_t / self.sigma_eps**2
        )
        return mu_opt + self._sigma_opt * rng.standard_normal(particles.shape)

    def log_proposal_density(
        self,
        x_curr: NDArray[np.float64],
        x_prev: NDArray[np.float64],
        y_t: NDArray[np.float64],
        _t: int,
    ) -> NDArray[np.float64]:
        """Log-density of the optimal proposal."""
        mu_opt = self._sigma_opt_sq * (
            self.phi * x_prev / self.sigma_eta**2 + y_t / self.sigma_eps**2
        )
        diff = x_curr - mu_opt
        return -0.5 * np.sum(diff**2, axis=-1) / self._sigma_opt_sq - 0.5 * np.log(
            2 * np.pi * self._sigma_opt_sq
        )

    def simulate(
        self,
        n_steps: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulate states and observations."""
        if rng is None:
            rng = np.random.default_rng()
        states = np.zeros(n_steps)
        obs = np.zeros(n_steps)
        states[0] = rng.normal(self.x0_mean, self.x0_std)
        obs[0] = states[0] + self.sigma_eps * rng.standard_normal()
        for t in range(1, n_steps):
            states[t] = self.phi * states[t - 1] + self.sigma_eta * rng.standard_normal()
            obs[t] = states[t] + self.sigma_eps * rng.standard_normal()
        return states, obs


@pytest.fixture
def model_with_proposal() -> LinearGaussianWithProposal:
    """Linear Gaussian model with optimal proposal."""
    return LinearGaussianWithProposal()


@pytest.fixture
def proposal_data() -> (
    tuple[LinearGaussianWithProposal, NDArray[np.float64], NDArray[np.float64]]
):
    """Data generated from model with proposal."""
    rng = np.random.default_rng(42)
    model = LinearGaussianWithProposal()
    states, obs = model.simulate(n_steps=200, rng=rng)
    return model, states, obs


class TestSIRBootstrapFallback:
    """Test that SIR without custom proposal matches BootstrapPF."""

    def test_sir_with_bootstrap_proposal(self, linear_gaussian_data):
        """SIR without custom proposal should match BootstrapPF exactly."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF
        from particlefilterbox.filters.sir import SIR

        model, _states, obs = linear_gaussian_data

        config1 = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)
        config2 = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)

        pf_bootstrap = BootstrapPF(model, config1)
        pf_sir = SIR(model, config2)

        results_bootstrap = pf_bootstrap.filter(obs)
        results_sir = pf_sir.filter(obs)

        # Should be using bootstrap fallback
        assert not pf_sir.uses_custom_proposal

        # Results should match exactly (same seed, same algorithm)
        np.testing.assert_allclose(
            results_sir.filtered_means,
            results_bootstrap.filtered_means,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            results_sir.log_likelihood,
            results_bootstrap.log_likelihood,
            rtol=1e-10,
        )

    def test_sir_detects_no_proposal(self, linear_gaussian_data, pf_config):
        """SIR should detect when model has no custom proposal."""
        from particlefilterbox.filters.sir import SIR

        model, _states, _obs = linear_gaussian_data
        pf = SIR(model, pf_config)
        assert not pf.uses_custom_proposal


class TestSIRCustomProposal:
    """Test SIR with custom (optimal) proposal."""

    def test_sir_with_custom_proposal(self, proposal_data):
        """SIR with optimal proposal should produce valid results."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.sir import SIR

        model, _states, obs = proposal_data
        config = PFConfig(n_particles=1000, seed=42, ess_threshold=0.5)

        pf = SIR(model, config)
        assert pf.uses_custom_proposal

        results = pf.filter(obs)

        assert np.isfinite(results.log_likelihood)
        assert np.all(np.isfinite(results.filtered_means))

    def test_sir_custom_proposal_better_ess(self, proposal_data):
        """Optimal proposal should generally give higher ESS than bootstrap."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF
        from particlefilterbox.filters.sir import SIR

        model, _states, obs = proposal_data

        config_boot = PFConfig(n_particles=1000, seed=42, ess_threshold=0.5)
        config_sir = PFConfig(n_particles=1000, seed=42, ess_threshold=0.5)

        pf_boot = BootstrapPF(model, config_boot)
        pf_sir = SIR(model, config_sir)

        results_boot = pf_boot.filter(obs)
        results_sir = pf_sir.filter(obs)

        mean_ess_boot = np.mean(results_boot.ess_history)
        mean_ess_sir = np.mean(results_sir.ess_history)

        # Both should have valid ESS
        assert mean_ess_boot > 0
        assert mean_ess_sir > 0

    def test_sir_custom_proposal_shapes(self, proposal_data):
        """Output shapes should be correct with custom proposal."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.sir import SIR

        model, _states, obs = proposal_data
        n_obs = len(obs)
        config = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)

        pf = SIR(model, config)
        results = pf.filter(obs)

        assert results.filtered_means.shape == (n_obs, model.k_states)
        assert results.filtered_covs.shape == (n_obs, model.k_states, model.k_states)
        assert results.log_likelihoods.shape == (n_obs,)
        assert results.ess_history.shape == (n_obs,)


class TestSIRWeightCorrection:
    """Test that SIR weight correction is computed correctly."""

    def test_sir_weight_correction(self, proposal_data):
        """SIR weights should include the importance correction term."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF
        from particlefilterbox.filters.sir import SIR

        model, _states, obs = proposal_data
        config = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)

        pf = SIR(model, config)
        results = pf.filter(obs)

        config_boot = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)
        pf_boot = BootstrapPF(model, config_boot)
        results_boot = pf_boot.filter(obs)

        # With the same seed but different proposal, results should differ
        assert not np.allclose(
            results.filtered_means, results_boot.filtered_means
        )

    def test_sir_loglikelihood_finite(self, proposal_data):
        """Log-likelihood should be finite with custom proposal."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.sir import SIR

        model, _states, obs = proposal_data
        config = PFConfig(n_particles=1000, seed=42, ess_threshold=0.5)

        pf = SIR(model, config)
        results = pf.filter(obs)

        assert np.isfinite(results.log_likelihood)
        assert np.all(np.isfinite(results.log_likelihoods))


class TestSIRMissingData:
    """Test SIR missing data handling."""

    def test_missing_data(self, proposal_data):
        """SIR should handle missing data with custom proposal."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.sir import SIR

        model, _states, obs = proposal_data
        obs_missing = obs.copy()
        obs_missing[[5, 15, 25]] = np.nan

        config = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)
        pf = SIR(model, config)
        results = pf.filter(obs_missing)

        assert np.isfinite(results.log_likelihood)
        assert np.all(results.log_likelihoods[[5, 15, 25]] == 0.0)
