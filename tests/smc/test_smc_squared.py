"""Tests for SMC^2 (SMCSquared)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from particlefilterbox.smc.results import SMCResults
from particlefilterbox.smc.smc_squared import InternalPF, SMCSquared

# =============================================================
# Simple local-level model for testing
# y_t = x_t + sigma_obs * e_t
# x_t = phi * x_{t-1} + sigma_state * eta_t
# =============================================================


class LocalLevelModel:
    """Simple local-level state-space model for testing.

    Parameters:
        theta[0] = phi (persistence)
        theta[1] = sigma_state
        theta[2] = sigma_obs
    """

    def initial_state_sample(
        self,
        theta: NDArray[np.floating[Any]],
        n_particles: int,
        rng: np.random.Generator,
    ) -> NDArray[np.floating[Any]]:
        sigma_state = abs(theta[1])
        return rng.standard_normal((n_particles, 1)) * sigma_state

    def transition_sample(
        self,
        x_prev: NDArray[np.floating[Any]],
        theta: NDArray[np.floating[Any]],
        rng: np.random.Generator,
    ) -> NDArray[np.floating[Any]]:
        phi = theta[0]
        sigma_state = abs(theta[1])
        noise = rng.standard_normal(x_prev.shape) * sigma_state
        return phi * x_prev + noise

    def observation_logpdf(
        self,
        y_t: NDArray[np.floating[Any]],
        x_t: NDArray[np.floating[Any]],
        theta: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        sigma_obs = abs(theta[2])
        diff = y_t[np.newaxis, :] - x_t
        return np.sum(
            -0.5 * (diff / sigma_obs) ** 2
            - 0.5 * np.log(2 * np.pi * sigma_obs**2),
            axis=1,
        )


@dataclass
class LocalLevelPrior:
    """Prior for local-level model parameters."""

    def logpdf(self, theta: NDArray[np.floating[Any]]) -> float:
        phi, sigma_s, sigma_o = theta[0], theta[1], theta[2]
        # phi ~ Uniform(-1, 1)
        if abs(phi) >= 1.0:
            return -np.inf
        # sigma_state ~ HalfNormal(0, 1)
        if sigma_s <= 0:
            return -np.inf
        # sigma_obs ~ HalfNormal(0, 1)
        if sigma_o <= 0:
            return -np.inf
        log_p = -0.5 * sigma_s**2 - 0.5 * sigma_o**2
        return float(log_p)

    def sample(
        self, rng: np.random.Generator
    ) -> NDArray[np.floating[Any]]:
        phi = rng.uniform(-0.99, 0.99)
        sigma_s = abs(rng.standard_normal()) * 0.5 + 0.1
        sigma_o = abs(rng.standard_normal()) * 0.5 + 0.1
        return np.array([phi, sigma_s, sigma_o])


@pytest.fixture
def local_level_data() -> tuple[
    NDArray[np.floating[Any]], dict[str, float]
]:
    """Generate data from a local-level model with known parameters."""
    rng = np.random.default_rng(123)
    n_obs = 50
    true_phi = 0.9
    true_sigma_state = 0.3
    true_sigma_obs = 0.5

    x = np.zeros(n_obs)
    y = np.zeros(n_obs)

    x[0] = rng.standard_normal() * true_sigma_state
    y[0] = x[0] + rng.standard_normal() * true_sigma_obs

    for t in range(1, n_obs):
        x[t] = true_phi * x[t - 1] + rng.standard_normal() * true_sigma_state
        y[t] = x[t] + rng.standard_normal() * true_sigma_obs

    true_params = {
        "phi": true_phi,
        "sigma_state": true_sigma_state,
        "sigma_obs": true_sigma_obs,
    }
    return y[:, np.newaxis], true_params


class TestSMCSquaredSVModel:
    """Tests for SMC^2 with state-space models."""

    def test_smc2_sv_model(
        self,
        local_level_data: tuple[NDArray[np.floating[Any]], dict[str, float]],
    ) -> None:
        """SMC^2 posterior mean for phi should be close to true value."""
        endog, true_params = local_level_data

        model = LocalLevelModel()
        prior = LocalLevelPrior()

        smc2 = SMCSquared(
            model_class=model,
            n_theta=100,
            n_x=30,
            prior=prior,
            n_mcmc_moves=2,
            seed=42,
        )
        results = smc2.run(endog=endog)

        assert isinstance(results, SMCResults)
        # Posterior mean for phi should be in a reasonable range
        mean = results.posterior_mean()
        # phi is the first parameter
        assert 0.0 < mean[0] < 1.0, (
            f"phi mean={mean[0]}, expected near {true_params['phi']}"
        )

    def test_smc2_log_evidence(
        self,
        local_level_data: tuple[NDArray[np.floating[Any]], dict[str, float]],
    ) -> None:
        """SMC^2 should produce a finite log-evidence estimate."""
        endog, _ = local_level_data

        model = LocalLevelModel()
        prior = LocalLevelPrior()

        smc2 = SMCSquared(
            model_class=model,
            n_theta=50,
            n_x=20,
            prior=prior,
            n_mcmc_moves=1,
            seed=42,
        )
        results = smc2.run(endog=endog)

        assert np.isfinite(results.log_evidence)

    def test_smc2_parameter_uncertainty(
        self,
        local_level_data: tuple[NDArray[np.floating[Any]], dict[str, float]],
    ) -> None:
        """SMC^2 should produce reasonable parameter uncertainty."""
        endog, _ = local_level_data

        model = LocalLevelModel()
        prior = LocalLevelPrior()

        smc2 = SMCSquared(
            model_class=model,
            n_theta=100,
            n_x=30,
            prior=prior,
            n_mcmc_moves=2,
            seed=42,
        )
        results = smc2.run(endog=endog)

        # Standard deviations should be positive and finite
        std = results.posterior_std()
        assert np.all(std > 0)
        assert np.all(np.isfinite(std))

        # 95% CI should be non-degenerate
        ci = results.credible_interval(level=0.95)
        for j in range(results.k_params):
            assert ci[j, 1] > ci[j, 0], f"Degenerate CI for param {j}"


class TestInternalPF:
    """Tests for the InternalPF helper class."""

    def test_internal_pf_extend(self) -> None:
        """InternalPF should extend by one observation."""
        rng = np.random.default_rng(42)
        model = LocalLevelModel()
        theta = np.array([0.9, 0.3, 0.5])

        pf = InternalPF(n_x=100, model=model, theta=theta, rng=rng)
        pf.initialize()

        y_t = np.array([1.0])
        log_lik = pf.extend(y_t)

        assert np.isfinite(log_lik)
        assert pf.t == 1

    def test_internal_pf_run_full(self) -> None:
        """run_full should produce finite log-evidence."""
        rng = np.random.default_rng(42)
        model = LocalLevelModel()
        theta = np.array([0.9, 0.3, 0.5])

        endog = rng.standard_normal((20, 1))

        pf = InternalPF(n_x=100, model=model, theta=theta, rng=rng)
        log_lik = pf.run_full(endog)

        assert np.isfinite(log_lik)
        assert pf.t == 20

    def test_internal_pf_copy(self) -> None:
        """Copied PF should be independent from original."""
        rng = np.random.default_rng(42)
        model = LocalLevelModel()
        theta = np.array([0.9, 0.3, 0.5])

        pf = InternalPF(n_x=50, model=model, theta=theta, rng=rng)
        pf.initialize()
        pf.extend(np.array([1.0]))

        pf_copy = pf.copy()

        # Modify original
        pf.extend(np.array([2.0]))

        # Copy should still be at t=1
        assert pf.t == 2
        assert pf_copy.t == 1
