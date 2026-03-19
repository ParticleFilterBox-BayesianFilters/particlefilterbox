"""Tests for SMCSampler and Tempering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from particlefilterbox.smc.sampler import SMCSampler
from particlefilterbox.smc.tempering import Tempering
from particlefilterbox.smc.results import SMCResults


# =============================================================
# Analytical Gaussian test case
# =============================================================
# Prior: N(0, sigma_prior^2 * I)
# Likelihood: y ~ N(theta, sigma_lik^2 * I)
# Posterior: N(posterior_mean, posterior_var * I)
# where posterior_var = 1/(1/sigma_prior^2 + n/sigma_lik^2)
#       posterior_mean = posterior_var * (sum(y) / sigma_lik^2)

SIGMA_PRIOR = 10.0
SIGMA_LIK = 1.0
TRUE_THETA = np.array([3.0, -1.0])
DIM = 2


@pytest.fixture
def gaussian_data(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """Generate Gaussian observations from true_theta."""
    n_obs = 50
    return TRUE_THETA[np.newaxis, :] + rng.standard_normal((n_obs, DIM)) * SIGMA_LIK


@pytest.fixture
def analytical_posterior_mean(
    gaussian_data: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Analytical posterior mean for Gaussian conjugate model."""
    n_obs = gaussian_data.shape[0]
    posterior_var = 1.0 / (1.0 / SIGMA_PRIOR**2 + n_obs / SIGMA_LIK**2)
    posterior_mean = posterior_var * (gaussian_data.sum(axis=0) / SIGMA_LIK**2)
    return posterior_mean


@pytest.fixture
def analytical_log_evidence(
    gaussian_data: NDArray[np.floating[Any]],
) -> float:
    """Analytical log marginal likelihood for Gaussian model.

    log p(y) = -n*d/2*log(2*pi) - n*d/2*log(sigma_lik^2)
               + d/2*log(posterior_var) - d/2*log(sigma_prior^2)
               - 0.5*(sum(y^2)/sigma_lik^2 - posterior_mean^2/posterior_var)
    """
    n_obs = gaussian_data.shape[0]
    posterior_var = 1.0 / (1.0 / SIGMA_PRIOR**2 + n_obs / SIGMA_LIK**2)
    posterior_mean = posterior_var * (gaussian_data.sum(axis=0) / SIGMA_LIK**2)

    log_z = 0.0
    for d in range(DIM):
        log_z += (
            -0.5 * n_obs * np.log(2 * np.pi * SIGMA_LIK**2)
            + 0.5 * np.log(posterior_var)
            - 0.5 * np.log(SIGMA_PRIOR**2)
            - 0.5 * (
                np.sum(gaussian_data[:, d] ** 2) / SIGMA_LIK**2
                - posterior_mean[d] ** 2 / posterior_var
            )
        )
    return float(log_z)


# =============================================================
# SMCSampler tests
# =============================================================


class TestSMCSamplerGaussianPosterior:
    """Test that SMCSampler converges to known Gaussian posterior."""

    def test_smc_gaussian_posterior(
        self,
        rng: np.random.Generator,
        gaussian_data: NDArray[np.floating[Any]],
        analytical_posterior_mean: NDArray[np.floating[Any]],
    ) -> None:
        """SMC posterior mean should be close to analytical mean (|diff|<0.3)."""
        data = gaussian_data

        def log_prior(theta: NDArray[np.floating[Any]]) -> float:
            return float(-0.5 * np.sum(theta**2 / SIGMA_PRIOR**2))

        def log_lik(theta: NDArray[np.floating[Any]]) -> float:
            n_obs = data.shape[0]
            diff = data - theta[np.newaxis, :]
            return float(
                -0.5 * n_obs * DIM * np.log(2 * np.pi * SIGMA_LIK**2)
                - 0.5 * np.sum(diff**2 / SIGMA_LIK**2)
            )

        def log_target(theta: NDArray[np.floating[Any]]) -> float:
            return log_prior(theta) + log_lik(theta)

        def sample_prior(gen: np.random.Generator) -> NDArray[np.floating[Any]]:
            return gen.standard_normal(DIM) * SIGMA_PRIOR

        sampler = SMCSampler(
            target_logpdf=log_target,
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=2000,
            n_mcmc_moves=5,
            ess_target_ratio=0.5,
            seed=42,
        )
        results = sampler.run()

        mean = results.posterior_mean()
        np.testing.assert_allclose(
            mean, analytical_posterior_mean, atol=0.3
        )

    def test_log_evidence_gaussian(
        self,
        rng: np.random.Generator,
        gaussian_data: NDArray[np.floating[Any]],
        analytical_log_evidence: float,
    ) -> None:
        """Log-evidence should be close to analytical value (|diff|<2.0)."""
        data = gaussian_data

        def log_prior(theta: NDArray[np.floating[Any]]) -> float:
            return float(-0.5 * np.sum(theta**2 / SIGMA_PRIOR**2))

        def log_lik(theta: NDArray[np.floating[Any]]) -> float:
            n_obs = data.shape[0]
            diff = data - theta[np.newaxis, :]
            return float(
                -0.5 * n_obs * DIM * np.log(2 * np.pi * SIGMA_LIK**2)
                - 0.5 * np.sum(diff**2 / SIGMA_LIK**2)
            )

        def log_target(theta: NDArray[np.floating[Any]]) -> float:
            return log_prior(theta) + log_lik(theta)

        def sample_prior(gen: np.random.Generator) -> NDArray[np.floating[Any]]:
            return gen.standard_normal(DIM) * SIGMA_PRIOR

        sampler = SMCSampler(
            target_logpdf=log_target,
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=2000,
            n_mcmc_moves=5,
            seed=42,
        )
        results = sampler.run()

        assert abs(results.log_evidence - analytical_log_evidence) < 2.0


class TestSMCSamplerAdaptiveSchedule:
    """Tests for adaptive beta schedule."""

    def test_adaptive_schedule(self, rng: np.random.Generator) -> None:
        """Schedule should be monotonically increasing, ending at 1.0."""

        def log_target(theta: NDArray[np.floating[Any]]) -> float:
            return float(-0.5 * np.sum(theta**2))

        def log_prior(theta: NDArray[np.floating[Any]]) -> float:
            return float(-0.5 * np.sum(theta**2 / 100))

        def sample_prior(gen: np.random.Generator) -> NDArray[np.floating[Any]]:
            return gen.standard_normal(2) * 10.0

        sampler = SMCSampler(
            target_logpdf=log_target,
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=500,
            n_mcmc_moves=3,
            seed=42,
        )
        results = sampler.run()

        schedule = results.schedule
        assert schedule[0] == 0.0
        assert abs(schedule[-1] - 1.0) < 1e-4
        # Monotonically increasing
        for i in range(1, len(schedule)):
            assert schedule[i] >= schedule[i - 1]

    def test_acceptance_rate(self, rng: np.random.Generator) -> None:
        """Acceptance rates should be in a reasonable range."""

        def log_target(theta: NDArray[np.floating[Any]]) -> float:
            return float(-0.5 * np.sum(theta**2))

        def log_prior(theta: NDArray[np.floating[Any]]) -> float:
            return float(-0.5 * np.sum(theta**2 / 100))

        def sample_prior(gen: np.random.Generator) -> NDArray[np.floating[Any]]:
            return gen.standard_normal(2) * 10.0

        sampler = SMCSampler(
            target_logpdf=log_target,
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=500,
            n_mcmc_moves=5,
            seed=42,
        )
        results = sampler.run()

        if results.acceptance_rates:
            mean_acc = np.mean(results.acceptance_rates)
            # Should be between 5% and 95%
            assert 0.05 < mean_acc < 0.95


# =============================================================
# Tempering tests
# =============================================================


@dataclass
class SimpleGaussianModel:
    """Simple Gaussian model for testing Tempering."""

    sigma: float = SIGMA_LIK

    def log_likelihood(
        self,
        theta: NDArray[np.floating[Any]],
        endog: NDArray[np.floating[Any]],
    ) -> float:
        n_obs = endog.shape[0]
        d = theta.shape[0]
        diff = endog - theta[np.newaxis, :]
        return float(
            -0.5 * n_obs * d * np.log(2 * np.pi * self.sigma**2)
            - 0.5 * np.sum(diff**2 / self.sigma**2)
        )


@dataclass
class GaussianPrior:
    """Gaussian prior for testing."""

    sigma: float = SIGMA_PRIOR
    dim: int = DIM

    def logpdf(self, theta: NDArray[np.floating[Any]]) -> float:
        return float(-0.5 * np.sum(theta**2 / self.sigma**2))

    def sample(
        self, rng: np.random.Generator
    ) -> NDArray[np.floating[Any]]:
        return rng.standard_normal(self.dim) * self.sigma


class TestTempering:
    """Tests for the Tempering class."""

    def test_tempering_runs(
        self,
        gaussian_data: NDArray[np.floating[Any]],
    ) -> None:
        """Tempering should run and return SMCResults."""
        model = SimpleGaussianModel()
        prior = GaussianPrior()

        tempering = Tempering(
            model=model,
            prior=prior,
            n_particles=500,
            n_mcmc_moves=3,
            seed=42,
        )
        results = tempering.run(endog=gaussian_data)

        assert isinstance(results, SMCResults)
        assert results.n_particles == 500
        assert results.n_steps > 0

    def test_tempering_schedule(
        self,
        gaussian_data: NDArray[np.floating[Any]],
    ) -> None:
        """Tempering schedule should go from 0 to 1."""
        model = SimpleGaussianModel()
        prior = GaussianPrior()

        tempering = Tempering(
            model=model,
            prior=prior,
            n_particles=500,
            n_mcmc_moves=3,
            seed=42,
        )
        results = tempering.run(endog=gaussian_data)

        assert results.schedule[0] == 0.0
        assert abs(results.schedule[-1] - 1.0) < 1e-4

    def test_tempering_log_bf(
        self,
        gaussian_data: NDArray[np.floating[Any]],
        analytical_log_evidence: float,
    ) -> None:
        """Log Bayes factor should be close to analytical log-evidence."""
        model = SimpleGaussianModel()
        prior = GaussianPrior()

        tempering = Tempering(
            model=model,
            prior=prior,
            n_particles=2000,
            n_mcmc_moves=5,
            seed=42,
        )
        results = tempering.run(endog=gaussian_data)

        log_bf = tempering.log_bayes_factor()
        # Tolerance is generous for Monte Carlo
        assert abs(log_bf - analytical_log_evidence) < 3.0
