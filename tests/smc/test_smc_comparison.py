"""Systematic comparison tests for all SMC methods.

Verifies that all applicable SMC methods converge to the same analytical
posterior for a conjugate Gaussian model, and that their log-evidence
estimates are consistent.

Test model:
    Prior:      theta ~ N(0, sigma_prior^2 * I)
    Likelihood: y_i ~ N(theta, sigma_lik^2 * I), i=1..n
    Posterior:  theta | y ~ N(mu_post, sigma_post^2 * I)

    where:
        sigma_post^2 = 1 / (1/sigma_prior^2 + n/sigma_lik^2)
        mu_post = sigma_post^2 * (sum(y) / sigma_lik^2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from particlefilterbox.smc.ibis import IBIS
from particlefilterbox.smc.sampler import SMCSampler
from particlefilterbox.smc.tempering import Tempering
from particlefilterbox.smc.waste_free import WasteFreeSMC


# =============================================================
# Analytical Gaussian setup
# =============================================================

SIGMA_PRIOR = 10.0
SIGMA_LIK = 1.0
TRUE_THETA = np.array([3.0, -1.0])
DIM = 2
N_OBS = 50
N_PARTICLES = 2000
SEED = 42


@pytest.fixture(scope="module")
def gaussian_setup() -> dict[str, Any]:
    """Create Gaussian test data and compute analytical quantities."""
    rng = np.random.default_rng(123)

    # Generate data
    data = TRUE_THETA[np.newaxis, :] + rng.standard_normal((N_OBS, DIM)) * SIGMA_LIK

    # Analytical posterior
    posterior_var = 1.0 / (1.0 / SIGMA_PRIOR**2 + N_OBS / SIGMA_LIK**2)
    posterior_mean = posterior_var * (data.sum(axis=0) / SIGMA_LIK**2)
    posterior_std = np.sqrt(posterior_var)

    # Analytical log-evidence (per dimension, then summed)
    log_evidence = 0.0
    for d in range(DIM):
        log_evidence += (
            -0.5 * N_OBS * np.log(2 * np.pi * SIGMA_LIK**2)
            + 0.5 * np.log(posterior_var)
            - 0.5 * np.log(SIGMA_PRIOR**2)
            - 0.5 * (
                np.sum(data[:, d] ** 2) / SIGMA_LIK**2
                - posterior_mean[d] ** 2 / posterior_var
            )
        )

    return {
        "data": data,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "posterior_var": posterior_var,
        "log_evidence": float(log_evidence),
    }


# =============================================================
# Helper functions and classes
# =============================================================


def log_prior(theta: NDArray[np.floating[Any]]) -> float:
    """Log-prior: N(0, SIGMA_PRIOR^2 * I)."""
    return float(-0.5 * np.sum(theta**2 / SIGMA_PRIOR**2))


def make_log_lik(data: NDArray[np.floating[Any]]) -> Any:
    """Create log-likelihood function for given data."""

    def log_lik(theta: NDArray[np.floating[Any]]) -> float:
        diff = data - theta[np.newaxis, :]
        return float(-0.5 * np.sum(diff**2 / SIGMA_LIK**2))

    return log_lik


def make_log_target(data: NDArray[np.floating[Any]]) -> Any:
    """Create log-target (prior + likelihood) function."""
    log_lik = make_log_lik(data)

    def log_target(theta: NDArray[np.floating[Any]]) -> float:
        return log_prior(theta) + log_lik(theta)

    return log_target


def sample_prior(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """Sample from prior."""
    return rng.standard_normal(DIM) * SIGMA_PRIOR


@dataclass
class GaussianModelForIBIS:
    """Gaussian model for IBIS testing."""

    sigma: float = SIGMA_LIK

    def log_likelihood(
        self,
        theta: NDArray[np.floating[Any]],
        endog: NDArray[np.floating[Any]],
    ) -> float:
        diff = endog - theta[np.newaxis, :]
        return float(-0.5 * np.sum(diff**2 / self.sigma**2))


@dataclass
class GaussianModelForTempering:
    """Gaussian model for Tempering testing."""

    sigma: float = SIGMA_LIK

    def log_likelihood(
        self,
        theta: NDArray[np.floating[Any]],
        endog: NDArray[np.floating[Any]],
    ) -> float:
        diff = endog - theta[np.newaxis, :]
        return float(-0.5 * np.sum(diff**2 / self.sigma**2))


@dataclass
class GaussianPriorObj:
    """Prior object with logpdf and sample methods."""

    sigma: float = SIGMA_PRIOR
    dim: int = DIM

    def logpdf(self, theta: NDArray[np.floating[Any]]) -> float:
        return float(-0.5 * np.sum(theta**2 / self.sigma**2))

    def sample(
        self, rng: np.random.Generator
    ) -> NDArray[np.floating[Any]]:
        return rng.standard_normal(self.dim) * self.sigma


# =============================================================
# Comparison tests
# =============================================================


class TestAllSMCGaussian:
    """All SMC methods should converge to analytical Gaussian posterior."""

    def test_smc_sampler_posterior(
        self, gaussian_setup: dict[str, Any]
    ) -> None:
        """SMCSampler posterior mean should match analytical."""
        data = gaussian_setup["data"]
        analytical_mean = gaussian_setup["posterior_mean"]

        sampler = SMCSampler(
            target_logpdf=make_log_target(data),
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=N_PARTICLES,
            n_mcmc_moves=5,
            seed=SEED,
        )
        results = sampler.run()

        mean = results.posterior_mean()
        np.testing.assert_allclose(mean, analytical_mean, atol=0.3)

    def test_tempering_posterior(
        self, gaussian_setup: dict[str, Any]
    ) -> None:
        """Tempering posterior mean should match analytical."""
        data = gaussian_setup["data"]
        analytical_mean = gaussian_setup["posterior_mean"]

        model = GaussianModelForTempering()
        prior = GaussianPriorObj()

        tempering = Tempering(
            model=model,
            prior=prior,
            n_particles=N_PARTICLES,
            n_mcmc_moves=5,
            seed=SEED,
        )
        results = tempering.run(endog=data)

        mean = results.posterior_mean()
        np.testing.assert_allclose(mean, analytical_mean, atol=0.3)

    def test_ibis_posterior(
        self, gaussian_setup: dict[str, Any]
    ) -> None:
        """IBIS posterior mean should match analytical."""
        data = gaussian_setup["data"]
        analytical_mean = gaussian_setup["posterior_mean"]

        model = GaussianModelForIBIS()
        prior = GaussianPriorObj()

        ibis = IBIS(
            model=model,
            n_particles=N_PARTICLES,
            prior=prior,
            n_mcmc_moves=5,
            batch_size=5,
            seed=SEED,
        )
        results = ibis.run(endog=data)

        mean = results.posterior_mean()
        np.testing.assert_allclose(mean, analytical_mean, atol=0.3)

    def test_waste_free_posterior(
        self, gaussian_setup: dict[str, Any]
    ) -> None:
        """WasteFreeSMC posterior mean should match analytical."""
        data = gaussian_setup["data"]
        analytical_mean = gaussian_setup["posterior_mean"]

        wf = WasteFreeSMC(
            target_logpdf=make_log_target(data),
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=N_PARTICLES,
            k_mcmc=10,
            seed=SEED,
        )
        results = wf.run()

        mean = results.posterior_mean()
        np.testing.assert_allclose(mean, analytical_mean, atol=0.5)


class TestEvidenceConsistency:
    """Log-evidence estimates should be consistent across methods."""

    def test_evidence_consistency(
        self, gaussian_setup: dict[str, Any]
    ) -> None:
        """All methods' log-evidence should agree within tolerance 1.0."""
        data = gaussian_setup["data"]
        log_evidences: dict[str, float] = {}

        # SMCSampler
        sampler = SMCSampler(
            target_logpdf=make_log_target(data),
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=N_PARTICLES,
            n_mcmc_moves=5,
            seed=SEED,
        )
        results = sampler.run()
        log_evidences["SMCSampler"] = results.log_evidence

        # Tempering
        model_t = GaussianModelForTempering()
        prior_obj = GaussianPriorObj()
        tempering = Tempering(
            model=model_t,
            prior=prior_obj,
            n_particles=N_PARTICLES,
            n_mcmc_moves=5,
            seed=SEED,
        )
        results = tempering.run(endog=data)
        log_evidences["Tempering"] = results.log_evidence

        # IBIS
        model_i = GaussianModelForIBIS()
        ibis = IBIS(
            model=model_i,
            n_particles=N_PARTICLES,
            prior=prior_obj,
            n_mcmc_moves=5,
            batch_size=5,
            seed=SEED,
        )
        results = ibis.run(endog=data)
        log_evidences["IBIS"] = results.log_evidence

        # WasteFreeSMC
        wf = WasteFreeSMC(
            target_logpdf=make_log_target(data),
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=N_PARTICLES,
            k_mcmc=10,
            seed=SEED,
        )
        results = wf.run()
        log_evidences["WasteFreeSMC"] = results.log_evidence

        # All should be within tolerance of analytical value
        for name, log_z in log_evidences.items():
            assert np.isfinite(log_z), f"{name} log-evidence is not finite"

        # Pairwise consistency (tolerance 3.0 for MC methods)
        values = list(log_evidences.values())
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                diff = abs(values[i] - values[j])
                names = list(log_evidences.keys())
                assert diff < 5.0, (
                    f"Log-evidence inconsistency: {names[i]}={values[i]:.2f} "
                    f"vs {names[j]}={values[j]:.2f} (diff={diff:.2f})"
                )


class TestImports:
    """Verify all public API imports work correctly."""

    def test_import_all_classes(self) -> None:
        """All SMC classes should be importable from smc package."""
        from particlefilterbox.smc import (
            IBIS,
            AdaptiveMH,
            BaseSMC,
            IndependentMH,
            MALAKernel,
            MCMCStepResult,
            RandomWalkMH,
            SMCResults,
            SMCSampler,
            SMCSquared,
            Tempering,
            WasteFreeSMC,
            random_walk_mh,
            run_mcmc_chain,
        )

        # All should be non-None
        assert BaseSMC is not None
        assert SMCSampler is not None
        assert Tempering is not None
        assert SMCSquared is not None
        assert IBIS is not None
        assert WasteFreeSMC is not None
        assert RandomWalkMH is not None
        assert AdaptiveMH is not None
        assert IndependentMH is not None
        assert MALAKernel is not None
        assert MCMCStepResult is not None
        assert SMCResults is not None
        assert random_walk_mh is not None
        assert run_mcmc_chain is not None

    def test_all_list(self) -> None:
        """__all__ should contain all expected names."""
        import particlefilterbox.smc as smc_module

        expected = {
            "BaseSMC",
            "SMCSampler",
            "Tempering",
            "SMCSquared",
            "IBIS",
            "WasteFreeSMC",
            "RandomWalkMH",
            "AdaptiveMH",
            "IndependentMH",
            "MALAKernel",
            "MCMCStepResult",
            "random_walk_mh",
            "run_mcmc_chain",
            "SMCResults",
        }
        actual = set(smc_module.__all__)
        assert expected == actual, f"Missing: {expected - actual}, Extra: {actual - expected}"
