"""Tests for Waste-Free SMC."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.smc.results import SMCResults
from particlefilterbox.smc.sampler import SMCSampler
from particlefilterbox.smc.waste_free import WasteFreeSMC

# =============================================================
# Test setup: 2-D Gaussian target
# =============================================================

TARGET_MEAN = np.array([3.0, -2.0])
TARGET_COV = np.eye(2)
PRIOR_SIGMA = 10.0


def log_target(theta: NDArray[np.floating[Any]]) -> float:
    """Log-density of 2-D Gaussian target N(TARGET_MEAN, I)."""
    diff = theta - TARGET_MEAN
    return float(-0.5 * np.sum(diff**2))


def log_prior(theta: NDArray[np.floating[Any]]) -> float:
    """Log-density of prior N(0, PRIOR_SIGMA^2 * I)."""
    return float(-0.5 * np.sum(theta**2 / PRIOR_SIGMA**2))


def sample_prior(rng: np.random.Generator) -> NDArray[np.floating[Any]]:
    """Sample from prior."""
    return rng.standard_normal(2) * PRIOR_SIGMA


class TestWasteFreeSMC:
    """Tests for WasteFreeSMC."""

    def test_waste_free_runs(self) -> None:
        """WasteFreeSMC should run and return SMCResults."""
        wf = WasteFreeSMC(
            target_logpdf=log_target,
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=200,
            k_mcmc=10,
            seed=42,
        )
        results = wf.run()

        assert isinstance(results, SMCResults)
        assert results.n_particles == 200
        assert results.n_steps > 0

    def test_waste_free_posterior(self) -> None:
        """WasteFreeSMC posterior should be close to target mean."""
        wf = WasteFreeSMC(
            target_logpdf=log_target,
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=1000,
            k_mcmc=10,
            seed=42,
        )
        results = wf.run()

        mean = results.posterior_mean()
        np.testing.assert_allclose(mean, TARGET_MEAN, atol=0.5)

    def test_waste_free_matches_standard(self) -> None:
        """Waste-free SMC should produce similar posterior to standard SMC.

        Compare weighted posterior means and standard deviations.
        """
        # Standard SMC
        standard = SMCSampler(
            target_logpdf=log_target,
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=1000,
            n_mcmc_moves=10,
            seed=42,
        )
        results_std = standard.run()

        # Waste-free SMC
        wf = WasteFreeSMC(
            target_logpdf=log_target,
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=1000,
            k_mcmc=10,
            seed=43,
        )
        results_wf = wf.run()

        # Compare weighted posterior means
        mean_std = results_std.posterior_mean()
        mean_wf = results_wf.posterior_mean()
        np.testing.assert_allclose(mean_std, mean_wf, atol=0.5)

        # Compare weighted posterior stds
        std_std = results_std.posterior_std()
        std_wf = results_wf.posterior_std()
        np.testing.assert_allclose(std_std, std_wf, atol=0.5)

    def test_waste_free_efficiency(self) -> None:
        """Waste-free SMC should have non-zero acceptance rate."""
        wf = WasteFreeSMC(
            target_logpdf=log_target,
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=500,
            k_mcmc=5,
            seed=42,
        )
        results = wf.run()

        if results.acceptance_rates:
            mean_acc = np.mean(results.acceptance_rates)
            assert mean_acc > 0.0, "All proposals were rejected"

    def test_waste_free_n_particles_divisible(self) -> None:
        """n_particles should be adjusted to be divisible by k_mcmc."""
        wf = WasteFreeSMC(
            target_logpdf=log_target,
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=103,  # Not divisible by 10
            k_mcmc=10,
            seed=42,
        )
        # Should round down to 100
        assert wf.n_particles == 100
        assert wf.n_mothers == 10

    def test_waste_free_log_evidence(self) -> None:
        """Waste-free SMC should produce finite log-evidence."""
        wf = WasteFreeSMC(
            target_logpdf=log_target,
            prior_logpdf=log_prior,
            prior_sample=sample_prior,
            n_particles=500,
            k_mcmc=10,
            seed=42,
        )
        results = wf.run()

        assert np.isfinite(results.log_evidence)
