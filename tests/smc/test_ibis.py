"""Tests for IBIS (Iterated Batch Importance Sampling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from particlefilterbox.smc.ibis import IBIS
from particlefilterbox.smc.results import SMCResults

# =============================================================
# Simple Gaussian model for IBIS testing
# =============================================================

SIGMA_PRIOR = 10.0
SIGMA_LIK = 1.0
TRUE_MU = np.array([2.0, -1.0])
DIM = 2


@dataclass
class GaussianModelIBIS:
    """Gaussian model where log-likelihood is computable directly.

    y_i ~ N(theta, sigma_lik^2 * I), i.i.d.
    """

    sigma: float = SIGMA_LIK

    def log_likelihood(
        self,
        theta: NDArray[np.floating[Any]],
        endog: NDArray[np.floating[Any]],
    ) -> float:
        """log p(y_{1:T} | theta) = sum_t log N(y_t; theta, sigma^2*I)."""
        diff = endog - theta[np.newaxis, :]
        return float(
            -0.5 * np.sum(diff**2 / self.sigma**2)
            - 0.5 * endog.shape[0] * endog.shape[1] * np.log(
                2 * np.pi * self.sigma**2
            )
        )


@dataclass
class GaussianPriorIBIS:
    """Gaussian prior: N(0, sigma_prior^2 * I)."""

    sigma: float = SIGMA_PRIOR
    dim: int = DIM

    def logpdf(self, theta: NDArray[np.floating[Any]]) -> float:
        return float(
            -0.5 * np.sum(theta**2 / self.sigma**2)
            - 0.5 * self.dim * np.log(2 * np.pi * self.sigma**2)
        )

    def sample(
        self, rng: np.random.Generator
    ) -> NDArray[np.floating[Any]]:
        return rng.standard_normal(self.dim) * self.sigma


@pytest.fixture
def ibis_data() -> NDArray[np.floating[Any]]:
    """Generate i.i.d. Gaussian data from TRUE_MU."""
    rng = np.random.default_rng(123)
    return TRUE_MU[np.newaxis, :] + rng.standard_normal((30, DIM)) * SIGMA_LIK


class TestIBISSimpleModel:
    """Tests for IBIS with a simple Gaussian model."""

    def test_ibis_simple_model(
        self, ibis_data: NDArray[np.floating[Any]]
    ) -> None:
        """IBIS should produce an SMCResults object."""
        model = GaussianModelIBIS()
        prior = GaussianPriorIBIS()

        ibis = IBIS(
            model=model,
            n_particles=500,
            prior=prior,
            n_mcmc_moves=3,
            batch_size=5,
            seed=42,
        )
        results = ibis.run(endog=ibis_data)

        assert isinstance(results, SMCResults)
        assert results.n_particles == 500
        assert results.n_steps > 0

    def test_ibis_marginal_likelihood(
        self, ibis_data: NDArray[np.floating[Any]]
    ) -> None:
        """IBIS log-evidence should be finite."""
        model = GaussianModelIBIS()
        prior = GaussianPriorIBIS()

        ibis = IBIS(
            model=model,
            n_particles=500,
            prior=prior,
            n_mcmc_moves=3,
            seed=42,
        )
        results = ibis.run(endog=ibis_data)

        assert np.isfinite(results.log_evidence)

    def test_ibis_rejuvenation_fires(
        self, ibis_data: NDArray[np.floating[Any]]
    ) -> None:
        """IBIS should trigger at least one rejuvenation."""
        model = GaussianModelIBIS()
        prior = GaussianPriorIBIS()

        ibis = IBIS(
            model=model,
            n_particles=200,
            prior=prior,
            n_mcmc_moves=3,
            batch_size=1,
            ess_threshold=0.8,  # Aggressive threshold to trigger rejuvenation
            seed=42,
        )
        results = ibis.run(endog=ibis_data)

        # Should have acceptance rates (meaning rejuvenation happened)
        assert len(results.acceptance_rates) > 0

    def test_ibis_posterior_reasonable(
        self, ibis_data: NDArray[np.floating[Any]]
    ) -> None:
        """IBIS posterior mean should be close to analytical posterior."""
        model = GaussianModelIBIS()
        prior = GaussianPriorIBIS()

        ibis = IBIS(
            model=model,
            n_particles=1000,
            prior=prior,
            n_mcmc_moves=5,
            batch_size=5,
            seed=42,
        )
        results = ibis.run(endog=ibis_data)

        mean = results.posterior_mean()
        # Should be roughly close to TRUE_MU (within MC tolerance)
        np.testing.assert_allclose(mean, TRUE_MU, atol=0.5)
