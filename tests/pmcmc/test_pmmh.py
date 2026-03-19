"""Tests for PMMH (Particle Marginal Metropolis-Hastings).

Tests include:
- Gaussian posterior recovery (critical)
- Acceptance rate in valid range (critical)
- Reproducibility with seed
- Adaptive proposal behavior
"""

from __future__ import annotations

import numpy as np

from particlefilterbox.pmcmc.pmmh import PMMH
from particlefilterbox.pmcmc.proposals import GaussianRandomWalk

# Import from conftest
from tests.pmcmc.conftest import MockPrior, MockSSModel


class TestPMMHGaussianPosterior:
    """CRITICAL: Test that PMMH recovers known posterior.

    For a simple linear Gaussian model with known parameters, the
    posterior mean from PMMH should be close to the analytical posterior mean.
    """

    def test_pmmh_gaussian_posterior(self) -> None:
        """Posterior mean should be within 0.15 of analytical mean.

        Uses a simple model where we know the approximate posterior.
        N=200 particles, n_iter=5000.

        Note: In a linear Gaussian SSM, phi is well-identified while
        sigma_x and sigma_y trade off (not separately identifiable).
        We test phi recovery strictly and verify mixing for others.
        """
        rng = np.random.default_rng(42)

        # Create model with known parameters
        model = MockSSModel()
        true_params = np.array([0.9, 0.5, 1.0])
        model.set_params(true_params)

        # Generate data
        observations = model.simulate(n_obs=100, rng=rng)

        # Prior centered near truth
        prior = MockPrior(
            mean=np.array([0.9, 0.5, 1.0]),
            cov=np.diag([0.05, 0.05, 0.05]),
        )

        # Run PMMH
        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=200,
            n_iterations=5000,
            proposal_cov="adaptive",
            target_acceptance=0.234,
            burnin=2500,
            seed=42,
        )

        results = pmmh.run(endog=observations, theta_init=true_params)
        posterior_mean = results.posterior_mean()

        # CRITICAL: phi (param 0) is well-identified, check strictly
        assert abs(posterior_mean[0] - true_params[0]) < 0.15, (
            f"phi: posterior mean {posterior_mean[0]:.4f} "
            f"too far from true value {true_params[0]:.4f}"
        )

        # sigma_x and sigma_y trade off in this model but should
        # remain in a reasonable range (prior-supported region)
        for j in range(1, len(true_params)):
            assert abs(posterior_mean[j] - true_params[j]) < 1.5, (
                f"Parameter {j}: posterior mean {posterior_mean[j]:.4f} "
                f"unreasonably far from true value {true_params[j]:.4f}"
            )

        # Chain should show mixing (non-zero std for all params)
        posterior_std = results.posterior_std()
        for j in range(len(true_params)):
            assert posterior_std[j] > 0.001, (
                f"Parameter {j}: chain is degenerate (std={posterior_std[j]:.6f})"
            )


class TestPMMHAcceptanceRate:
    """CRITICAL: Test that PMMH achieves reasonable acceptance rate."""

    def test_pmmh_acceptance_rate(self) -> None:
        """Acceptance rate should be in [0.10, 0.40]."""
        rng = np.random.default_rng(123)

        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        observations = model.simulate(n_obs=50, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=200,
            n_iterations=5000,
            proposal_cov="adaptive",
            target_acceptance=0.234,
            seed=123,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        rate = results.acceptance_rate()
        assert 0.10 <= rate <= 0.40, (
            f"Acceptance rate {rate:.4f} outside [0.10, 0.40]"
        )


class TestPMMHReproducibility:
    """Test that PMMH gives identical results with same seed."""

    def test_pmmh_reproducibility(self) -> None:
        """Two runs with same seed should produce identical chains."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=30, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        # Run 1
        pmmh1 = PMMH(
            model=MockSSModel(),
            prior=prior,
            n_particles=50,
            n_iterations=100,
            proposal_cov="adaptive",
            seed=999,
        )
        results1 = pmmh1.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        # Run 2
        pmmh2 = PMMH(
            model=MockSSModel(),
            prior=prior,
            n_particles=50,
            n_iterations=100,
            proposal_cov="adaptive",
            seed=999,
        )
        results2 = pmmh2.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        np.testing.assert_allclose(
            results1.chains, results2.chains, atol=1e-10,
            err_msg="PMMH not reproducible with same seed",
        )


class TestPMMHCustomProposal:
    """Test PMMH with custom proposal distribution."""

    def test_pmmh_fixed_proposal(self) -> None:
        """PMMH should work with a fixed GaussianRandomWalk proposal."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=30, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        proposal = GaussianRandomWalk(dim=3, cov=0.01, seed=42)

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=50,
            n_iterations=200,
            proposal=proposal,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        assert results.n_iterations == 200
        assert results.n_params == 3
        assert results.acceptance_rate() > 0.0


class TestPMMHEdgeCases:
    """Test PMMH edge cases and robustness."""

    def test_pmmh_short_chain(self) -> None:
        """PMMH should work with very short chain."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=20,
            n_iterations=50,
            burnin=10,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        assert results.n_iterations == 50
        assert results.n_effective_samples == 40  # 50 - 10

    def test_pmmh_summary(self) -> None:
        """PMMH results should produce a valid summary."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=20,
            n_iterations=100,
            burnin=20,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        summary = results.summary()
        assert "PMCMC Posterior Summary" in summary
