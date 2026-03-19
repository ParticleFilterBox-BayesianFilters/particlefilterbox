"""Tests for MCMC kernels used in SMC rejuvenation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from particlefilterbox.smc.mcmc_moves import (
    AdaptiveMH,
    RandomWalkMH,
    run_mcmc_chain,
)
from particlefilterbox.smc.results import SMCResults


class TestRandomWalkMH:
    """Tests for RandomWalkMH kernel."""

    def test_step_returns_result(
        self,
        rng: np.random.Generator,
        gaussian_log_target: Any,
    ) -> None:
        """Step returns MCMCStepResult with correct fields."""
        kernel = RandomWalkMH(proposal_cov=np.eye(2) * 0.5)
        result = kernel.step(np.zeros(2), gaussian_log_target, rng)

        assert result.theta.shape == (2,)
        assert isinstance(result.accepted, bool)
        assert isinstance(result.log_target, float)

    def test_chain_converges_to_target_mean(
        self,
        rng: np.random.Generator,
        gaussian_log_target: Any,
    ) -> None:
        """Long RW-MH chain mean should be close to target mean [1, 2]."""
        kernel = RandomWalkMH(proposal_cov=np.eye(2) * 0.5)
        theta = np.zeros(2)
        samples = []
        log_t: float | None = None

        for _ in range(5000):
            result = kernel.step(
                theta, gaussian_log_target, rng, log_target_current=log_t
            )
            theta = result.theta
            log_t = result.log_target
            samples.append(theta.copy())

        # Discard burn-in
        samples_arr = np.array(samples[1000:])
        mean = samples_arr.mean(axis=0)

        np.testing.assert_allclose(mean, [1.0, 2.0], atol=0.2)

    def test_proposal_cov_diagonal(self, rng: np.random.Generator) -> None:
        """1-D proposal covariance should work."""
        kernel = RandomWalkMH(proposal_cov=np.array([0.5, 0.3]))
        assert kernel.proposal_cov.shape == (2, 2)

    def test_acceptance_nonzero(
        self,
        rng: np.random.Generator,
        gaussian_log_target: Any,
    ) -> None:
        """Acceptance rate should be nonzero for reasonable proposal."""
        kernel = RandomWalkMH(proposal_cov=np.eye(2) * 0.5)
        n_accepted = 0
        theta = np.zeros(2)

        for _ in range(100):
            result = kernel.step(theta, gaussian_log_target, rng)
            theta = result.theta
            if result.accepted:
                n_accepted += 1

        assert n_accepted > 0


class TestAdaptiveMH:
    """Tests for AdaptiveMH kernel."""

    def test_adaptation_changes_scale(
        self,
        rng: np.random.Generator,
        gaussian_log_target: Any,
    ) -> None:
        """Adaptive kernel should change its scale during sampling."""
        kernel = AdaptiveMH(dim=2, target_acceptance=0.234)
        initial_scale = kernel._scale
        theta = np.zeros(2)

        for _ in range(200):
            result = kernel.step(theta, gaussian_log_target, rng)
            theta = result.theta

        assert kernel._scale != initial_scale

    def test_acceptance_rate_reasonable(
        self,
        rng: np.random.Generator,
        gaussian_log_target: Any,
    ) -> None:
        """After adaptation, acceptance rate should approach target."""
        kernel = AdaptiveMH(dim=2, target_acceptance=0.234)
        theta = np.zeros(2)

        for _ in range(2000):
            result = kernel.step(theta, gaussian_log_target, rng)
            theta = result.theta

        # Acceptance rate should be in a reasonable range
        assert 0.05 < kernel.acceptance_rate < 0.8


class TestSMCResultsSummary:
    """Tests for SMCResults summary methods."""

    def test_summary_format(self) -> None:
        """Summary should return a non-empty string."""
        rng = np.random.default_rng(42)
        particles = rng.standard_normal((500, 2))
        weights = np.ones(500) / 500

        results = SMCResults(
            particles=particles,
            weights=weights,
            log_evidence=-10.0,
            param_names=["mu", "sigma"],
            n_steps=10,
        )

        summary = results.summary()
        assert "SMC Results Summary" in summary
        assert "mu" in summary
        assert "sigma" in summary
        assert "Log-evidence" in summary

    def test_posterior_mean(self) -> None:
        """Posterior mean of uniform-weighted particles = sample mean."""
        rng = np.random.default_rng(42)
        particles = rng.standard_normal((1000, 2)) + np.array([3.0, -1.0])
        weights = np.ones(1000) / 1000

        results = SMCResults(
            particles=particles,
            weights=weights,
            log_evidence=0.0,
            n_steps=1,
        )

        mean = results.posterior_mean()
        np.testing.assert_allclose(mean, [3.0, -1.0], atol=0.15)

    def test_credible_interval_contains_mean(self) -> None:
        """95% CI should contain the posterior mean."""
        rng = np.random.default_rng(42)
        particles = rng.standard_normal((1000, 2))
        weights = np.ones(1000) / 1000

        results = SMCResults(
            particles=particles,
            weights=weights,
            log_evidence=0.0,
            n_steps=1,
        )

        ci = results.credible_interval(level=0.95)
        mean = results.posterior_mean()

        for j in range(2):
            assert ci[j, 0] <= mean[j] <= ci[j, 1]

    def test_to_dataframe(self) -> None:
        """to_dataframe should return a DataFrame with correct columns."""
        particles = np.array([[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([0.5, 0.5])

        results = SMCResults(
            particles=particles,
            weights=weights,
            log_evidence=0.0,
            param_names=["a", "b"],
            n_steps=1,
        )

        df = results.to_dataframe()
        assert list(df.columns) == ["a", "b", "weight"]
        assert len(df) == 2
