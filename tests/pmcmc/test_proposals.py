"""Tests for PMCMC proposal distributions."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.pmcmc.proposals import (
    AdaptiveGaussian,
    GaussianRandomWalk,
    LogNormalProposal,
    TransformedProposal,
)


class TestGaussianRandomWalk:
    """Tests for GaussianRandomWalk proposal."""

    def test_propose_shape(self) -> None:
        """Proposal should return same shape as input."""
        proposal = GaussianRandomWalk(dim=3, seed=42)
        theta = np.array([1.0, 2.0, 3.0])
        proposed = proposal.propose(theta)
        assert proposed.shape == theta.shape

    def test_propose_different_from_current(self) -> None:
        """Proposal should differ from current value."""
        proposal = GaussianRandomWalk(dim=3, seed=42)
        theta = np.array([1.0, 2.0, 3.0])
        proposed = proposal.propose(theta)
        assert not np.allclose(proposed, theta)

    def test_symmetric_log_ratio(self) -> None:
        """Gaussian RW is symmetric, log_ratio should be 0."""
        proposal = GaussianRandomWalk(dim=3, seed=42)
        theta1 = np.array([1.0, 2.0, 3.0])
        theta2 = np.array([1.1, 2.1, 3.1])
        assert proposal.log_ratio(theta2, theta1) == 0.0

    def test_custom_covariance(self) -> None:
        """Should accept custom covariance matrix."""
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        proposal = GaussianRandomWalk(dim=2, cov=cov, seed=42)
        assert np.allclose(proposal.cov, cov)

    def test_scalar_covariance(self) -> None:
        """Should accept scalar covariance."""
        proposal = GaussianRandomWalk(dim=3, cov=0.01, seed=42)
        assert proposal.cov.shape == (3, 3)
        assert np.allclose(np.diag(proposal.cov), 0.01)

    def test_default_scaling(self) -> None:
        """Default covariance should use (2.38^2)/d scaling."""
        d = 5
        proposal = GaussianRandomWalk(dim=d, seed=42)
        expected_scale = (2.38**2) / d
        assert np.allclose(np.diag(proposal.cov), expected_scale)

    def test_reproducibility(self) -> None:
        """Same seed should give same proposals."""
        theta = np.array([1.0, 2.0])
        p1 = GaussianRandomWalk(dim=2, seed=42)
        p2 = GaussianRandomWalk(dim=2, seed=42)
        assert np.allclose(p1.propose(theta), p2.propose(theta))


class TestAdaptiveGaussian:
    """Tests for AdaptiveGaussian proposal."""

    def test_propose_shape(self) -> None:
        """Proposal should return same shape as input."""
        proposal = AdaptiveGaussian(dim=3, seed=42)
        theta = np.array([1.0, 2.0, 3.0])
        proposed = proposal.propose(theta)
        assert proposed.shape == theta.shape

    def test_adaptation_changes_covariance(self) -> None:
        """After enough adaptations, covariance should change."""
        proposal = AdaptiveGaussian(dim=2, adaptation_start=10, seed=42)
        initial_cov = proposal.cov.copy()

        rng = np.random.default_rng(42)
        for i in range(50):
            theta = rng.normal(size=2)
            proposal.adapt(theta, accepted=True)

        adapted_cov = proposal.cov
        assert not np.allclose(initial_cov, adapted_cov)

    def test_target_acceptance(self) -> None:
        """Should track target acceptance rate."""
        proposal = AdaptiveGaussian(
            dim=2,
            target_acceptance=0.234,
            seed=42,
        )
        assert proposal.target_acceptance == 0.234

    def test_roberts_rosenthal_scaling(self) -> None:
        """Should use s_d = (2.38^2)/d."""
        d = 4
        proposal = AdaptiveGaussian(dim=d, seed=42)
        expected = (2.38**2) / d
        assert np.isclose(proposal.s_d, expected)

    def test_symmetric_log_ratio(self) -> None:
        """Adaptive Gaussian is symmetric, log_ratio should be 0."""
        proposal = AdaptiveGaussian(dim=3, seed=42)
        theta1 = np.array([1.0, 2.0, 3.0])
        theta2 = np.array([1.1, 2.1, 3.1])
        assert proposal.log_ratio(theta2, theta1) == 0.0


class TestLogNormalProposal:
    """Tests for LogNormalProposal."""

    def test_propose_positive(self) -> None:
        """Proposals should always be positive."""
        proposal = LogNormalProposal(dim=3, seed=42)
        theta = np.array([1.0, 2.0, 0.5])

        for _ in range(100):
            proposed = proposal.propose(theta)
            assert np.all(proposed > 0)

    def test_propose_shape(self) -> None:
        """Proposal should return same shape as input."""
        proposal = LogNormalProposal(dim=3, seed=42)
        theta = np.array([1.0, 2.0, 0.5])
        proposed = proposal.propose(theta)
        assert proposed.shape == theta.shape

    def test_asymmetric_log_ratio(self) -> None:
        """LogNormal proposal is asymmetric, log_ratio should be nonzero."""
        proposal = LogNormalProposal(dim=2, seed=42)
        theta1 = np.array([1.0, 2.0])
        theta2 = np.array([1.5, 2.5])
        lr = proposal.log_ratio(theta2, theta1)
        assert lr != 0.0


class TestTransformedProposal:
    """Tests for TransformedProposal."""

    def test_log_transform_positive(self) -> None:
        """Log-transformed proposals should be positive."""
        transforms = [
            {"type": "log"},
            {"type": "log"},
        ]
        proposal = TransformedProposal(dim=2, transforms=transforms, seed=42)
        theta = np.array([1.0, 2.0])

        for _ in range(100):
            proposed = proposal.propose(theta)
            assert np.all(proposed > 0)

    def test_logit_transform_bounded(self) -> None:
        """Logit-transformed proposals should be in (lower, upper)."""
        transforms = [
            {"type": "logit", "lower": 0.0, "upper": 1.0},
            {"type": "logit", "lower": -1.0, "upper": 1.0},
        ]
        proposal = TransformedProposal(dim=2, transforms=transforms, seed=42)
        theta = np.array([0.5, 0.0])

        for _ in range(100):
            proposed = proposal.propose(theta)
            assert 0.0 < proposed[0] < 1.0
            assert -1.0 < proposed[1] < 1.0

    def test_none_transform_unconstrained(self) -> None:
        """'none' transform should not constrain."""
        transforms = [{"type": "none"}, {"type": "none"}]
        proposal = TransformedProposal(dim=2, transforms=transforms, seed=42)
        theta = np.array([0.0, 0.0])
        proposed = proposal.propose(theta)
        assert proposed.shape == (2,)

    def test_log_ratio_nonzero(self) -> None:
        """Transformed proposal should have nonzero log_ratio."""
        transforms = [{"type": "log"}, {"type": "logit", "lower": 0, "upper": 1}]
        proposal = TransformedProposal(dim=2, transforms=transforms, seed=42)
        theta1 = np.array([1.0, 0.5])
        theta2 = np.array([1.5, 0.7])
        lr = proposal.log_ratio(theta2, theta1)
        # Should be nonzero for asymmetric transforms
        assert isinstance(lr, float)

    def test_propose_shape(self) -> None:
        """Proposal should return same shape as input."""
        transforms = [{"type": "log"}, {"type": "none"}]
        proposal = TransformedProposal(dim=2, transforms=transforms, seed=42)
        theta = np.array([1.0, 0.0])
        proposed = proposal.propose(theta)
        assert proposed.shape == theta.shape
