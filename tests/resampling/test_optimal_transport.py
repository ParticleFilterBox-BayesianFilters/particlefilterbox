"""Tests for optimal transport resampling."""

from __future__ import annotations

import numpy as np
import pytest

from particlefilterbox.resampling.optimal_transport import optimal_transport_resample


class TestOptimalTransportResample:
    def test_output_shape(self) -> None:
        """Returns particles (N, k), not indices."""
        rng = np.random.default_rng(42)
        n, k = 50, 2
        w = np.ones(n) / n
        particles = rng.standard_normal((n, k))
        new_p = optimal_transport_resample(w, particles, rng=rng)
        assert new_p.shape == (n, k)

    def test_preserves_weighted_mean(self) -> None:
        """Mean of new particles ~ weighted mean of old."""
        rng = np.random.default_rng(42)
        n, k = 100, 1
        particles = rng.standard_normal((n, k))
        w = rng.dirichlet(np.ones(n))
        weighted_mean = np.sum(w[:, None] * particles, axis=0)
        new_p = optimal_transport_resample(w, particles, rng=rng)
        new_mean = np.mean(new_p, axis=0)
        np.testing.assert_allclose(new_mean, weighted_mean, atol=0.1)

    def test_uniform_weights_no_change(self) -> None:
        """Uniform weights -> particles should not change much."""
        rng = np.random.default_rng(42)
        n, k = 50, 2
        w = np.ones(n) / n
        particles = rng.standard_normal((n, k))
        new_p = optimal_transport_resample(w, particles, rng=rng)
        # With uniform weights, new particles should be very close to originals
        # (may be reordered, so compare sorted)
        old_sorted = np.sort(particles, axis=0)
        new_sorted = np.sort(new_p, axis=0)
        np.testing.assert_allclose(new_sorted, old_sorted, atol=0.5)

    def test_exact_method(self) -> None:
        """Exact method should also produce valid output."""
        rng = np.random.default_rng(42)
        n, k = 20, 1  # small N for exact method
        w = rng.dirichlet(np.ones(n))
        particles = rng.standard_normal((n, k))
        new_p = optimal_transport_resample(w, particles, rng=rng, method="exact")
        assert new_p.shape == (n, k)

    def test_invalid_method(self) -> None:
        """Invalid method raises ValueError."""
        rng = np.random.default_rng(42)
        w = np.ones(10) / 10
        particles = rng.standard_normal((10, 1))
        with pytest.raises(ValueError, match="Unknown OT method"):
            optimal_transport_resample(w, particles, method="invalid")
