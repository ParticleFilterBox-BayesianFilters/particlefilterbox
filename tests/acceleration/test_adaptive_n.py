"""Tests for adaptive particle count."""

import numpy as np
import pytest

from particlefilterbox.acceleration.adaptive_n import AdaptiveN


class TestAdaptiveN:
    """Tests for AdaptiveN."""

    def test_adaptive_grows(self) -> None:
        """Low ESS should cause N to grow."""
        adaptive = AdaptiveN(n_min=100, n_max=10000, growth_factor=2.0)
        # ESS = 10 out of 500 -> ratio = 0.02 < 0.2 -> grow
        new_n = adaptive.adapt(current_n=500, ess=10.0, n_particles=500)
        assert new_n == 1000, f"Expected 1000, got {new_n}"

    def test_adaptive_shrinks(self) -> None:
        """High ESS should cause N to shrink."""
        adaptive = AdaptiveN(n_min=100, n_max=10000, shrink_factor=0.5)
        # ESS = 450 out of 500 -> ratio = 0.9 > 0.8 -> shrink
        new_n = adaptive.adapt(current_n=500, ess=450.0, n_particles=500)
        assert new_n == 250, f"Expected 250, got {new_n}"

    def test_adaptive_stays(self) -> None:
        """Moderate ESS should keep N unchanged."""
        adaptive = AdaptiveN(n_min=100, n_max=10000)
        # ESS = 250 out of 500 -> ratio = 0.5 in [0.2, 0.8] -> stay
        new_n = adaptive.adapt(current_n=500, ess=250.0, n_particles=500)
        assert new_n == 500, f"Expected 500, got {new_n}"

    def test_adaptive_bounds_max(self) -> None:
        """N should not exceed n_max."""
        adaptive = AdaptiveN(n_min=100, n_max=1000, growth_factor=2.0)
        new_n = adaptive.adapt(current_n=800, ess=10.0, n_particles=800)
        assert new_n == 1000, f"Expected 1000 (clamped), got {new_n}"

    def test_adaptive_bounds_min(self) -> None:
        """N should not go below n_min."""
        adaptive = AdaptiveN(n_min=100, n_max=10000, shrink_factor=0.5)
        new_n = adaptive.adapt(current_n=150, ess=140.0, n_particles=150)
        assert new_n == 100, f"Expected 100 (clamped), got {new_n}"

    def test_add_particles(self) -> None:
        """add_particles should increase particle count."""
        adaptive = AdaptiveN()
        rng = np.random.default_rng(42)
        particles = rng.normal(0, 1, size=(100, 2))
        weights = np.ones(100) / 100

        new_p, new_w = adaptive.add_particles(particles, weights, n_new=50, rng=rng)
        assert new_p.shape == (150, 2)
        assert len(new_w) == 150
        assert abs(np.sum(new_w) - 1.0) < 1e-10

    def test_add_particles_1d(self) -> None:
        """add_particles should work with 1D particles."""
        adaptive = AdaptiveN()
        rng = np.random.default_rng(42)
        particles = rng.normal(0, 1, size=100)
        weights = np.ones(100) / 100

        new_p, new_w = adaptive.add_particles(particles, weights, n_new=30, rng=rng)
        assert new_p.shape == (130,)
        assert len(new_w) == 130

    def test_prune_particles(self) -> None:
        """prune_particles should decrease particle count."""
        adaptive = AdaptiveN()
        rng = np.random.default_rng(42)
        particles = rng.normal(0, 1, size=(100, 2))
        weights = rng.dirichlet(np.ones(100))

        pruned_p, pruned_w = adaptive.prune_particles(particles, weights, n_keep=50)
        assert pruned_p.shape == (50, 2)
        assert len(pruned_w) == 50
        assert abs(np.sum(pruned_w) - 1.0) < 1e-10

    def test_prune_keeps_highest_weights(self) -> None:
        """Pruning should keep particles with highest weights."""
        adaptive = AdaptiveN()
        particles = np.arange(10, dtype=np.float64)
        weights = np.array(
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.1, 0.2, 0.3, 0.3]
        )

        pruned_p, pruned_w = adaptive.prune_particles(particles, weights, n_keep=3)
        # Should keep indices 7, 8, 9 (highest weights)
        assert set(pruned_p.astype(int).tolist()) == {7, 8, 9}

    def test_n_history(self) -> None:
        """History should track all adaptations."""
        adaptive = AdaptiveN(n_min=100, n_max=10000)
        adaptive.adapt(current_n=500, ess=10.0)
        adaptive.adapt(current_n=1000, ess=900.0)
        assert len(adaptive.n_history) == 2

    def test_reset(self) -> None:
        """Reset should clear history."""
        adaptive = AdaptiveN()
        adaptive.adapt(current_n=500, ess=10.0)
        adaptive.reset()
        assert len(adaptive.n_history) == 0

    def test_invalid_params(self) -> None:
        """Invalid parameters should raise."""
        with pytest.raises(ValueError, match="n_min"):
            AdaptiveN(n_min=0)
        with pytest.raises(ValueError, match="n_max"):
            AdaptiveN(n_min=100, n_max=50)
        with pytest.raises(ValueError, match="growth_factor"):
            AdaptiveN(growth_factor=0.5)
        with pytest.raises(ValueError, match="shrink_factor"):
            AdaptiveN(shrink_factor=1.5)
