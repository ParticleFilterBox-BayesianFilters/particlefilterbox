"""Tests for systematic resampling."""

from __future__ import annotations

import numpy as np

from particlefilterbox.resampling.systematic import systematic_resample


class TestSystematicResample:
    def test_uniform_weights_identity(self) -> None:
        """Uniform weights -> each particle selected exactly once."""
        n = 100
        w = np.ones(n) / n
        indices = systematic_resample(w, rng=np.random.default_rng(42))
        counts = np.bincount(indices, minlength=n)
        np.testing.assert_array_equal(counts, np.ones(n, dtype=int))

    def test_single_particle_dominance(self) -> None:
        w = np.zeros(100)
        w[5] = 1.0
        indices = systematic_resample(w, rng=np.random.default_rng(42))
        assert np.all(indices == 5)

    def test_correct_number_of_indices(self) -> None:
        w = np.ones(200) / 200
        indices = systematic_resample(w, rng=np.random.default_rng(42))
        assert len(indices) == 200

    def test_indices_in_range(self) -> None:
        w = np.ones(50) / 50
        indices = systematic_resample(w, rng=np.random.default_rng(42))
        assert np.all(indices >= 0)
        assert np.all(indices < 50)

    def test_preserves_weighted_mean(self) -> None:
        rng = np.random.default_rng(42)
        n = 10000
        x = rng.standard_normal(n)
        w = rng.dirichlet(np.ones(n))
        weighted_mean = np.sum(w * x)
        indices = systematic_resample(w, rng=rng)
        resampled_mean = np.mean(x[indices])
        assert abs(resampled_mean - weighted_mean) < 0.05

    def test_reproducibility(self) -> None:
        w = np.ones(100) / 100
        idx1 = systematic_resample(w, rng=np.random.default_rng(42))
        idx2 = systematic_resample(w, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(idx1, idx2)
