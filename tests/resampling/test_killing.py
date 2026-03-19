"""Tests for killing resampling."""

from __future__ import annotations

import numpy as np

from particlefilterbox.resampling.killing import killing_resample


class TestKillingResample:
    def test_correct_number_of_indices(self) -> None:
        w = np.ones(100) / 100
        indices = killing_resample(w, rng=np.random.default_rng(42))
        assert len(indices) == 100

    def test_indices_in_range(self) -> None:
        w = np.ones(50) / 50
        indices = killing_resample(w, rng=np.random.default_rng(42))
        assert np.all(indices >= 0)
        assert np.all(indices < 50)

    def test_single_particle_dominance(self) -> None:
        """One weight = 1 -> that particle always survives."""
        w = np.zeros(100)
        w[0] = 1.0
        indices = killing_resample(w, rng=np.random.default_rng(42))
        assert 0 in indices

    def test_high_weight_survives(self) -> None:
        """Particles with w_i >= 1/N always survive."""
        n = 10
        w = np.ones(n) / n  # all w_i = 1/N = 0.1, p_survive = 1.0
        indices = killing_resample(w, rng=np.random.default_rng(42))
        # All should appear
        assert len(np.unique(indices)) == n

    def test_reproducibility(self) -> None:
        w = np.ones(100) / 100
        idx1 = killing_resample(w, rng=np.random.default_rng(42))
        idx2 = killing_resample(w, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(idx1, idx2)
