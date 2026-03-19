"""Tests for stratified resampling."""

from __future__ import annotations

import numpy as np

from particlefilterbox.resampling.stratified import stratified_resample


class TestStratifiedResample:
    def test_correct_number_of_indices(self) -> None:
        w = np.ones(100) / 100
        indices = stratified_resample(w, rng=np.random.default_rng(42))
        assert len(indices) == 100

    def test_indices_in_range(self) -> None:
        w = np.ones(50) / 50
        indices = stratified_resample(w, rng=np.random.default_rng(42))
        assert np.all(indices >= 0)
        assert np.all(indices < 50)

    def test_single_particle_dominance(self) -> None:
        w = np.zeros(100)
        w[0] = 1.0
        indices = stratified_resample(w, rng=np.random.default_rng(42))
        assert np.all(indices == 0)

    def test_preserves_weighted_mean(self) -> None:
        rng = np.random.default_rng(42)
        n = 10000
        x = rng.standard_normal(n)
        w = rng.dirichlet(np.ones(n))
        weighted_mean = np.sum(w * x)
        indices = stratified_resample(w, rng=rng)
        resampled_mean = np.mean(x[indices])
        assert abs(resampled_mean - weighted_mean) < 0.05

    def test_reproducibility(self) -> None:
        w = np.ones(100) / 100
        idx1 = stratified_resample(w, rng=np.random.default_rng(42))
        idx2 = stratified_resample(w, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(idx1, idx2)
