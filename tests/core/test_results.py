"""Tests for ParticleFilterResults."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from particlefilterbox.core.results import ParticleFilterResults


def _make_dummy_results(T: int = 50, k: int = 1, N: int = 100) -> ParticleFilterResults:
    """Create dummy results for testing."""
    rng = np.random.default_rng(42)
    return ParticleFilterResults(
        filtered_mean=rng.standard_normal((T, k)),
        filtered_cov=np.tile(np.eye(k), (T, 1, 1)),
        filtered_quantiles={
            0.025: rng.standard_normal((T, k)),
            0.500: rng.standard_normal((T, k)),
            0.975: rng.standard_normal((T, k)),
        },
        log_likelihood=-250.0,
        log_likelihood_increments=rng.standard_normal(T),
        ess_history=rng.uniform(50, 100, T),
        resampled=rng.random(T) > 0.5,
        n_particles=N,
        nobs=T,
        computation_time=1.5,
    )


class TestParticleFilterResults:
    def test_summary(self) -> None:
        results = _make_dummy_results()
        s = results.summary()
        assert "Particle Filter Results" in s
        assert "Log-likelihood" in s
        assert "50" in s  # nobs

    def test_to_dataframe(self) -> None:
        results = _make_dummy_results(T=50, k=1)
        df = results.to_dataframe()
        assert len(df) == 50
        assert "state_mean" in df.columns
        assert "ess" in df.columns
        assert "resampled" in df.columns

    def test_to_dataframe_multistate(self) -> None:
        results = _make_dummy_results(T=30, k=3)
        df = results.to_dataframe()
        assert "state_0_mean" in df.columns
        assert "state_2_mean" in df.columns

    def test_save_load_roundtrip(self) -> None:
        results = _make_dummy_results(T=20, k=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.npz"
            results.save(path)
            loaded = ParticleFilterResults.load(path)
            assert loaded.nobs == results.nobs
            assert loaded.n_particles == results.n_particles
            np.testing.assert_allclose(loaded.log_likelihood, results.log_likelihood)
            np.testing.assert_allclose(loaded.filtered_mean, results.filtered_mean)
            np.testing.assert_allclose(loaded.ess_history, results.ess_history)
