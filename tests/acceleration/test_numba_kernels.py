"""Tests for Numba JIT kernels.

Tests are skipped if numba is not installed.
"""

import time

import numpy as np
import pytest

from particlefilterbox.acceleration.numba_kernels import (
    _NUMBA_AVAILABLE,
    log_sum_exp_numba,
    log_sum_exp_python,
    normalize_log_weights_numba,
    normalize_log_weights_python,
    systematic_resample_numba,
    systematic_resample_python,
    weighted_mean_numba,
    weighted_mean_python,
    is_numba_available,
    enable_numba,
    disable_numba,
)

pytestmark = pytest.mark.skipif(
    not _NUMBA_AVAILABLE,
    reason="Numba not installed",
)


class TestSystematicResample:
    """Tests for systematic resampling kernel."""

    def test_numba_same_result(self) -> None:
        """Numba and Python versions should produce identical results."""
        rng = np.random.default_rng(42)
        n = 1000
        weights = rng.dirichlet(np.ones(n))
        u = rng.uniform(0, 1.0 / n)

        indices_py = systematic_resample_python(weights, u)
        indices_nb = systematic_resample_numba(weights, u)

        np.testing.assert_array_equal(indices_py, indices_nb)

    def test_uniform_weights(self) -> None:
        """Uniform weights should produce identity-like resampling."""
        n = 100
        weights = np.ones(n) / n
        u = 0.0
        indices_py = systematic_resample_python(weights, u)
        indices_nb = systematic_resample_numba(weights, u)
        # Both implementations should agree on the result
        np.testing.assert_array_equal(indices_py, indices_nb)


class TestLogSumExp:
    """Tests for log-sum-exp kernel."""

    def test_numba_same_result(self) -> None:
        """Numba and Python should give same result to high precision."""
        rng = np.random.default_rng(42)
        log_w = rng.normal(-5, 2, size=1000)

        result_py = log_sum_exp_python(log_w)
        result_nb = log_sum_exp_numba(log_w)

        assert abs(result_py - result_nb) < 1e-10, (
            f"Python={result_py}, Numba={result_nb}, diff={abs(result_py - result_nb)}"
        )

    def test_numerical_stability(self) -> None:
        """Should handle very large/small values without overflow."""
        log_w = np.array([-1000.0, -999.0, -998.0])
        result = log_sum_exp_numba(log_w)
        assert np.isfinite(result)

    def test_single_element(self) -> None:
        """Single element should return that element."""
        log_w = np.array([5.0])
        result = log_sum_exp_numba(log_w)
        assert abs(result - 5.0) < 1e-10


class TestWeightedMean:
    """Tests for weighted mean kernel."""

    def test_numba_same_result(self) -> None:
        """Numba and Python should give same result."""
        rng = np.random.default_rng(42)
        n, d = 1000, 3
        particles = rng.normal(0, 1, size=(n, d))
        weights = rng.dirichlet(np.ones(n))

        result_py = weighted_mean_python(particles, weights)
        result_nb = weighted_mean_numba(particles, weights)

        np.testing.assert_allclose(result_py, result_nb, atol=1e-10)

    def test_uniform_weights(self) -> None:
        """Uniform weights should give regular mean."""
        rng = np.random.default_rng(42)
        n, d = 500, 2
        particles = rng.normal(0, 1, size=(n, d))
        weights = np.ones(n) / n

        result = weighted_mean_numba(particles, weights)
        expected = np.mean(particles, axis=0)

        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestNormalizeLogWeights:
    """Tests for log-weight normalization kernel."""

    def test_numba_same_result(self) -> None:
        """Numba and Python should give same result."""
        rng = np.random.default_rng(42)
        log_w = rng.normal(-5, 2, size=500)

        result_py = normalize_log_weights_python(log_w)
        result_nb = normalize_log_weights_numba(log_w)

        np.testing.assert_allclose(result_py, result_nb, atol=1e-10)

    def test_sums_to_one(self) -> None:
        """Normalized weights should sum to 1."""
        rng = np.random.default_rng(42)
        log_w = rng.normal(-3, 1, size=1000)
        weights = normalize_log_weights_numba(log_w)
        assert abs(np.sum(weights) - 1.0) < 1e-10


class TestNumbaSpeedup:
    """Tests for Numba speedup."""

    def test_numba_speedup(self) -> None:
        """Numba should be at least 5x faster than Python for N=10000."""
        rng = np.random.default_rng(42)
        n = 10000
        weights = rng.dirichlet(np.ones(n))
        u = rng.uniform(0, 1.0 / n)

        # Warmup numba JIT
        _ = systematic_resample_numba(weights, u)

        # Time Python
        n_repeats = 100
        start = time.perf_counter()
        for _ in range(n_repeats):
            systematic_resample_python(weights, u)
        time_python = time.perf_counter() - start

        # Time Numba
        start = time.perf_counter()
        for _ in range(n_repeats):
            systematic_resample_numba(weights, u)
        time_numba = time.perf_counter() - start

        speedup = time_python / time_numba
        assert speedup >= 5.0, (
            f"Numba speedup = {speedup:.1f}x (need >= 5x). "
            f"Python: {time_python:.3f}s, Numba: {time_numba:.3f}s"
        )


class TestEnableDisable:
    """Tests for enable/disable numba."""

    def test_enable_disable_cycle(self) -> None:
        """Enable and disable should work without errors."""
        result = enable_numba()
        assert result is True
        disable_numba()

    def test_is_numba_available(self) -> None:
        """is_numba_available should return True when numba is installed."""
        assert is_numba_available() is True
