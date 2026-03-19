"""Tests for adaptive resampling."""

from __future__ import annotations

import numpy as np

from particlefilterbox.resampling.adaptive import adaptive_resample, should_resample


class TestAdaptiveResample:
    def test_no_resample_when_ess_high(self) -> None:
        """Nearly uniform weights with threshold=0.5 -> returns None."""
        n = 100
        w = np.ones(n) / n  # uniform => ESS = N
        result = adaptive_resample(w, threshold=0.5, rng=np.random.default_rng(42))
        assert result is None

    def test_resample_when_ess_low(self) -> None:
        """Very unequal weights with threshold=0.5 -> returns indices."""
        n = 100
        w = np.zeros(n)
        w[0] = 0.99
        w[1:] = 0.01 / (n - 1)
        result = adaptive_resample(w, threshold=0.5, rng=np.random.default_rng(42))
        assert result is not None
        assert len(result) == n

    def test_threshold_boundary(self) -> None:
        """ESS exactly at threshold * N -> should NOT resample (>=)."""
        # For 2 equal weights: ESS = 2, threshold * N = 2 * 0.5 = 1.0
        # ESS = 2 >= 1.0, so no resample
        w = np.array([0.5, 0.5])
        result = adaptive_resample(w, threshold=0.5, rng=np.random.default_rng(42))
        assert result is None

    def test_should_resample_function(self) -> None:
        """Test should_resample helper."""
        w_uniform = np.ones(100) / 100
        assert should_resample(w_uniform, 0.5) is False

        w_degen = np.zeros(100)
        w_degen[0] = 1.0
        assert should_resample(w_degen, 0.5) is True

    def test_resample_with_multinomial(self) -> None:
        """Test adaptive with multinomial base method."""
        n = 100
        w = np.zeros(n)
        w[0] = 0.99
        w[1:] = 0.01 / (n - 1)
        result = adaptive_resample(
            w, threshold=0.5, base_method="multinomial", rng=np.random.default_rng(42)
        )
        assert result is not None
        assert len(result) == n

    def test_unknown_base_method(self) -> None:
        """Unknown base method raises ValueError."""
        import pytest

        n = 100
        w = np.zeros(n)
        w[0] = 0.99
        w[1:] = 0.01 / (n - 1)
        with pytest.raises(ValueError, match="Unknown base method"):
            adaptive_resample(
                w, threshold=0.5, base_method="nonexistent", rng=np.random.default_rng(42)
            )
