"""Tests for weight analysis."""

import numpy as np
import pytest

from particlefilterbox.diagnostics.weight_analysis import WeightAnalysis


class TestWeightAnalysis:
    """Tests for WeightAnalysis."""

    def test_entropy_uniform_equals_log_n(self) -> None:
        """Uniform weights should have entropy = log(N)."""
        n = 200
        weights = np.ones(n) / n
        h = WeightAnalysis.entropy(weights)
        expected = np.log(n)
        assert abs(h - expected) < 1e-10, f"Entropy should be {expected}, got {h}"

    def test_entropy_degenerate_equals_zero(self) -> None:
        """Degenerate weights should have entropy = 0."""
        n = 200
        weights = np.zeros(n)
        weights[0] = 1.0
        h = WeightAnalysis.entropy(weights)
        assert abs(h) < 1e-10, f"Entropy should be 0, got {h}"

    def test_gini_uniform_near_zero(self) -> None:
        """Uniform weights should have Gini index near 0."""
        n = 1000
        weights = np.ones(n) / n
        gini = WeightAnalysis.gini_index(weights)
        assert abs(gini) < 0.01, f"Gini should be ~0 for uniform, got {gini}"

    def test_gini_degenerate_near_one(self) -> None:
        """Degenerate weights should have Gini index near 1."""
        n = 1000
        weights = np.zeros(n)
        weights[0] = 1.0
        gini = WeightAnalysis.gini_index(weights)
        assert gini > 0.99, f"Gini should be ~1 for degenerate, got {gini}"

    def test_cv_uniform_zero(self) -> None:
        """Uniform weights should have CV = 0."""
        n = 100
        weights = np.ones(n) / n
        cv = WeightAnalysis.coefficient_of_variation(weights)
        assert abs(cv) < 1e-10, f"CV should be 0 for uniform, got {cv}"

    def test_cv_degenerate_high(self) -> None:
        """Degenerate weights should have high CV."""
        n = 100
        weights = np.zeros(n)
        weights[0] = 1.0
        cv = WeightAnalysis.coefficient_of_variation(weights)
        assert cv > 5.0, f"CV should be high for degenerate, got {cv}"

    def test_entropy_history(self) -> None:
        """Test entropy_history tracking."""
        wa = WeightAnalysis()
        n = 100
        for _ in range(5):
            w = np.ones(n) / n
            wa.update_from_weights(w)
        hist = wa.entropy_history()
        assert len(hist) == 5
        assert all(abs(h - np.log(n)) < 1e-10 for h in hist)

    def test_gini_history(self) -> None:
        """Test gini_history tracking."""
        wa = WeightAnalysis()
        n = 100
        wa.update_from_weights(np.ones(n) / n)
        hist = wa.gini_history()
        assert len(hist) == 1
        assert abs(hist[0]) < 0.05

    def test_summary(self) -> None:
        """Test summary output."""
        wa = WeightAnalysis()
        n = 100
        wa.update_from_weights(np.ones(n) / n)
        s = wa.summary()
        assert s["n_time_steps"] == 1
        assert "entropy_current" in s
        assert "gini_current" in s
        assert "cv_current" in s
        assert "max_weight_current" in s
        assert abs(s["entropy_ratio"] - 1.0) < 1e-8

    def test_update_from_cloud(self) -> None:
        """Test update from cloud-like object."""

        class FakeCloud:
            def __init__(self, w: np.ndarray) -> None:  # type: ignore[type-arg]
                self.weights = w

        wa = WeightAnalysis()
        wa.update(FakeCloud(np.ones(50) / 50))
        assert wa.summary()["n_time_steps"] == 1

    def test_reset(self) -> None:
        """Test reset clears state."""
        wa = WeightAnalysis()
        wa.update_from_weights(np.ones(100) / 100)
        wa.reset()
        assert wa.summary()["n_time_steps"] == 0
