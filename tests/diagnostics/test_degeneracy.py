"""Tests for degeneracy detection."""

import numpy as np
import pytest

from particlefilterbox.diagnostics.degeneracy import DegeneracyDetector


def make_ancestors_no_resampling(n_particles: int, n_time_steps: int) -> np.ndarray:  # type: ignore[type-arg]
    """Create ancestors for no-resampling case (identity mapping)."""
    ancestors = np.zeros((n_time_steps, n_particles), dtype=np.int64)
    for t in range(n_time_steps):
        ancestors[t] = np.arange(n_particles)
    return ancestors


def make_ancestors_full_coalescence(n_particles: int, n_time_steps: int) -> np.ndarray:  # type: ignore[type-arg]
    """Create ancestors where all particles coalesce to particle 0."""
    ancestors = np.zeros((n_time_steps, n_particles), dtype=np.int64)
    for t in range(n_time_steps):
        # All particles descend from particle 0
        ancestors[t] = np.zeros(n_particles, dtype=np.int64)
    return ancestors


class TestDegeneracyDetector:
    """Tests for DegeneracyDetector."""

    def test_no_resampling_full_diversity(self) -> None:
        """Without resampling, all ancestors should be unique."""
        n, t = 50, 20
        ancestors = make_ancestors_no_resampling(n, t)
        dd = DegeneracyDetector()
        dd.load_ancestors(ancestors)

        ua = dd.unique_ancestors(t=15, lag=5)
        assert ua == n, f"Expected {n} unique ancestors, got {ua}"

    def test_coalescence_detection(self) -> None:
        """Full coalescence should be detected with lag=1."""
        n, t = 50, 20
        ancestors = make_ancestors_full_coalescence(n, t)
        dd = DegeneracyDetector()
        dd.load_ancestors(ancestors)

        # With lag=1, all particles share ancestor 0
        ua = dd.unique_ancestors(t=10, lag=1)
        assert ua == 1, f"Expected 1 unique ancestor, got {ua}"

        ct = dd.coalescence_time(t=10)
        assert ct == 1, f"Expected coalescence time 1, got {ct}"

    def test_is_degenerate(self) -> None:
        """Full coalescence should be flagged as degenerate."""
        n, t = 50, 20
        ancestors = make_ancestors_full_coalescence(n, t)
        dd = DegeneracyDetector()
        dd.load_ancestors(ancestors)
        assert dd.is_degenerate(threshold=0.5, lag=1)

    def test_not_degenerate(self) -> None:
        """No resampling should not be flagged as degenerate."""
        n, t = 50, 20
        ancestors = make_ancestors_no_resampling(n, t)
        dd = DegeneracyDetector()
        dd.load_ancestors(ancestors)
        assert not dd.is_degenerate(threshold=0.5, lag=1)

    def test_requires_ancestors_load(self) -> None:
        """Operations before loading ancestors should raise."""
        dd = DegeneracyDetector()
        with pytest.raises(RuntimeError, match="No ancestors loaded"):
            dd.unique_ancestors(t=5, lag=2)

    def test_requires_2d(self) -> None:
        """1D ancestors should raise ValueError."""
        dd = DegeneracyDetector()
        with pytest.raises(ValueError, match="2D"):
            dd.load_ancestors(np.array([1, 2, 3]))

    def test_mean_coalescence_time(self) -> None:
        """Test mean coalescence time computation."""
        n, t = 50, 20
        ancestors = make_ancestors_full_coalescence(n, t)
        dd = DegeneracyDetector()
        dd.load_ancestors(ancestors)
        mct = dd.mean_coalescence_time()
        assert mct == 1.0, f"Expected mean coalescence time 1.0, got {mct}"

    def test_summary(self) -> None:
        """Test summary output."""
        n, t = 50, 20
        ancestors = make_ancestors_no_resampling(n, t)
        dd = DegeneracyDetector()
        dd.load_ancestors(ancestors)
        s = dd.summary()
        assert s["n_particles"] == n
        assert s["n_time_steps"] == t
        assert "is_degenerate" in s

    def test_ancestral_tree_data(self) -> None:
        """Test ancestral tree data output."""
        n, t = 50, 30
        ancestors = make_ancestors_no_resampling(n, t)
        dd = DegeneracyDetector()
        dd.load_ancestors(ancestors)
        data = dd.ancestral_tree_data()
        assert "unique_ancestor_fractions" in data
        assert "lags" in data
        assert "time_steps" in data

    def test_unique_ancestors_lag_zero(self) -> None:
        """Lag 0 should return N."""
        n, t = 50, 20
        ancestors = make_ancestors_no_resampling(n, t)
        dd = DegeneracyDetector()
        dd.load_ancestors(ancestors)
        assert dd.unique_ancestors(t=10, lag=0) == n

    def test_load_from_result(self) -> None:
        """Test loading from a result-like object."""

        class FakeResult:
            def __init__(self, anc: np.ndarray) -> None:  # type: ignore[type-arg]
                self.ancestors = anc

        n, t = 20, 10
        dd = DegeneracyDetector()
        dd.load_from_result(FakeResult(make_ancestors_no_resampling(n, t)))
        assert dd.unique_ancestors(t=5, lag=2) == n

    def test_load_from_result_no_attr(self) -> None:
        """Result without ancestors should raise."""

        class BadResult:
            pass

        dd = DegeneracyDetector()
        with pytest.raises(AttributeError):
            dd.load_from_result(BadResult())

    def test_load_from_result_none_ancestors(self) -> None:
        """Result with None ancestors should raise."""

        class NoneResult:
            ancestors = None

        dd = DegeneracyDetector()
        with pytest.raises(RuntimeError, match="store_ancestors"):
            dd.load_from_result(NoneResult())
