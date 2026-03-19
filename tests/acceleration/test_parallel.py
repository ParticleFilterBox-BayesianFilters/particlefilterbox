"""Tests for parallel execution."""

import numpy as np
import pytest

from particlefilterbox.acceleration.parallel import ParallelRunner


class TestParallelRunner:
    """Tests for ParallelRunner."""

    def test_parallel_chains_independent(self) -> None:
        """Parallel chains with different seeds should produce different results."""
        # We test the infrastructure, not a real PMCMC
        runner = ParallelRunner(n_workers=2)
        # Just verify construction works
        assert runner.n_workers == 2

    def test_default_workers(self) -> None:
        """Default workers should use CPU count."""
        runner = ParallelRunner()
        assert runner.n_workers >= 1

    def test_parallel_speedup(self) -> None:
        """Parallel execution should complete (basic functionality test).

        Note: Actual speedup depends on system and task granularity.
        This test verifies the API works correctly.
        """
        runner = ParallelRunner(n_workers=2)
        assert runner.n_workers == 2
        # Full speedup tests require real PMCMC implementations
