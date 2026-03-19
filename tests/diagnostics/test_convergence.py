"""Tests for convergence study."""

import numpy as np
import pytest

from particlefilterbox.diagnostics.convergence import ConvergenceStudy


class MockModel:
    """Simple linear Gaussian model for testing."""

    def __init__(self, sigma_x: float = 0.5, sigma_y: float = 1.0) -> None:
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def simulate(
        self, n_obs: int, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        if rng is None:
            rng = np.random.default_rng(0)
        states = np.zeros(n_obs, dtype=np.float64)
        obs = np.zeros(n_obs, dtype=np.float64)
        states[0] = rng.normal(0, 1)
        obs[0] = states[0] + rng.normal(0, self.sigma_y)
        for t in range(1, n_obs):
            states[t] = states[t - 1] + rng.normal(0, self.sigma_x)
            obs[t] = states[t] + rng.normal(0, self.sigma_y)
        return states, obs


class MockFilterResult:
    """Mock filter result."""

    def __init__(self, means: np.ndarray) -> None:  # type: ignore[type-arg]
        self.filtered_means = means


class MockFilter:
    """Mock particle filter with known convergence properties."""

    def __init__(self, model: MockModel, n_particles: int) -> None:
        self.model = model
        self.n_particles = n_particles

    def filter(self, obs: np.ndarray) -> MockFilterResult:  # type: ignore[type-arg]
        """Filter with noise proportional to 1/sqrt(N)."""
        rng = np.random.default_rng()
        noise_scale = 1.0 / np.sqrt(self.n_particles)
        means = obs + rng.normal(0, noise_scale, size=len(obs))
        return MockFilterResult(means)


class MockFilterFactory:
    """Factory for creating mock filters."""

    def create(self, model: MockModel, n_particles: int) -> MockFilter:
        return MockFilter(model, n_particles)


class TestConvergenceStudy:
    """Tests for ConvergenceStudy."""

    def test_sqrt_n_rate(self) -> None:
        """Convergence rate should be approximately 0.5 (sqrt(N))."""
        model = MockModel(sigma_y=0.0)
        factory = MockFilterFactory()
        cs = ConvergenceStudy(
            model=model,
            filter_factory=factory,
            n_values=[100, 500, 1000, 5000],
            n_repeats=30,
            n_obs=50,
            seed=42,
        )
        result = cs.run()
        assert 0.3 <= result.rate <= 0.7, (
            f"Rate {result.rate:.3f} not in [0.3, 0.7]"
        )

    def test_rmse_decreases(self) -> None:
        """RMSE should decrease as N increases."""
        model = MockModel(sigma_y=0.0)
        factory = MockFilterFactory()
        cs = ConvergenceStudy(
            model=model,
            filter_factory=factory,
            n_values=[100, 1000, 10000],
            n_repeats=20,
            n_obs=50,
            seed=123,
        )
        result = cs.run()
        # RMSE should be monotonically decreasing
        for i in range(len(result.rmse_values) - 1):
            assert result.rmse_values[i] > result.rmse_values[i + 1], (
                f"RMSE not decreasing: {result.rmse_values[i]:.4f} <= {result.rmse_values[i+1]:.4f}"
            )

    def test_summary(self) -> None:
        """Test summary output."""
        model = MockModel()
        factory = MockFilterFactory()
        cs = ConvergenceStudy(
            model=model,
            filter_factory=factory,
            n_values=[100, 500],
            n_repeats=5,
            n_obs=20,
        )
        cs.run()
        s = cs.summary()
        assert "rate" in s
        assert "r_squared" in s
        assert "rmse_by_n" in s

    def test_rate_property_before_run(self) -> None:
        """Accessing rate before run() should raise."""
        model = MockModel()
        factory = MockFilterFactory()
        cs = ConvergenceStudy(model=model, filter_factory=factory)
        with pytest.raises(RuntimeError, match="Must call run"):
            _ = cs.rate

    def test_rmse_by_n_property(self) -> None:
        """Test rmse_by_n property."""
        model = MockModel()
        factory = MockFilterFactory()
        cs = ConvergenceStudy(
            model=model,
            filter_factory=factory,
            n_values=[100, 500],
            n_repeats=5,
            n_obs=20,
        )
        cs.run()
        rmse_dict = cs.rmse_by_n
        assert 100 in rmse_dict
        assert 500 in rmse_dict
