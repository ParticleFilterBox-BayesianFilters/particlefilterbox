"""Shared fixtures for PMCMC tests."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pytest
from numpy.typing import NDArray


@dataclass
class SimpleFilterResult:
    """Minimal particle filter result for testing."""

    log_likelihood: float
    filtered_means: NDArray[np.float64] | None = None


class MockPrior:
    """Mock prior distribution for testing.

    Implements a multivariate normal prior.
    """

    def __init__(
        self,
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.dim = len(self.mean)

    def logpdf(self, theta: NDArray[np.float64]) -> float:
        """Evaluate log-prior density."""
        diff = theta - self.mean
        if self.cov.ndim == 1:
            # Diagonal covariance
            return float(
                -0.5 * np.sum(diff**2 / self.cov)
                - 0.5 * self.dim * np.log(2 * np.pi)
                - 0.5 * np.sum(np.log(self.cov))
            )
        inv_cov = np.linalg.inv(self.cov)
        return float(
            -0.5 * diff @ inv_cov @ diff
            - 0.5 * self.dim * np.log(2 * np.pi)
            - 0.5 * np.log(np.linalg.det(self.cov))
        )

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Sample from the prior."""
        return rng.multivariate_normal(self.mean, self.cov)


class MockSSModel:
    """Mock state-space model for testing PMCMC.

    Simple linear Gaussian model:
        x_t = phi * x_{t-1} + sigma_x * eps_t
        y_t = x_t + sigma_y * eta_t

    Parameters theta = [phi, sigma_x, sigma_y].
    """

    def __init__(self) -> None:
        self.params: NDArray[np.float64] = np.array([0.9, 0.5, 1.0])
        self.param_names: list[str] = ["phi", "sigma_x", "sigma_y"]

    def set_params(self, theta: NDArray[np.float64]) -> None:
        """Set model parameters."""
        self.params = np.asarray(theta, dtype=np.float64)

    def get_params(self) -> NDArray[np.float64]:
        """Get current parameters."""
        return self.params.copy()

    def filter(
        self,
        endog: NDArray[np.float64],
        n_particles: int = 100,
        rng: np.random.Generator | None = None,
    ) -> SimpleFilterResult:
        """Run a simple bootstrap particle filter.

        Returns an approximate log-likelihood for the linear Gaussian model.
        """
        if rng is None:
            rng = np.random.default_rng()

        phi, sigma_x, sigma_y = self.params
        T = len(endog)

        # Simple particle filter
        particles = rng.normal(0, sigma_x, size=n_particles)
        log_lik = 0.0

        for t in range(T):
            # Weight: p(y_t | x_t)
            log_weights = -0.5 * ((endog[t] - particles) / sigma_y) ** 2
            max_lw = np.max(log_weights)
            weights = np.exp(log_weights - max_lw)
            sum_weights = np.sum(weights)

            if sum_weights < 1e-300:
                log_lik = -np.inf
                break

            log_lik += max_lw + np.log(sum_weights) - np.log(n_particles)

            # Normalize and resample
            weights /= sum_weights
            indices = rng.choice(n_particles, size=n_particles, p=weights)
            particles = particles[indices]

            # Propagate
            particles = phi * particles + sigma_x * rng.standard_normal(n_particles)

        return SimpleFilterResult(log_likelihood=log_lik)

    def simulate(
        self,
        n_obs: int,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Simulate observations from the model."""
        if rng is None:
            rng = np.random.default_rng()

        phi, sigma_x, sigma_y = self.params
        x = np.zeros(n_obs)
        y = np.zeros(n_obs)

        x[0] = sigma_x * rng.standard_normal()
        y[0] = x[0] + sigma_y * rng.standard_normal()

        for t in range(1, n_obs):
            x[t] = phi * x[t - 1] + sigma_x * rng.standard_normal()
            y[t] = x[t] + sigma_y * rng.standard_normal()

        return y


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def mock_model() -> MockSSModel:
    """Mock state-space model."""
    return MockSSModel()


@pytest.fixture
def mock_prior() -> MockPrior:
    """Mock prior for 3-parameter model."""
    return MockPrior(
        mean=np.array([0.8, 0.5, 1.0]),
        cov=np.diag([0.1, 0.1, 0.1]),
    )


@pytest.fixture
def mock_observations(mock_model: MockSSModel) -> NDArray[np.float64]:
    """Simulated observations from mock model."""
    rng = np.random.default_rng(123)
    return mock_model.simulate(n_obs=50, rng=rng)
