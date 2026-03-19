"""Shared test fixtures for particle smoother tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest


@dataclass
class MockFilterResults:
    """Mock filter results for testing smoothers.

    Simulates a linear Gaussian model:
        x_t = A * x_{t-1} + w_t,  w_t ~ N(0, Q)
        y_t = H * x_t + v_t,      v_t ~ N(0, R)

    Attributes
    ----------
    particles_history : list[np.ndarray]
        List of particle arrays, each (N, k).
    weights_history : list[np.ndarray]
        List of normalized weight arrays, each (N,).
    filtered_mean : np.ndarray
        Filtered means, shape (T, k).
    filtered_cov : np.ndarray
        Filtered covariances, shape (T, k, k).
    true_states : np.ndarray
        True hidden states, shape (T, k).
    observations : np.ndarray
        Observations, shape (T, m).
    """

    particles_history: list[np.ndarray] = field(default_factory=list)
    weights_history: list[np.ndarray] = field(default_factory=list)
    filtered_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    filtered_cov: np.ndarray = field(default_factory=lambda: np.array([]))
    true_states: np.ndarray = field(default_factory=lambda: np.array([]))
    observations: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class MockLinearGaussianModel:
    """Mock linear Gaussian state-space model for testing.

    x_t = A * x_{t-1} + w_t,  w_t ~ N(0, Q)
    y_t = H * x_t + v_t,      v_t ~ N(0, R)

    Attributes
    ----------
    A : np.ndarray
        State transition matrix, shape (k, k).
    Q : np.ndarray
        Process noise covariance, shape (k, k).
    H : np.ndarray
        Observation matrix, shape (m, k).
    R : np.ndarray
        Observation noise covariance, shape (m, m).
    """

    A: np.ndarray = field(default_factory=lambda: np.array([[0.9]]))
    Q: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))
    H: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))
    R: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))

    def log_transition_density(
        self, x_new: np.ndarray, x_old: np.ndarray, t: int
    ) -> np.ndarray:
        """Compute log p(x_new | x_old) for linear Gaussian transition.

        Parameters
        ----------
        x_new : np.ndarray
            New state, shape (N, k) or (k,).
        x_old : np.ndarray
            Old state, shape (N, k) or (k,).
        t : int
            Time index (unused for time-homogeneous model).

        Returns
        -------
        np.ndarray
            Log transition densities, shape (N,) or scalar.
        """
        if x_new.ndim == 1:
            x_new = x_new.reshape(1, -1)
        if x_old.ndim == 1:
            x_old = x_old.reshape(1, -1)

        mean = x_old @ self.A.T  # (N, k)
        diff = x_new - mean  # (N, k)

        k = self.Q.shape[0]
        Q_inv = np.linalg.inv(self.Q)
        log_det_Q = np.log(np.linalg.det(self.Q))

        # Mahalanobis distance
        mahal = np.sum(diff @ Q_inv * diff, axis=1)  # (N,)

        log_density: np.ndarray = -0.5 * (k * np.log(2 * np.pi) + log_det_Q + mahal)
        return log_density

    def transition_sample(
        self, x_old: np.ndarray, t: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample from transition density p(x_t | x_{t-1}).

        Parameters
        ----------
        x_old : np.ndarray
            Old state, shape (N, k).
        t : int
            Time index.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            New state samples, shape (N, k).
        """
        N, k = x_old.shape
        mean = x_old @ self.A.T
        noise = rng.multivariate_normal(np.zeros(k), self.Q, size=N)
        return mean + noise

    def log_observation_density(
        self, y: np.ndarray, x: np.ndarray, t: int
    ) -> np.ndarray:
        """Compute log p(y_t | x_t) for linear Gaussian observation.

        Parameters
        ----------
        y : np.ndarray
            Observation, shape (m,).
        x : np.ndarray
            State, shape (N, k).
        t : int
            Time index.

        Returns
        -------
        np.ndarray
            Log observation densities, shape (N,).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        predicted_obs = x @ self.H.T  # (N, m)
        diff = y - predicted_obs  # (N, m)

        m = self.R.shape[0]
        R_inv = np.linalg.inv(self.R)
        log_det_R = np.log(np.linalg.det(self.R))

        mahal = np.sum(diff @ R_inv * diff, axis=1)
        log_density: np.ndarray = -0.5 * (m * np.log(2 * np.pi) + log_det_R + mahal)
        return log_density


def _generate_linear_gaussian_data(
    T: int = 50,
    N: int = 200,
    k: int = 1,
    A: float = 0.9,
    Q: float = 1.0,
    R: float = 1.0,
    seed: int = 42,
) -> tuple[MockFilterResults, MockLinearGaussianModel]:
    """Generate synthetic data and mock filter results for a linear Gaussian model.

    Parameters
    ----------
    T : int
        Number of timesteps.
    N : int
        Number of particles.
    k : int
        State dimension.
    A : float
        Scalar transition coefficient (used as A * I_k).
    Q : float
        Scalar process noise variance (used as Q * I_k).
    R : float
        Scalar observation noise variance (used as R * I_k).
    seed : int
        Random seed.

    Returns
    -------
    tuple[MockFilterResults, MockLinearGaussianModel]
        Mock filter results and the model.
    """
    rng = np.random.default_rng(seed)

    A_mat = A * np.eye(k)
    Q_mat = Q * np.eye(k)
    H_mat = np.eye(k)
    R_mat = R * np.eye(k)

    model = MockLinearGaussianModel(A=A_mat, Q=Q_mat, H=H_mat, R=R_mat)

    # Generate true states and observations
    true_states = np.zeros((T, k))
    observations = np.zeros((T, k))

    true_states[0] = rng.multivariate_normal(np.zeros(k), Q_mat)
    observations[0] = true_states[0] + rng.multivariate_normal(np.zeros(k), R_mat)

    for t in range(1, T):
        true_states[t] = A * true_states[t - 1] + rng.multivariate_normal(
            np.zeros(k), Q_mat
        )
        observations[t] = true_states[t] + rng.multivariate_normal(np.zeros(k), R_mat)

    # Run a simple bootstrap PF to generate realistic filter results
    particles_history: list[np.ndarray] = []
    weights_history: list[np.ndarray] = []
    filtered_mean = np.zeros((T, k))
    filtered_cov = np.zeros((T, k, k))

    # Initialize particles
    particles = rng.multivariate_normal(np.zeros(k), 2 * Q_mat, size=N)  # (N, k)

    for t in range(T):
        if t > 0:
            # Propagate
            particles = model.transition_sample(particles, t, rng)

        # Weight
        log_w = model.log_observation_density(observations[t], particles, t)
        log_weights = log_w

        # Normalize
        max_lw = np.max(log_weights)
        w = np.exp(log_weights - max_lw)
        w_sum = np.sum(w)
        w_normalized = w / w_sum

        # Store
        particles_history.append(particles.copy())
        weights_history.append(w_normalized.copy())

        # Compute filtered statistics
        filtered_mean[t] = np.sum(w_normalized[:, np.newaxis] * particles, axis=0)
        diff = particles - filtered_mean[t]
        filtered_cov[t] = np.sum(
            w_normalized[:, np.newaxis, np.newaxis]
            * (diff[:, :, np.newaxis] * diff[:, np.newaxis, :]),
            axis=0,
        )

        # Resample (systematic)
        ess = 1.0 / np.sum(w_normalized**2)
        if ess < N / 2:
            cumsum = np.cumsum(w_normalized)
            u = (rng.random() + np.arange(N)) / N
            indices = np.searchsorted(cumsum, u)
            indices = np.clip(indices, 0, N - 1)
            particles = particles[indices].copy()

    filter_results = MockFilterResults(
        particles_history=particles_history,
        weights_history=weights_history,
        filtered_mean=filtered_mean,
        filtered_cov=filtered_cov,
        true_states=true_states,
        observations=observations,
    )

    return filter_results, model


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def linear_gaussian_data() -> tuple[MockFilterResults, MockLinearGaussianModel]:
    """Generate linear Gaussian filter results and model.

    Returns T=50 timesteps, N=200 particles, k=1 state dimension.
    """
    return _generate_linear_gaussian_data(T=50, N=200, k=1, seed=42)


@pytest.fixture
def linear_gaussian_data_2d() -> tuple[MockFilterResults, MockLinearGaussianModel]:
    """Generate 2D linear Gaussian filter results and model.

    Returns T=30 timesteps, N=300 particles, k=2 state dimension.
    """
    return _generate_linear_gaussian_data(T=30, N=300, k=2, seed=123)


@pytest.fixture
def mock_model() -> MockLinearGaussianModel:
    """Simple 1D linear Gaussian model with A=0.9, Q=1.0, R=1.0."""
    return MockLinearGaussianModel(
        A=np.array([[0.9]]),
        Q=np.array([[1.0]]),
        H=np.array([[1.0]]),
        R=np.array([[1.0]]),
    )


@pytest.fixture
def mock_filter_results_no_particles() -> Any:
    """Filter results without particles_history (should fail validation)."""

    @dataclass
    class BadResults:
        particles_history: None = None
        weights_history: list[np.ndarray] = field(default_factory=list)

    return BadResults()


@pytest.fixture
def mock_filter_results_no_weights() -> Any:
    """Filter results without weights_history (should fail validation)."""

    @dataclass
    class BadResults:
        particles_history: list[np.ndarray] = field(
            default_factory=lambda: [np.ones((10, 1))]
        )
        weights_history: None = None

    return BadResults()
