"""Shared fixtures for SMC tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def gaussian_log_target() -> Any:
    """Standard 2-D Gaussian log-target for testing.

    Target: N(mu=[1, 2], Sigma=I)
    """
    mu = np.array([1.0, 2.0])

    def log_target(theta: NDArray[np.floating[Any]]) -> float:
        diff = theta - mu
        return float(-0.5 * np.sum(diff**2))

    return log_target


@pytest.fixture
def gaussian_1d_log_target() -> Any:
    """1-D Gaussian log-target: N(mu=3, sigma=1)."""
    mu = 3.0

    def log_target(theta: NDArray[np.floating[Any]]) -> float:
        return float(-0.5 * (theta[0] - mu) ** 2)

    return log_target
