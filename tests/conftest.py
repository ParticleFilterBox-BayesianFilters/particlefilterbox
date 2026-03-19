"""Shared test fixtures for particlefilterbox."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def uniform_weights_100() -> np.ndarray:
    """100 uniform weights."""
    return np.ones(100) / 100


@pytest.fixture
def degenerate_weights_100() -> np.ndarray:
    """100 weights with all mass on first particle."""
    w = np.zeros(100)
    w[0] = 1.0
    return w
