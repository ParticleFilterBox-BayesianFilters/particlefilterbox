"""Random number generator management for reproducibility."""

from __future__ import annotations

import numpy as np


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Create or return a NumPy random Generator.

    Parameters
    ----------
    seed : int or None
        Seed for the generator. If None, uses entropy from the OS.

    Returns
    -------
    np.random.Generator
        A seeded random number generator.
    """
    return np.random.default_rng(seed)


def spawn_rngs(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Spawn N independent random generators from a parent.

    Uses SeedSequence.spawn() for statistically independent streams.

    Parameters
    ----------
    rng : np.random.Generator
        Parent generator.
    n : int
        Number of child generators to spawn.

    Returns
    -------
    list[np.random.Generator]
        List of N independent generators.
    """
    bit_gen = rng.bit_generator
    ss = bit_gen.seed_seq
    if not isinstance(ss, np.random.SeedSequence):
        ss = np.random.SeedSequence()
    child_seeds = ss.spawn(n)
    return [np.random.default_rng(s) for s in child_seeds]
