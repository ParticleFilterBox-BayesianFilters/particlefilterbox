"""Generate simulated datasets for validation.

Creates simple datasets where the true model is known exactly,
enabling validation of particle filter accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_linear_gaussian(
    n: int = 500,
    seed: int = 42,
    a: float = 0.9,
    c: float = 1.0,
    q: float = 1.0,
    r: float = 1.0,
) -> pd.DataFrame:
    """Generate linear Gaussian state-space data.

    x_t = A * x_{t-1} + w_t,  w_t ~ N(0, Q)
    y_t = C * x_t + v_t,      v_t ~ N(0, R)

    This is the gold standard for validation: the Kalman filter provides
    the exact solution, so particle filter accuracy can be verified.

    Parameters
    ----------
    n : int
        Number of observations. Default is 500.
    seed : int
        Random seed.
    A, C, Q, R : float
        Model parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: t, state, observation, A, C, Q, R.
    """
    rng = np.random.default_rng(seed)

    states = np.zeros(n)
    observations = np.zeros(n)

    states[0] = rng.standard_normal() * np.sqrt(q / (1 - a**2))

    for t in range(1, n):
        states[t] = a * states[t - 1] + np.sqrt(q) * rng.standard_normal()

    for t in range(n):
        observations[t] = c * states[t] + np.sqrt(r) * rng.standard_normal()

    df = pd.DataFrame(
        {
            "t": np.arange(n),
            "state": states,
            "observation": observations,
            "A": a,
            "C": c,
            "Q": q,
            "R": r,
        }
    )

    return df


if __name__ == "__main__":
    import os

    output_dir = os.path.dirname(os.path.abspath(__file__))

    lg = generate_linear_gaussian()
    lg.to_csv(os.path.join(output_dir, "linear_gaussian.csv"), index=False)
    print(f"Generated linear_gaussian.csv: {len(lg)} rows")
