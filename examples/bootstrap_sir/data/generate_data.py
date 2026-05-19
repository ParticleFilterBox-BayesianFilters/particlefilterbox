"""Generate synthetic datasets for Bootstrap PF and SIR examples.

Datasets:
- simulated_linear_gaussian.csv: Linear-Gaussian state-space model (Kalman benchmark)
- simulated_sv.csv: Stochastic volatility model

All datasets use fixed seeds for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_linear_gaussian(
    n: int = 500,
    phi: float = 0.95,
    sigma_x: float = 0.5,
    sigma_y: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate linear-Gaussian state-space model data.

    x_t = phi * x_{t-1} + sigma_x * w_t
    y_t = x_t + sigma_y * v_t
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)

    x[0] = sigma_x / np.sqrt(1 - phi**2) * rng.standard_normal()
    y[0] = x[0] + sigma_y * rng.standard_normal()

    for t in range(1, n):
        x[t] = phi * x[t - 1] + sigma_x * rng.standard_normal()
        y[t] = x[t] + sigma_y * rng.standard_normal()

    return pd.DataFrame({
        "t": np.arange(n),
        "x_true": np.round(x, 8),
        "y_obs": np.round(y, 8),
    })


def generate_sv(
    n: int = 1000,
    mu: float = -1.0,
    phi: float = 0.97,
    sigma_h: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate stochastic volatility model data.

    h_t = mu + phi*(h_{t-1} - mu) + sigma_h * w_t
    y_t = exp(h_t/2) * v_t
    """
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    y = np.zeros(n)

    h[0] = mu + sigma_h / np.sqrt(1 - phi**2) * rng.standard_normal()
    y[0] = np.exp(h[0] / 2) * rng.standard_normal()

    for t in range(1, n):
        h[t] = mu + phi * (h[t - 1] - mu) + sigma_h * rng.standard_normal()
        y[t] = np.exp(h[t] / 2) * rng.standard_normal()

    return pd.DataFrame({
        "t": np.arange(n),
        "h_true": np.round(h, 8),
        "y_obs": np.round(y, 8),
    })


if __name__ == "__main__":
    import os

    data_dir = os.path.dirname(os.path.abspath(__file__))

    df_lg = generate_linear_gaussian()
    df_lg.to_csv(os.path.join(data_dir, "simulated_linear_gaussian.csv"), index=False)
    print(f"Linear-Gaussian: {len(df_lg)} obs, columns: {list(df_lg.columns)}")

    df_sv = generate_sv()
    df_sv.to_csv(os.path.join(data_dir, "simulated_sv.csv"), index=False)
    print(f"Stochastic Volatility: {len(df_sv)} obs, columns: {list(df_sv.columns)}")
