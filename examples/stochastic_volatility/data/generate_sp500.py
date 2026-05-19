"""Generate synthetic SP500-like returns dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sp500_returns(
    n: int = 2500,
    mu_sv: float = -9.0,
    phi: float = 0.98,
    sigma_h: float = 0.12,
    mu_ret: float = 0.0003,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate SP500-calibrated returns from SV model."""
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    returns = np.zeros(n)

    h[0] = mu_sv + sigma_h / np.sqrt(1 - phi**2) * rng.standard_normal()

    for t in range(n):
        if t > 0:
            h[t] = mu_sv + phi * (h[t - 1] - mu_sv) + sigma_h * rng.standard_normal()
        returns[t] = mu_ret + np.exp(h[t] / 2) * rng.standard_normal()

    dates = pd.bdate_range("2014-01-02", periods=n)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "returns": np.round(returns, 8),
    })


if __name__ == "__main__":
    import os

    data_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_sp500_returns()
    df.to_csv(os.path.join(data_dir, "sp500_returns.csv"), index=False)
    print(f"SP500 returns: {len(df)} obs, mean={df['returns'].mean():.6f}, std={df['returns'].std():.6f}")
