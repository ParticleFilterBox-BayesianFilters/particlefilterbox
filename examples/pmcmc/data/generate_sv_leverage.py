"""Generate SV with leverage dataset for PMCMC examples."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sv_leverage(
    n: int = 1000,
    mu: float = -1.0,
    phi: float = 0.97,
    sigma_h: float = 0.15,
    rho: float = -0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate SV model with leverage (correlated innovations).

    h_t = mu + phi*(h_{t-1} - mu) + sigma_h * eta_t
    y_t = exp(h_t/2) * eps_t
    Corr(eta_t, eps_t) = rho
    """
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    y = np.zeros(n)

    # Correlated innovations via Cholesky
    h[0] = mu + sigma_h / np.sqrt(1 - phi**2) * rng.standard_normal()

    for t in range(n):
        z1 = rng.standard_normal()
        z2 = rng.standard_normal()
        eta = z1
        eps = rho * z1 + np.sqrt(1 - rho**2) * z2

        if t > 0:
            h[t] = mu + phi * (h[t - 1] - mu) + sigma_h * eta
        y[t] = np.exp(h[t] / 2) * eps

    return pd.DataFrame({
        "t": np.arange(n),
        "h_true": np.round(h, 8),
        "y_obs": np.round(y, 8),
    })


if __name__ == "__main__":
    import os

    data_dir = os.path.dirname(os.path.abspath(__file__))

    df = generate_sv_leverage()
    df.to_csv(os.path.join(data_dir, "simulated_sv_leverage.csv"), index=False)
    print(f"SV Leverage: {len(df)} obs, rho=-0.5")
