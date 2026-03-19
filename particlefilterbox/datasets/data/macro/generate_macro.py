"""Generate simulated macroeconomic datasets.

Creates realistic-looking macro time series data for testing
DSGE and macro models. All data is SIMULATED.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_us_gdp_inflation(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate simulated US GDP growth and inflation (quarterly).

    Uses a simple bivariate VAR-like process calibrated to match
    typical US macro dynamics.

    Parameters
    ----------
    n : int
        Number of quarterly observations. Default is 100 (~25 years).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, gdp_growth, inflation, output_gap.
    """
    rng = np.random.default_rng(seed)

    # VAR(1) parameters (annualized quarterly)
    mu_gdp = 0.5  # ~2% annual GDP growth
    mu_inf = 0.5  # ~2% annual inflation
    rho_gdp = 0.3  # GDP growth persistence
    rho_inf = 0.7  # inflation persistence
    rho_cross = -0.1  # Phillips curve effect
    sigma_gdp = 0.5
    sigma_inf = 0.3

    gdp_growth = np.zeros(n)
    inflation = np.zeros(n)
    output_gap = np.zeros(n)

    gdp_growth[0] = mu_gdp
    inflation[0] = mu_inf

    for t in range(1, n):
        eps_g = rng.standard_normal()
        eps_i = rng.standard_normal()

        gdp_growth[t] = (
            mu_gdp * (1 - rho_gdp)
            + rho_gdp * gdp_growth[t - 1]
            + rho_cross * (inflation[t - 1] - mu_inf)
            + sigma_gdp * eps_g
        )
        inflation[t] = (
            mu_inf * (1 - rho_inf)
            + rho_inf * inflation[t - 1]
            + 0.05 * gdp_growth[t - 1]
            + sigma_inf * eps_i
        )
        output_gap[t] = 0.8 * output_gap[t - 1] + 0.5 * eps_g

    dates = pd.date_range(start="2000-01-01", periods=n, freq="QS")

    df = pd.DataFrame(
        {
            "date": dates,
            "gdp_growth": gdp_growth,
            "inflation": inflation,
            "output_gap": output_gap,
        }
    )

    return df


def generate_interest_rates(n: int = 200, seed: int = 99) -> pd.DataFrame:
    """Generate simulated interest rate data (monthly).

    Uses a CIR-like (Cox-Ingersoll-Ross) process with regime switching.

    Parameters
    ----------
    n : int
        Number of monthly observations.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, fed_funds, tbill_3m, tbond_10y, spread.
    """
    rng = np.random.default_rng(seed)

    # CIR parameters for fed funds
    kappa = 0.1  # mean reversion speed
    theta = 2.0  # long-run mean (%)
    sigma = 0.5  # volatility

    r = np.zeros(n)
    r[0] = 2.0

    for t in range(1, n):
        dr = (
            kappa * (theta - r[t - 1])
            + sigma * np.sqrt(max(r[t - 1], 0.01)) * rng.standard_normal()
        )
        r[t] = max(r[t - 1] + dr / 12, 0.0)  # monthly increment

    # Derive other rates
    tbill_3m = r + 0.1 + 0.2 * rng.standard_normal(n)
    tbond_10y = r + 1.5 + 0.3 * rng.standard_normal(n)
    spread = tbond_10y - tbill_3m

    dates = pd.date_range(start="2005-01-01", periods=n, freq="MS")

    df = pd.DataFrame(
        {
            "date": dates,
            "fed_funds": np.clip(r, 0, None),
            "tbill_3m": np.clip(tbill_3m, 0, None),
            "tbond_10y": np.clip(tbond_10y, 0, None),
            "spread": spread,
        }
    )

    return df


if __name__ == "__main__":
    import os

    output_dir = os.path.dirname(os.path.abspath(__file__))

    gdp = generate_us_gdp_inflation()
    gdp.to_csv(os.path.join(output_dir, "us_gdp_inflation.csv"), index=False)
    print(f"Generated us_gdp_inflation.csv: {len(gdp)} rows")

    rates = generate_interest_rates()
    rates.to_csv(os.path.join(output_dir, "interest_rates.csv"), index=False)
    print(f"Generated interest_rates.csv: {len(rates)} rows")
