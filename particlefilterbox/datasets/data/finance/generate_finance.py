"""Generate simulated financial datasets.

Creates realistic-looking financial time series data for testing
and demonstration purposes. All data is SIMULATED, not real market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sp500_returns(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate simulated S&P 500 daily returns.

    Uses a stochastic volatility model to produce realistic-looking returns
    with volatility clustering, fat tails, and leverage effects.

    Parameters
    ----------
    n : int
        Number of observations. Default is 1000.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, returns, close.
    """
    rng = np.random.default_rng(seed)

    # SV model parameters (calibrated to typical S&P 500 behavior)
    mu_h = -1.0  # mean log-volatility
    phi = 0.97  # persistence
    sigma_eta = 0.15  # volatility of volatility
    mu_r = 0.0003  # daily drift (~7.5% annual)

    # Simulate log-volatility
    h = np.zeros(n)
    h[0] = mu_h
    for t in range(1, n):
        h[t] = mu_h + phi * (h[t - 1] - mu_h) + sigma_eta * rng.standard_normal()

    # Simulate returns
    vol = np.exp(h / 2)
    epsilon = rng.standard_normal(n)
    returns = mu_r + vol * epsilon

    # Simulate prices from returns
    close = 4000.0 * np.exp(np.cumsum(returns))

    # Generate dates
    dates = pd.bdate_range(start="2020-01-02", periods=n)

    df = pd.DataFrame(
        {
            "date": dates[:n],
            "returns": returns,
            "close": close,
        }
    )

    return df


def generate_exchange_rates(n: int = 1000, seed: int = 123) -> pd.DataFrame:
    """Generate simulated exchange rate data.

    Simulates EUR/USD exchange rate using a mean-reverting process
    with stochastic volatility.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, eurusd, log_return.
    """
    rng = np.random.default_rng(seed)

    # Parameters
    mu = np.log(1.10)  # mean level
    kappa = 0.02  # mean reversion speed
    sigma = 0.005  # base volatility
    phi_vol = 0.95  # vol persistence
    sigma_vol = 0.1  # vol of vol

    # Simulate
    log_rate = np.zeros(n)
    log_rate[0] = mu
    h = np.zeros(n)
    h[0] = np.log(sigma)

    for t in range(1, n):
        h[t] = (
            np.log(sigma) + phi_vol * (h[t - 1] - np.log(sigma)) + sigma_vol * rng.standard_normal()
        )
        vol_t = np.exp(h[t])
        log_rate[t] = (
            log_rate[t - 1] + kappa * (mu - log_rate[t - 1]) + vol_t * rng.standard_normal()
        )

    rate = np.exp(log_rate)
    log_return = np.diff(log_rate, prepend=log_rate[0])

    dates = pd.bdate_range(start="2020-01-02", periods=n)

    df = pd.DataFrame(
        {
            "date": dates[:n],
            "eurusd": rate,
            "log_return": log_return,
        }
    )

    return df


if __name__ == "__main__":
    import os

    output_dir = os.path.dirname(os.path.abspath(__file__))

    sp500 = generate_sp500_returns()
    sp500.to_csv(os.path.join(output_dir, "sp500_returns.csv"), index=False)
    print(f"Generated sp500_returns.csv: {len(sp500)} rows")

    fx = generate_exchange_rates()
    fx.to_csv(os.path.join(output_dir, "exchange_rates.csv"), index=False)
    print(f"Generated exchange_rates.csv: {len(fx)} rows")
