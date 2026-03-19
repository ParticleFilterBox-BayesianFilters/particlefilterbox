"""Dataset loading utilities for particlefilterbox.

Provides a unified interface for loading bundled datasets including
financial, macroeconomic, and simulated data.

All datasets are SIMULATED for testing and demonstration purposes.

Examples
--------
>>> from particlefilterbox.datasets import load_dataset, list_datasets
>>> print(list_datasets())
>>> sp500 = load_dataset('sp500_returns')
>>> print(sp500.head())
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_DATA_DIR = Path(__file__).parent / "data"

DATASETS: dict[str, dict[str, Any]] = {
    # Finance
    "sp500_returns": {
        "path": "finance/sp500_returns.csv",
        "description": "Simulated S&P 500 daily returns (~1000 obs)",
        "category": "finance",
        "columns": ["date", "returns", "close"],
    },
    "exchange_rates": {
        "path": "finance/exchange_rates.csv",
        "description": "Simulated EUR/USD exchange rates (~1000 obs)",
        "category": "finance",
        "columns": ["date", "eurusd", "log_return"],
    },
    # Macro
    "us_gdp_inflation": {
        "path": "macro/us_gdp_inflation.csv",
        "description": "Simulated US GDP growth and inflation (quarterly, ~100 obs)",
        "category": "macro",
        "columns": ["date", "gdp_growth", "inflation", "output_gap"],
    },
    "interest_rates": {
        "path": "macro/interest_rates.csv",
        "description": "Simulated interest rates (monthly, ~200 obs)",
        "category": "macro",
        "columns": ["date", "fed_funds", "tbill_3m", "tbond_10y", "spread"],
    },
    # Simulated
    "linear_gaussian": {
        "path": "simulated/linear_gaussian.csv",
        "description": "Linear Gaussian state-space (validation, ~500 obs)",
        "category": "simulated",
        "columns": ["t", "state", "observation", "A", "C", "Q", "R"],
    },
    "sv_basic": {
        "path": "simulated/sv_basic.csv",
        "description": "Simulated stochastic volatility data (T=500, mu=-1, phi=0.97, sigma=0.15)",
        "category": "simulated",
        "columns": ["y", "h"],
    },
    "sv_leverage": {
        "path": "simulated/sv_leverage.csv",
        "description": "Simulated SV with leverage effect (~500 obs)",
        "category": "simulated",
        "columns": ["observation", "state_h"],
    },
    "sv_jumps": {
        "path": "simulated/sv_jumps.csv",
        "description": "Simulated SV with jumps (~500 obs)",
        "category": "simulated",
        "columns": ["observation", "state_h", "state_q"],
    },
    "jump_diffusion": {
        "path": "simulated/jump_diffusion.csv",
        "description": "Simulated jump-diffusion returns (~1000 obs)",
        "category": "simulated",
        "columns": [
            "t", "merton_return", "merton_log_price",
            "kou_return", "kou_log_price",
            "bates_return", "bates_log_price", "bates_variance",
        ],
    },
    "sir_epidemic": {
        "path": "simulated/sir_epidemic.csv",
        "description": "Simulated SIR epidemic data (~200 obs)",
        "category": "simulated",
        "columns": ["t", "S", "I", "R", "reported_cases"],
    },
}


def generate_sv_data(
    t_steps: int = 500,
    mu: float = -1.0,
    phi: float = 0.97,
    sigma: float = 0.15,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate simulated Stochastic Volatility data.

    Model:
        h_t = mu + phi * (h_{t-1} - mu) + sigma * eta_t,  eta_t ~ N(0,1)
        y_t = exp(h_t / 2) * eps_t,                        eps_t ~ N(0,1)

    Parameters
    ----------
    t_steps : int
        Number of time steps.
    mu : float
        Long-run mean of log-volatility.
    phi : float
        Persistence parameter.
    sigma : float
        Volatility of log-volatility.
    seed : int
        Random seed.

    Returns
    -------
    tuple[ndarray, ndarray]
        (y, h) where y is observations (T,) and h is latent log-volatility (T,).
    """
    rng = np.random.default_rng(seed)
    h = np.zeros(t_steps)
    y = np.zeros(t_steps)
    h[0] = mu
    for i in range(1, t_steps):
        h[i] = mu + phi * (h[i - 1] - mu) + sigma * rng.standard_normal()
    for i in range(t_steps):
        y[i] = np.exp(h[i] / 2) * rng.standard_normal()
    return y, h


def load_dataset(
    name: str,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load a bundled dataset by name.

    Parameters
    ----------
    name : str
        Dataset name. Use `list_datasets()` to see available names.
    parse_dates : bool
        Whether to parse date columns. Default is True.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    ValueError
        If dataset name is not recognized.
    FileNotFoundError
        If the dataset file is missing.

    Examples
    --------
    >>> df = load_dataset('sp500_returns')
    >>> print(df.shape)
    (1000, 3)
    """
    if name not in DATASETS:
        available = ", ".join(sorted(DATASETS.keys()))
        msg = f"Unknown dataset '{name}'. Available: {available}"
        raise ValueError(msg)

    info = DATASETS[name]
    filepath = _DATA_DIR / info["path"]

    if not filepath.exists():
        msg = f"Dataset file not found: {filepath}. Run the data generation scripts to create it."
        raise FileNotFoundError(msg)

    columns = info.get("columns", [])
    date_cols = ["date"] if "date" in columns and parse_dates else None
    df = pd.read_csv(filepath, parse_dates=date_cols)

    return df


def list_datasets(category: str | None = None) -> list[dict[str, str]]:
    """List available datasets.

    Parameters
    ----------
    category : str or None
        Filter by category ('finance', 'macro', 'simulated').
        If None, returns all datasets.

    Returns
    -------
    list of dict
        List of dicts with 'name', 'description', 'category'.
    """
    result = []
    for name, info in sorted(DATASETS.items()):
        if category is not None and info.get("category") != category:
            continue
        result.append(
            {
                "name": name,
                "description": info["description"],
                "category": info.get("category", ""),
            }
        )
    return result


def get_dataset_info(name: str) -> dict[str, Any]:
    """Get metadata for a dataset.

    Parameters
    ----------
    name : str
        Dataset name.

    Returns
    -------
    dict
        Dataset metadata including path, description, columns.
    """
    if name not in DATASETS:
        available = ", ".join(sorted(DATASETS.keys()))
        msg = f"Unknown dataset '{name}'. Available: {available}"
        raise ValueError(msg)
    return DATASETS[name].copy()


# ---------------------------------------------------------------------------
# CSV-based convenience loaders (numpy arrays, no pandas dependency)
# ---------------------------------------------------------------------------


def _load_csv(
    filepath: Path,
    columns: list[str] | None = None,
    skip_header: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """Load a CSV file into a dict of numpy arrays.

    Parameters
    ----------
    filepath : Path
        Path to CSV file.
    columns : list[str] | None
        Column names to load. If None, load all.
    skip_header : bool
        Whether to skip header row.

    Returns
    -------
    dict
        Column name -> array.
    """
    with open(filepath) as f:
        reader = csv.reader(f)
        header = next(reader) if skip_header else [f"col_{i}" for i in range(100)]
        rows = list(reader)

    if not rows:
        return {}

    data: dict[str, list[float]] = {h: [] for h in header}
    for row in rows:
        for i, val in enumerate(row):
            if i < len(header):
                try:
                    data[header[i]].append(float(val))
                except ValueError:
                    data[header[i]].append(float("nan"))

    result = {}
    for h in header:
        if columns is None or h in columns:
            result[h] = np.array(data[h])
    return result


def load_sv_leverage() -> dict[str, NDArray[np.float64]]:
    """Load simulated SV leverage dataset."""
    return _load_csv(_DATA_DIR / "simulated" / "sv_leverage.csv")


def load_sv_jumps() -> dict[str, NDArray[np.float64]]:
    """Load simulated SV jumps dataset."""
    return _load_csv(_DATA_DIR / "simulated" / "sv_jumps.csv")


def load_jump_diffusion() -> dict[str, NDArray[np.float64]]:
    """Load simulated jump diffusion dataset."""
    return _load_csv(_DATA_DIR / "simulated" / "jump_diffusion.csv")


def load_sir_epidemic() -> dict[str, NDArray[np.float64]]:
    """Load simulated SIR epidemic dataset."""
    return _load_csv(_DATA_DIR / "simulated" / "sir_epidemic.csv")


def load_sp500_returns() -> dict[str, NDArray[np.float64]]:
    """Load simulated S&P 500 returns dataset."""
    return _load_csv(
        _DATA_DIR / "finance" / "sp500_returns.csv",
        columns=["returns", "close"],
    )


def load_us_gdp_inflation() -> dict[str, NDArray[np.float64]]:
    """Load simulated US GDP and inflation dataset."""
    return _load_csv(
        _DATA_DIR / "macro" / "us_gdp_inflation.csv",
        columns=["gdp_growth", "inflation", "output_gap"],
    )


def list_datasets_csv() -> list[str]:
    """List all available datasets as category/name strings."""
    datasets = []
    for subdir in ["simulated", "finance", "macro"]:
        dirpath = _DATA_DIR / subdir
        if dirpath.exists():
            for f in sorted(dirpath.glob("*.csv")):
                datasets.append(f"{subdir}/{f.stem}")
    return datasets


def load_dataset_csv(name: str) -> dict[str, NDArray[np.float64]]:
    """Load a dataset by name (e.g., 'simulated/sv_leverage').

    Parameters
    ----------
    name : str
        Dataset name in format 'category/name'.

    Returns
    -------
    dict
        Column name -> array.
    """
    filepath = _DATA_DIR / f"{name}.csv"
    if not filepath.exists():
        available = list_datasets_csv()
        raise FileNotFoundError(
            f"Dataset '{name}' not found. Available: {available}"
        )
    return _load_csv(filepath)
