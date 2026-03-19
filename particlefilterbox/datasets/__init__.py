"""Datasets module for particlefilterbox.

Provides bundled datasets for testing, benchmarking, and demonstration.
All data is SIMULATED for reproducibility.

Categories:
    - finance: Simulated financial time series (returns, exchange rates)
    - macro: Simulated macroeconomic data (GDP, inflation, interest rates)
    - simulated: Simple synthetic data for validation

Examples
--------
>>> from particlefilterbox.datasets import load_dataset, list_datasets
>>> datasets = list_datasets()
>>> sp500 = load_dataset('sp500_returns')
"""

from __future__ import annotations

from particlefilterbox.datasets.load import (
    DATASETS,
    generate_sv_data,
    get_dataset_info,
    list_datasets,
    list_datasets_csv,
    load_dataset,
    load_dataset_csv,
    load_jump_diffusion,
    load_sir_epidemic,
    load_sp500_returns,
    load_sv_jumps,
    load_sv_leverage,
    load_us_gdp_inflation,
)

__all__ = [
    "DATASETS",
    "generate_sv_data",
    "get_dataset_info",
    "list_datasets",
    "list_datasets_csv",
    "load_dataset",
    "load_dataset_csv",
    "load_jump_diffusion",
    "load_sir_epidemic",
    "load_sp500_returns",
    "load_sv_jumps",
    "load_sv_leverage",
    "load_us_gdp_inflation",
]
