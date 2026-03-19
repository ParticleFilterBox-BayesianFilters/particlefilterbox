"""Command-line interface for particlefilterbox.

Provides the `pfbox` CLI tool for running particle filters, PMCMC estimation,
model comparison, and data simulation from the terminal.

Usage
-----
    $ pfbox filter data.csv --model sv --n-particles 1000
    $ pfbox estimate data.csv --model sv --method pmmh --n-iterations 5000
    $ pfbox compare data.csv --models sv,dsge --n-particles 2000
    $ pfbox simulate --model sv --n-obs 500 --seed 42
"""

from __future__ import annotations
