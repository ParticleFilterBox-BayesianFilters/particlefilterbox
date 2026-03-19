"""Benchmark: Resampling algorithms.

Measures execution time for different resampling methods and particle counts.
"""

from __future__ import annotations

import time

import numpy as np


def run_resampling_benchmarks() -> None:
    """Run and print resampling benchmarks."""
    try:
        from particlefilterbox.core.resampling import (
            multinomial_resampling,
            residual_resampling,
            stratified_resampling,
            systematic_resampling,
        )
    except ImportError:
        print("Resampling module not available")
        return

    algorithms = {
        "multinomial": multinomial_resampling,
        "systematic": systematic_resampling,
        "stratified": stratified_resampling,
        "residual": residual_resampling,
    }

    particle_counts = [100, 1000, 10000, 100000]
    n_runs = 20

    print("=" * 70)
    print("Resampling Algorithm Benchmarks")
    print("=" * 70)
    print(f"{'Algorithm':<15} ", end="")
    for N in particle_counts:
        print(f"{'N=' + str(N):>12}", end="")
    print()
    print("-" * 70)

    rng = np.random.default_rng(42)

    for name, fn in algorithms.items():
        print(f"{name:<15} ", end="")
        for N in particle_counts:
            weights = rng.dirichlet(np.ones(N))
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                fn(weights, rng)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
            mean_ms = np.mean(times) * 1000
            print(f"{mean_ms:>10.3f}ms", end="")
        print()

    print("=" * 70)


if __name__ == "__main__":
    run_resampling_benchmarks()
