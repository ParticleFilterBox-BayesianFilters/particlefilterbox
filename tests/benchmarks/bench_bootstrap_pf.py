"""Benchmark: Bootstrap Particle Filter performance.

Measures execution time for different particle counts and time series lengths.
Run with: pytest tests/benchmarks/bench_bootstrap_pf.py -v --benchmark-only
(or simply: python tests/benchmarks/bench_bootstrap_pf.py)
"""

from __future__ import annotations

import time

import numpy as np


def benchmark_bootstrap_pf(
    T: int = 200,
    N: int = 1000,
    n_runs: int = 3,
) -> dict[str, float]:
    """Benchmark bootstrap PF for given T and N.

    Parameters
    ----------
    T : int
        Time series length.
    N : int
        Number of particles.
    n_runs : int
        Number of runs to average.

    Returns
    -------
    dict with 'mean_time', 'std_time', 'particles_per_second'.
    """
    try:
        from particlefilterbox.models.linear_gaussian import LinearGaussianModel

        from particlefilterbox.filters.bootstrap import BootstrapFilter

        rng = np.random.default_rng(42)
        y = rng.standard_normal(T)
        model = LinearGaussianModel()

        times = []
        for i in range(n_runs):
            run_rng = np.random.default_rng(42 + i)
            pf = BootstrapFilter(model=model, n_particles=N, rng=run_rng)

            t0 = time.perf_counter()
            pf.filter(y)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)
        pps = T * N / mean_time

        return {
            "T": T,
            "N": N,
            "mean_time": float(mean_time),
            "std_time": float(std_time),
            "particles_per_second": float(pps),
        }

    except ImportError:
        return {"error": "BootstrapFilter not available"}


def benchmark_resampling(
    N: int = 10000,
    n_runs: int = 10,
) -> dict[str, dict[str, float]]:
    """Benchmark resampling algorithms.

    Parameters
    ----------
    N : int
        Number of particles.
    n_runs : int
        Number of runs.

    Returns
    -------
    dict mapping algorithm name to timing results.
    """
    try:
        from particlefilterbox.core.resampling import (
            multinomial_resampling,
            residual_resampling,
            stratified_resampling,
            systematic_resampling,
        )

        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(N))

        results: dict[str, dict[str, float]] = {}
        algorithms = {
            "multinomial": multinomial_resampling,
            "systematic": systematic_resampling,
            "stratified": stratified_resampling,
            "residual": residual_resampling,
        }

        for name, fn in algorithms.items():
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                fn(weights, rng)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)

            results[name] = {
                "mean_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
            }

        return results

    except ImportError:
        return {"error": {"mean_time": 0.0, "std_time": 0.0}}


if __name__ == "__main__":
    print("=" * 60)
    print("Bootstrap PF Benchmarks")
    print("=" * 60)

    configs = [
        (100, 500),
        (100, 1000),
        (100, 5000),
        (500, 1000),
        (1000, 1000),
    ]

    for T, N in configs:
        result = benchmark_bootstrap_pf(T=T, N=N)
        if "error" not in result:
            print(
                f"  T={T:>5}, N={N:>5}: "
                f"{result['mean_time']:.4f}s +/- {result['std_time']:.4f}s "
                f"({result['particles_per_second']:.0f} particles/s)"
            )
        else:
            print(f"  T={T}, N={N}: {result['error']}")

    print()
    print("Resampling Benchmarks (N=10000)")
    print("-" * 40)
    resampling_results = benchmark_resampling()
    for name, timing in resampling_results.items():
        if isinstance(timing, dict) and "mean_time" in timing:
            print(f"  {name:<15}: {timing['mean_time']*1000:.4f}ms")
