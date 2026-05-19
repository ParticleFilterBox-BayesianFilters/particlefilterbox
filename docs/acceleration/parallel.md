# Parallel Execution

## Overview

particlefilterbox supports **parallel execution** at multiple levels --- from running independent filter replicas across CPU cores to distributing large-scale experiments over Dask clusters. Parallelism is orthogonal to the backend choice: you can combine parallel execution with Numba or GPU backends for maximum throughput.

```python
from particlefilterbox import ParallelRunner

runner = ParallelRunner(n_jobs=4)
results = runner.run_filters(filters, observations)
```

---

## Parallelism Levels

| Level | Mechanism | Use Case | Overhead |
|-------|-----------|----------|----------|
| **Independent replicas** | `multiprocessing` | Monte Carlo repeats, bootstrap CI | Low |
| **Experiment sweeps** | `multiprocessing` / Dask | Grid search over $N$, parameters | Low--Medium |
| **SMC² theta particles** | `multiprocessing` | Parallel likelihood evaluation per $\theta$ | Medium |
| **Two-Filter smoother** | `multiprocessing` | Forward + backward filters in parallel | Low |
| **I/O-bound tasks** | `threading` | Loading data, writing results | Very low |
| **Cluster-scale** | Dask | Thousands of runs across machines | Higher |

---

## ParallelRunner API

### Basic usage

```python
from particlefilterbox import BootstrapFilter, ParallelRunner

model = StochasticVolatility(mu=0, phi=0.95, sigma=0.2)

# Create a runner with 4 workers
runner = ParallelRunner(n_jobs=4)

# Run 100 independent filter replications
filters = [
    BootstrapFilter(model, n_particles=5000, backend='numba', seed=i)
    for i in range(100)
]
results = runner.run_filters(filters, observations)

# results is a list of FilterResult objects
mean_ll = np.mean([r.log_likelihood for r in results])
```

### Parameter sweeps

```python
from particlefilterbox import ParallelRunner, BootstrapFilter

runner = ParallelRunner(n_jobs=8)

# Sweep over particle counts
n_values = [500, 1000, 2000, 5000, 10000]
filters = [
    BootstrapFilter(model, n_particles=n, backend='numba')
    for n in n_values
]
results = runner.run_filters(filters, observations)

for n, r in zip(n_values, results):
    print(f"N={n:>6d}  log-lik={r.log_likelihood:.2f}  ESS_mean={r.ess.mean():.0f}")
```

---

## Multiprocessing for Independent Replicas

The most common parallel pattern is running **independent Monte Carlo replicas** for variance estimation or confidence intervals.

```python
from particlefilterbox import BootstrapFilter, ParallelRunner
import numpy as np

runner = ParallelRunner(n_jobs=-1)  # Use all available cores

# Run 200 replications for log-likelihood variance estimation
filters = [
    BootstrapFilter(model, n_particles=5000, backend='numba', seed=i)
    for i in range(200)
]
results = runner.run_filters(filters, observations)

log_liks = np.array([r.log_likelihood for r in results])
print(f"log p(y) = {log_liks.mean():.2f} ± {log_liks.std():.2f}")
```

!!! tip "Seed management"
    Always set distinct seeds for each replica to ensure independent random streams. particlefilterbox's `seed` parameter ensures reproducible results even under parallel execution.

---

## Parallelising SMC²

In SMC², each $\theta$-particle requires running a **full particle filter** to evaluate $p(y_{1:t} \mid \theta)$. These evaluations are independent and embarrassingly parallel:

```python
from particlefilterbox import SMC2

smc2 = SMC2(
    model_class=StochasticVolatility,
    prior=prior,
    n_theta=500,          # 500 theta particles
    n_particles=1000,     # 1000 x-particles per theta
    backend='numba',
    n_jobs=8,             # Parallelise over 8 cores
)
result = smc2.fit(observations)
```

### Scaling behaviour

| $N_\theta$ | $N_x$ | Sequential (s) | 4 cores (s) | 8 cores (s) | 16 cores (s) |
|:----------:|:-----:|:--------------:|:-----------:|:-----------:|:------------:|
| 100 | 500 | 60 | 16 | 9 | 5.5 |
| 500 | 1,000 | 1,500 | 390 | 205 | 115 |
| 1,000 | 2,000 | 12,000 | 3,100 | 1,650 | 900 |

!!! info "Near-linear scaling"
    SMC² parallelism scales nearly linearly with the number of cores because the per-$\theta$ filter runs are fully independent. Overhead comes primarily from the resampling step, which requires synchronisation across all $\theta$-particles.

---

## Parallelising the Two-Filter Smoother

The Two-Filter smoother runs a **forward filter** and a **backward information filter** independently, then combines them. These two passes can run in parallel:

```python
from particlefilterbox import TwoFilterSmoother

smoother = TwoFilterSmoother(
    model,
    n_particles=10000,
    backend='numba',
    parallel_passes=True,   # Run forward and backward in parallel
)
result = smoother.smooth(observations)
```

This yields a **~1.8× speedup** (not exactly 2× due to the combining step).

---

## Threading for I/O-Bound Tasks

For tasks dominated by I/O (loading data files, writing results, fetching observations from a database), use threading:

```python
from particlefilterbox import ParallelRunner

runner = ParallelRunner(n_jobs=4, backend='threading')

# Load and filter multiple datasets in parallel
datasets = ['data_2020.csv', 'data_2021.csv', 'data_2022.csv', 'data_2023.csv']
results = runner.run_on_datasets(
    filter_factory=lambda: BootstrapFilter(model, n_particles=5000),
    datasets=datasets,
)
```

!!! warning "GIL limitation"
    Python's Global Interpreter Lock (GIL) prevents true parallel CPU execution with threads. Use `multiprocessing` (the default) for CPU-bound particle filtering. Threading is only beneficial for I/O-bound workloads.

---

## Dask Integration for Clusters

For large-scale experiments spanning multiple machines, particlefilterbox integrates with [Dask](https://dask.org/):

### Local Dask cluster

```python
from dask.distributed import Client
from particlefilterbox import DaskRunner

# Start a local Dask cluster
client = Client(n_workers=8, threads_per_worker=1)

runner = DaskRunner(client=client)

# Run 1000 replications distributed across workers
filters = [
    BootstrapFilter(model, n_particles=5000, backend='numba', seed=i)
    for i in range(1000)
]
results = runner.run_filters(filters, observations)
```

### Remote Dask cluster

```python
from dask.distributed import Client
from particlefilterbox import DaskRunner

# Connect to an existing Dask scheduler
client = Client('scheduler-address:8786')
print(f"Connected to {len(client.scheduler_info()['workers'])} workers")

runner = DaskRunner(client=client)
results = runner.run_filters(filters, observations)
```

### Dask experiment grid

```python
from particlefilterbox import DaskRunner, experiment_grid

runner = DaskRunner(client=client)

# Define a parameter grid
grid = experiment_grid(
    model_class=StochasticVolatility,
    n_particles=[1000, 5000, 10000],
    backend=['python', 'numba'],
    phi=[0.90, 0.95, 0.99],
    seeds=range(50),  # 50 replications per configuration
)

# Submit all combinations to the cluster
results = runner.run_grid(grid, observations)
# Returns a DataFrame with columns: n_particles, backend, phi, seed, log_lik, ess_mean, time
```

---

## Benchmarks

### Multiprocessing scaling (100 replicas, $N = 5\,000$, Numba)

| Workers | Wall Time (s) | Speedup | Efficiency |
|:-------:|:------------:|:-------:|:----------:|
| 1 | 48.0 | 1.0× | 100% |
| 2 | 24.5 | 2.0× | 98% |
| 4 | 12.8 | 3.8× | 94% |
| 8 | 6.9 | 7.0× | 87% |
| 16 | 4.2 | 11.4× | 71% |

### Dask cluster scaling (1000 replicas, $N = 5\,000$, Numba)

| Workers | Wall Time (s) | Speedup | Efficiency |
|:-------:|:------------:|:-------:|:----------:|
| 8 | 62.0 | 1.0× | 100% |
| 16 | 32.5 | 1.9× | 95% |
| 32 | 17.8 | 3.5× | 87% |
| 64 | 10.5 | 5.9× | 74% |

!!! info "Efficiency drop-off"
    Parallel efficiency decreases at high core counts due to OS scheduling overhead, memory bandwidth contention, and (for Dask) network serialisation costs. For CPU-bound particle filtering, **8--16 cores** is typically the sweet spot on a single machine.

---

## Combining Parallel + GPU

Parallel execution and GPU backends can be combined for maximum throughput. A common pattern is running **multiple GPU-accelerated filters** across several GPUs:

```python
from particlefilterbox import ParallelRunner, BootstrapFilter

# Each replica on a different GPU
filters = [
    BootstrapFilter(model, n_particles=100000, backend='cupy',
                    gpu_devices=[i % 4], seed=i)
    for i in range(20)
]

runner = ParallelRunner(n_jobs=4)  # 4 processes, one per GPU
results = runner.run_filters(filters, observations)
```

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_jobs` | `int` | `1` | Number of parallel workers. `-1` = all cores |
| `backend` | `str` | `'multiprocessing'` | `'multiprocessing'` or `'threading'` |
| `verbose` | `int` | `0` | Verbosity level (0=silent, 1=progress, 2=debug) |
| `timeout` | `float` | `None` | Per-task timeout in seconds |
| `batch_size` | `int` | `'auto'` | Number of tasks per batch dispatch |

```python
runner = ParallelRunner(
    n_jobs=8,
    backend='multiprocessing',
    verbose=1,          # Show progress bar
    timeout=300,        # 5-minute timeout per task
)
```

---

## See Also

- [Acceleration Overview](index.md) --- Backend comparison
- [Numba JIT](numba.md) --- Combine with parallel for CPU-level speedups
- [GPU Acceleration](gpu.md) --- Combine with parallel for multi-GPU
- [Adaptive N](adaptive-n.md) --- Reduce cost per replica
- [Convergence Diagnostic](../diagnostics/convergence.md) --- run parallel N-studies efficiently
- [Filter Comparison](../diagnostics/filter-comparison.md) --- parallelise multi-filter comparison experiments
- [SMC²](../user-guide/smc/smc-squared.md) --- embarrassingly parallel $\theta$-particle evaluations
- [Experiment Framework](../user-guide/experiment.md) --- built-in support for parallel experiment sweeps
- [PMMH](../user-guide/pmcmc/pmmh.md) --- run multiple MCMC chains in parallel for convergence diagnostics
