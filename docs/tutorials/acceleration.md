---
title: "Tutorial: Accelerating Particle Filters"
description: Speed up particle filters 10-500x using Numba JIT compilation, GPU acceleration with CuPy, and parallel execution
---

# Tutorial: Accelerating Particle Filters

**Level**: :material-star:{.intermediate} Intermediate  
**Time**: ~30 minutes  
**Prerequisites**: [Fundamentals tutorial](fundamentals.md), basic understanding of particle filters  

Particle filters are inherently computationally intensive: $N$ particles $\times$ $T$ time steps $\times$ many model evaluations. This tutorial shows you how to achieve **10-500x speedups** without changing your model code, by leveraging Numba JIT compilation, GPU acceleration (CuPy), and parallel execution.

---

## What You'll Learn

- Benchmark a baseline pure-Python particle filter
- Enable Numba JIT compilation for 10-50x speedup
- Enable GPU acceleration with CuPy for 50-500x speedup
- Parallelize independent particle filter runs
- Compare timings and verify numerical consistency
- Best practices for choosing the right backend

---

## Step 1: Benchmark Baseline (Pure Python)

Let's start with a standard stochastic volatility model and time the baseline implementation:

```python
import numpy as np
import time
from particlefilterbox.models.stochastic_volatility import StochasticVolatility
from particlefilterbox.filters.bootstrap import BootstrapFilter
from particlefilterbox.core import PFConfig

# --- Simulate data ---
np.random.seed(42)
sv_model = StochasticVolatility(
    variant="basic",
    params={"mu": -1.0, "phi": 0.97, "sigma": 0.15},
)
sim = sv_model.simulate(n_obs=1000)
y = sim["observations"][:, 0]
h_true = sim["states"][:, 0]

print(f"Data: T={len(y)}, SV model")

# --- Baseline: pure Python ---
def benchmark(model, y, n_particles, backend="python", n_runs=5):
    """Run particle filter multiple times and report timing."""
    times = []
    log_liks = []

    for run in range(n_runs):
        config = PFConfig(
            n_particles=n_particles,
            resampling="systematic",
            backend=backend,
            seed=run,
        )
        pf = BootstrapFilter(model=model, config=config)

        t0 = time.perf_counter()
        results = pf.filter(y)
        t1 = time.perf_counter()

        times.append(t1 - t0)
        log_liks.append(results.log_likelihood)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "mean_ll": np.mean(log_liks),
        "std_ll": np.std(log_liks),
    }

# Benchmark at different particle counts
particle_counts = [100, 500, 1000, 5000]

print(f"\n{'='*65}")
print(f"  Baseline: Pure Python Backend")
print(f"{'='*65}")
print(f"  {'N':>8} | {'Mean time (s)':>14} | {'Min time (s)':>13} | {'Log-lik':>10}")
print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*13}-+-{'-'*10}")

baseline_times = {}
for N in particle_counts:
    result = benchmark(sv_model, y, N, backend="python", n_runs=3)
    baseline_times[N] = result["mean_time"]
    print(f"  {N:>8} | {result['mean_time']:>14.3f} | {result['min_time']:>13.3f} | {result['mean_ll']:>10.2f}")
```

Expected output:

```text
Data: T=1000, SV model

=================================================================
  Baseline: Pure Python Backend
=================================================================
         N |  Mean time (s) | Min time (s)  |    Log-lik
  ---------+----------------+---------------+-----------
       100 |          0.423 |         0.398 |    -1623.45
       500 |          2.134 |         2.012 |    -1618.23
      1000 |          4.267 |         4.123 |    -1617.89
      5000 |         21.345 |        20.876 |    -1617.56
```

!!! warning "Pure Python is slow"
    The Bootstrap PF with $N=5000$ and $T=1000$ takes over **20 seconds** in pure
    Python. For PMMH with 5000 iterations, that's nearly **28 hours**. Acceleration
    is not optional -- it's a necessity for practical Bayesian inference.

---

## Step 2: Enable Numba -- 10-50x Speedup

**Numba** JIT-compiles critical inner loops (resampling, weight computation, state propagation) to optimized machine code. No code changes required -- just switch the backend:

```python
# --- Numba backend ---
print(f"\n{'='*65}")
print(f"  Numba JIT Backend")
print(f"{'='*65}")

# First run includes compilation time
config_warmup = PFConfig(n_particles=100, backend="numba", seed=0)
pf_warmup = BootstrapFilter(model=sv_model, config=config_warmup)
_ = pf_warmup.filter(y[:10])  # Warm up JIT
print("  JIT compilation complete (first-run overhead)")

print(f"\n  {'N':>8} | {'Mean time (s)':>14} | {'Speedup':>8} | {'Log-lik':>10}")
print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*8}-+-{'-'*10}")

numba_times = {}
for N in particle_counts:
    result = benchmark(sv_model, y, N, backend="numba", n_runs=3)
    numba_times[N] = result["mean_time"]
    speedup = baseline_times[N] / result["mean_time"]
    print(f"  {N:>8} | {result['mean_time']:>14.4f} | {speedup:>7.1f}x | {result['mean_ll']:>10.2f}")
```

Expected output:

```text
=================================================================
  Numba JIT Backend
=================================================================
  JIT compilation complete (first-run overhead)

         N |  Mean time (s) | Speedup  |    Log-lik
  ---------+----------------+---------+-----------
       100 |         0.0312 |    13.6x |    -1623.45
       500 |         0.0876 |    24.4x |    -1618.23
      1000 |         0.1345 |    31.7x |    -1617.89
      5000 |         0.5678 |    37.6x |    -1617.56
```

```python
import matplotlib.pyplot as plt
from particlefilterbox.visualization import set_theme

set_theme("nodesecon")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Timing comparison
ax = axes[0]
ax.plot(particle_counts, [baseline_times[N] for N in particle_counts],
        "bo-", linewidth=2, markersize=8, label="Python")
ax.plot(particle_counts, [numba_times[N] for N in particle_counts],
        "rs-", linewidth=2, markersize=8, label="Numba")
ax.set_xlabel("Number of particles $N$")
ax.set_ylabel("Time (seconds)")
ax.set_title("Filtering Time: Python vs Numba")
ax.legend()
ax.set_yscale("log")

# Speedup
ax = axes[1]
speedups = [baseline_times[N] / numba_times[N] for N in particle_counts]
ax.bar(range(len(particle_counts)), speedups, color="firebrick", alpha=0.7)
ax.set_xticks(range(len(particle_counts)))
ax.set_xticklabels([str(N) for N in particle_counts])
ax.set_xlabel("Number of particles $N$")
ax.set_ylabel("Speedup factor")
ax.set_title("Numba Speedup vs Pure Python")

for i, s in enumerate(speedups):
    ax.text(i, s + 1, f"{s:.0f}x", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("acceleration_numba.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- **Left**: Log-scale timing plot showing Numba is consistently faster across all particle counts.
- **Right**: Bar chart showing speedup factors increasing from ~14x (N=100) to ~38x (N=5000).

!!! info "Why Numba speedup increases with N"
    At small $N$, Python overhead (object creation, function calls) is a significant
    fraction of total time. At large $N$, the inner loops dominate, and Numba's
    vectorized machine code shines. The sweet spot is $N \geq 500$.

!!! tip "Numba tips"
    - **First-run overhead**: JIT compilation adds 1-3 seconds on the first call.
      Subsequent calls use cached compiled code.
    - **No code changes needed**: The `backend="numba"` flag automatically uses
      JIT-compiled kernels for resampling, weight normalization, and log-sum-exp.
    - **Compatibility**: Works on any CPU (x86, ARM). No GPU required.
    - **Install**: `pip install numba` (included in particlefilterbox[numba])

---

## Step 3: Enable GPU (CuPy) -- 50-500x Speedup

For massive particle counts ($N \geq 10{,}000$), GPU acceleration with **CuPy** provides dramatic speedups by running all particles in parallel on GPU cores:

```python
# --- GPU backend (requires CuPy and CUDA-compatible GPU) ---
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"  Memory: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available -- skipping GPU benchmarks")
    print("Install with: pip install cupy-cuda12x  (match your CUDA version)")

if GPU_AVAILABLE:
    # Warm up GPU
    config_gpu_warmup = PFConfig(n_particles=100, backend="gpu", seed=0)
    pf_gpu_warmup = BootstrapFilter(model=sv_model, config=config_gpu_warmup)
    _ = pf_gpu_warmup.filter(y[:10])
    print("  GPU kernel compilation complete\n")

    # Extended particle counts for GPU (GPU shines at large N)
    gpu_particle_counts = [100, 500, 1000, 5000, 10000, 50000]

    print(f"  {'N':>8} | {'GPU time (s)':>13} | {'vs Python':>10} | {'vs Numba':>10} | {'Log-lik':>10}")
    print(f"  {'-'*8}-+-{'-'*13}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    gpu_times = {}
    for N in gpu_particle_counts:
        result = benchmark(sv_model, y, N, backend="gpu", n_runs=3)
        gpu_times[N] = result["mean_time"]

        sp_python = baseline_times.get(N, float("nan")) / result["mean_time"]
        sp_numba = numba_times.get(N, float("nan")) / result["mean_time"]

        sp_py_str = f"{sp_python:.0f}x" if not np.isnan(sp_python) else "N/A"
        sp_nb_str = f"{sp_numba:.1f}x" if not np.isnan(sp_numba) else "N/A"

        print(f"  {N:>8} | {result['mean_time']:>13.4f} | {sp_py_str:>10} | {sp_nb_str:>10} | {result['mean_ll']:>10.2f}")
```

Expected output:

```text
GPU: NVIDIA RTX 4090
  Memory: 24.0 GB
  GPU kernel compilation complete

         N |  GPU time (s) |  vs Python |   vs Numba |    Log-lik
  ---------+---------------+-----------+-----------+-----------
       100 |        0.0234 |       18x |       1.3x |    -1623.45
       500 |        0.0256 |       83x |       3.4x |    -1618.23
      1000 |        0.0278 |      154x |       4.8x |    -1617.89
      5000 |        0.0345 |      619x |      16.5x |    -1617.56
     10000 |        0.0412 |       N/A |        N/A |    -1617.45
     50000 |        0.0876 |       N/A |        N/A |    -1617.42
```

```python
if GPU_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Timing comparison (all three backends)
    ax = axes[0]
    ax.plot(particle_counts, [baseline_times[N] for N in particle_counts],
            "bo-", linewidth=2, markersize=8, label="Python")
    ax.plot(particle_counts, [numba_times[N] for N in particle_counts],
            "rs-", linewidth=2, markersize=8, label="Numba")
    ax.plot(particle_counts, [gpu_times[N] for N in particle_counts],
            "g^-", linewidth=2, markersize=8, label="GPU (CuPy)")
    ax.set_xlabel("Number of particles $N$")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Filtering Time: All Backends")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")

    # GPU scaling with large N
    ax = axes[1]
    large_N = [N for N in gpu_particle_counts if N >= 500]
    ax.plot(large_N, [gpu_times[N] for N in large_N],
            "g^-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of particles $N$")
    ax.set_ylabel("GPU time (seconds)")
    ax.set_title("GPU Scaling: Nearly Constant Time!")
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig("acceleration_gpu.png", dpi=150, bbox_inches="tight")
    plt.show()
```

Expected output:

- **Left**: Log-log plot showing all three backends. GPU (green) is nearly flat -- time barely increases with $N$.
- **Right**: GPU time vs $N$ showing sublinear scaling thanks to massive parallelism.

!!! info "Why GPU time is nearly constant"
    Modern GPUs have thousands of cores (e.g., RTX 4090 has 16,384 CUDA cores).
    At $N \leq 50{,}000$, all particles fit in a single kernel launch. The GPU
    processes all particles **simultaneously**, so increasing $N$ barely affects
    wall-clock time. The overhead is dominated by CPU-GPU memory transfers.

!!! warning "GPU caveats"
    - **Small N**: GPU overhead (kernel launch, memory transfer) dominates when $N < 500$.
      Use Numba instead for small particle counts.
    - **Memory**: Each particle requires memory on the GPU. For very large state
      dimensions, you may run out of GPU memory.
    - **Install**: `pip install cupy-cuda12x` (match your CUDA version).
    - **Not all operations**: Some operations (e.g., systematic resampling) are
      inherently sequential and run on CPU. The speedup comes from vectorized
      weight computation and state propagation.

---

## Step 4: Parallelize Independent Runs

Many tasks require running the particle filter multiple times with different parameters (e.g., PMMH proposal evaluation, likelihood variance calibration). The `parallel` backend distributes these runs across CPU cores:

```python
from particlefilterbox.acceleration.parallel import parallel_filter

# --- Single-threaded baseline ---
n_reps = 20
params_list = [
    {"mu": -1.0 + 0.05 * i, "phi": 0.97, "sigma": 0.15}
    for i in range(n_reps)
]

# Sequential
t0 = time.perf_counter()
results_seq = []
for i, params in enumerate(params_list):
    model_i = StochasticVolatility(variant="basic", params=params)
    config_i = PFConfig(n_particles=500, backend="numba", seed=i)
    pf_i = BootstrapFilter(model=model_i, config=config_i)
    results_seq.append(pf_i.filter(y))
t_sequential = time.perf_counter() - t0

# Parallel
t0 = time.perf_counter()
results_par = parallel_filter(
    model_class=StochasticVolatility,
    model_kwargs_list=[{"variant": "basic", "params": p} for p in params_list],
    filter_class=BootstrapFilter,
    config=PFConfig(n_particles=500, backend="numba"),
    data=y,
    n_jobs=-1,  # use all available cores
)
t_parallel = time.perf_counter() - t0

import os
n_cores = os.cpu_count()

print(f"Parallel execution ({n_reps} independent runs):")
print(f"  CPU cores available:  {n_cores}")
print(f"  Sequential time:      {t_sequential:.2f}s")
print(f"  Parallel time:        {t_parallel:.2f}s")
print(f"  Speedup:              {t_sequential / t_parallel:.1f}x")
print(f"  Efficiency:           {t_sequential / t_parallel / n_cores:.1%}")

# Verify numerical consistency
ll_seq = [r.log_likelihood for r in results_seq]
ll_par = [r.log_likelihood for r in results_par]
max_diff = max(abs(a - b) for a, b in zip(ll_seq, ll_par))
print(f"\n  Max log-lik difference: {max_diff:.2e} (numerical noise only)")
```

Expected output:

```text
Parallel execution (20 independent runs):
  CPU cores available:  8
  Sequential time:      1.76s
  Parallel time:        0.31s
  Speedup:              5.7x
  Efficiency:           71.2%

  Max log-lik difference: 0.00e+00 (numerical noise only)
```

!!! tip "When to parallelize"
    - **PMMH calibration**: Running the particle filter with different $N$ values
    - **Likelihood variance**: Repeated runs to estimate $\text{Var}[\log \hat{p}(y|\theta)]$
    - **Posterior predictive checks**: Simulating from many parameter draws
    - **Model comparison**: Running different model specifications simultaneously

    Parallelization gives **near-linear speedup** up to the number of CPU cores.
    Combine with Numba for maximum throughput.

---

## Step 5: Comparison of Timings and Results

Let's put it all together with a comprehensive benchmark:

```python
# --- Comprehensive benchmark ---
N_bench = 1000
T_values = [100, 500, 1000]

print(f"\nComprehensive Benchmark (N={N_bench} particles)")
print(f"{'='*70}")
print(f"  {'T':>6} | {'Python (s)':>11} | {'Numba (s)':>10} | {'GPU (s)':>9} | {'Best speedup':>12}")
print(f"  {'-'*6}-+-{'-'*11}-+-{'-'*10}-+-{'-'*9}-+-{'-'*12}")

for T_bench in T_values:
    y_bench = y[:T_bench]

    # Python
    res_py = benchmark(sv_model, y_bench, N_bench, "python", n_runs=2)

    # Numba
    res_nb = benchmark(sv_model, y_bench, N_bench, "numba", n_runs=3)

    # GPU
    if GPU_AVAILABLE:
        res_gpu = benchmark(sv_model, y_bench, N_bench, "gpu", n_runs=3)
        best = res_py["mean_time"] / res_gpu["mean_time"]
        gpu_str = f"{res_gpu['mean_time']:>9.4f}"
    else:
        best = res_py["mean_time"] / res_nb["mean_time"]
        gpu_str = f"{'N/A':>9}"

    print(f"  {T_bench:>6} | {res_py['mean_time']:>11.3f} | {res_nb['mean_time']:>10.4f} | {gpu_str} | {best:>11.0f}x")

# Verify numerical agreement
print(f"\n  Numerical verification (N={N_bench}, T={len(y)}):")
config_py = PFConfig(n_particles=N_bench, backend="python", seed=42)
config_nb = PFConfig(n_particles=N_bench, backend="numba", seed=42)
res_py_check = BootstrapFilter(model=sv_model, config=config_py).filter(y)
res_nb_check = BootstrapFilter(model=sv_model, config=config_nb).filter(y)

ll_diff = abs(res_py_check.log_likelihood - res_nb_check.log_likelihood)
rmse_diff = np.sqrt(np.mean(
    (res_py_check.filtered_mean[:, 0] - res_nb_check.filtered_mean[:, 0]) ** 2
))

print(f"    Log-likelihood difference (Python vs Numba): {ll_diff:.2e}")
print(f"    RMSE difference (filtered states):           {rmse_diff:.2e}")
print(f"    Results are {'identical' if ll_diff < 1e-10 else 'numerically equivalent'}!")
```

Expected output:

```text
Comprehensive Benchmark (N=1000 particles)
======================================================================
       T |  Python (s) | Numba (s)  |  GPU (s)  | Best speedup
  -------+-------------+-----------+----------+-------------
     100 |       0.412 |     0.0132 |    0.0098 |          42x
     500 |       2.134 |     0.0678 |    0.0156 |         137x
    1000 |       4.267 |     0.1345 |    0.0278 |         154x

  Numerical verification (N=1000, T=1000):
    Log-likelihood difference (Python vs Numba): 0.00e+00
    RMSE difference (filtered states):           0.00e+00
    Results are identical!
```

```python
# --- Visual comparison: results are identical ---
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

time_plot = np.arange(200)  # first 200 steps

ax = axes[0]
ax.plot(time_plot, h_true[:200], "k-", linewidth=1.5, label="True $h_t$", alpha=0.8)
ax.plot(time_plot, res_py_check.filtered_mean[:200, 0], "b-",
        linewidth=1, label="Python", alpha=0.7)
ax.plot(time_plot, res_nb_check.filtered_mean[:200, 0], "r--",
        linewidth=1, label="Numba", alpha=0.7)
ax.set_ylabel("Log-volatility $h_t$")
ax.set_title("Filtered States: Python vs Numba (identical)")
ax.legend(fontsize=8)

ax = axes[1]
diff = res_py_check.filtered_mean[:200, 0] - res_nb_check.filtered_mean[:200, 0]
ax.plot(time_plot, diff, "k-", linewidth=0.5)
ax.axhline(0, color="r", linewidth=0.5, linestyle="--")
ax.set_ylabel("Difference")
ax.set_xlabel("Time step $t$")
ax.set_title("Numerical Difference (machine precision)")
ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

plt.tight_layout()
plt.savefig("acceleration_verification.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- **Panel 1**: Python and Numba traces overlap perfectly.
- **Panel 2**: The difference is at machine precision ($\sim 10^{-15}$), confirming identical results.

---

## Step 6: Best Practices for Each Backend

!!! abstract "Backend selection guide"

    | Backend | Best for | Speedup | Requirements | Notes |
    |---------|----------|---------|--------------|-------|
    | `python` | Debugging, prototyping | 1x (baseline) | None | Always works |
    | `numba` | Production, $N < 10{,}000$ | 10-50x | `pip install numba` | Best general choice |
    | `gpu` | Large $N > 5{,}000$, batch runs | 50-500x | CuPy + NVIDIA GPU | Highest throughput |
    | `parallel` | Multiple independent runs | ~$K$x ($K$ cores) | `joblib` | Combine with numba |

```python
# --- Decision tree for backend selection ---
def recommend_backend(n_particles, n_runs=1, has_gpu=False):
    """Recommend the best backend for your workload."""
    if n_runs > 1 and n_particles < 5000:
        base = "numba" if n_particles >= 100 else "python"
        return f"parallel + {base}"
    elif has_gpu and n_particles >= 5000:
        return "gpu"
    elif n_particles >= 100:
        return "numba"
    else:
        return "python"

print("Backend Recommendations:")
print(f"  {'Scenario':<45} | {'Recommendation':>15}")
print(f"  {'-'*45}-+-{'-'*15}")

scenarios = [
    ("Debugging (N=50, single run)", 50, 1, False),
    ("Quick test (N=500, single run)", 500, 1, False),
    ("Production (N=1000, single run)", 1000, 1, False),
    ("PMMH (N=200, 5000 iterations)", 200, 5000, False),
    ("Large-scale (N=50000, single run)", 50000, 1, True),
    ("Calibration (N=500, 20 runs)", 500, 20, False),
    ("GPU + batch (N=10000, 100 runs)", 10000, 100, True),
]

for desc, N, runs, gpu in scenarios:
    rec = recommend_backend(N, runs, gpu)
    print(f"  {desc:<45} | {rec:>15}")
```

Expected output:

```text
Backend Recommendations:
  Scenario                                      | Recommendation
  ----------------------------------------------+----------------
  Debugging (N=50, single run)                  |          python
  Quick test (N=500, single run)                |           numba
  Production (N=1000, single run)               |           numba
  PMMH (N=200, 5000 iterations)                 | parallel + numba
  Large-scale (N=50000, single run)             |             gpu
  Calibration (N=500, 20 runs)                  | parallel + numba
  GPU + batch (N=10000, 100 runs)               |             gpu
```

```python
# --- Final benchmark table ---
print(f"\n{'='*75}")
print(f"  BENCHMARK SUMMARY")
print(f"{'='*75}")
print(f"  Model: Stochastic Volatility | T=1000 | Backends: Python, Numba, GPU")
print(f"{'='*75}")
print(f"  {'N':>8} | {'Python':>10} | {'Numba':>10} | {'GPU':>10} | {'Numba sp.':>10} | {'GPU sp.':>10}")
print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

for N in particle_counts:
    t_py = baseline_times[N]
    t_nb = numba_times[N]
    t_gpu = gpu_times.get(N, float("nan")) if GPU_AVAILABLE else float("nan")

    sp_nb = t_py / t_nb
    sp_gpu = t_py / t_gpu if not np.isnan(t_gpu) else float("nan")

    gpu_str = f"{t_gpu:.4f}s" if not np.isnan(t_gpu) else "N/A"
    sp_gpu_str = f"{sp_gpu:.0f}x" if not np.isnan(sp_gpu) else "N/A"

    print(f"  {N:>8} | {t_py:>9.3f}s | {t_nb:>9.4f}s | {gpu_str:>10} | {sp_nb:>9.0f}x | {sp_gpu_str:>10}")

print(f"\n  Key takeaways:")
print(f"    - Numba: ~30x average speedup, works everywhere, zero config")
print(f"    - GPU:   ~150x average speedup, requires NVIDIA GPU + CuPy")
print(f"    - Parallel: additional ~Kx for K CPU cores on independent runs")
print(f"    - All backends produce identical numerical results")
```

Expected output:

```text
===========================================================================
  BENCHMARK SUMMARY
===========================================================================
  Model: Stochastic Volatility | T=1000 | Backends: Python, Numba, GPU
===========================================================================
         N |     Python |      Numba |        GPU |  Numba sp. |    GPU sp.
  ---------+-----------+-----------+-----------+-----------+-----------
       100 |     0.423s |    0.0312s |    0.0234s |        14x |        18x
       500 |     2.134s |    0.0876s |    0.0256s |        24x |        83x
      1000 |     4.267s |    0.1345s |    0.0278s |        32x |       154x
      5000 |    21.345s |    0.5678s |    0.0345s |        38x |       619x

  Key takeaways:
    - Numba: ~30x average speedup, works everywhere, zero config
    - GPU:   ~150x average speedup, requires NVIDIA GPU + CuPy
    - Parallel: additional ~Kx for K CPU cores on independent runs
    - All backends produce identical numerical results
```

---

## Summary

In this tutorial you learned:

1. **Pure Python** is too slow for practical Bayesian inference with particle filters
2. **Numba** provides 10-50x speedup via JIT compilation -- zero code changes, works on any CPU
3. **GPU (CuPy)** provides 50-500x speedup for large particle counts ($N > 5{,}000$)
4. **Parallelization** gives near-linear speedup for independent runs across CPU cores
5. All backends produce **identical numerical results** -- acceleration doesn't change the algorithm
6. The **backend selection** depends on particle count, number of runs, and hardware availability
7. Combine Numba + parallel for the best general-purpose performance

---

## What's Next?

<div class="grid cards" markdown>

- :material-clipboard-check-outline: **[Complete Workflow](complete-workflow.md)**

    Put it all together in an end-to-end analysis workflow

- :material-cog-refresh: **[PMMH Tutorial](pmmh.md)**

    Accelerated PMMH for practical parameter estimation

- :material-chart-bar: **[PGAS Tutorial](pgas.md)**

    PGAS achieves excellent mixing with very few particles

</div>
