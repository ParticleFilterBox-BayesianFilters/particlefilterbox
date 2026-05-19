---
title: "Benchmarks Overview"
description: "Performance methodology and summary for particlefilterbox — hardware, software, replication, and cross-page index of filter, acceleration, PMCMC, and library comparisons."
---

# Benchmarks Overview

This section reports **measured performance** of particlefilterbox across filters, backends, and PMCMC methods, plus cross-library comparisons. Every number on the following pages comes from the `benchmarks/` harness shipped with the repository and can be reproduced on your own machine.

!!! info "Benchmark Environment"
    Unless noted otherwise, results were measured on:

    - **CPU**: Intel Core i9-13900K (24 cores, 5.8 GHz boost), **RAM** 64 GB DDR5
    - **GPU**: NVIDIA RTX 4090 (24 GB, CUDA 12.4)
    - **OS**: Ubuntu 22.04 LTS, Linux kernel 6.6
    - **Python**: 3.12.2 | **NumPy** 2.0 (OpenBLAS) | **SciPy** 1.14
    - **Numba**: 0.60 | **CuPy**: 13.2 | **JAX**: 0.4.30

    Results will vary with CPU generation, BLAS backend, GPU, and the exact model under test.

## Page Index

<div class="grid cards" markdown>

-   :material-scatter-plot: **[Filters](filters.md)**

    Bootstrap vs SIR vs Auxiliary vs RBPF vs UPF on the canonical SV model. RMSE, log-likelihood, ESS, runtime at $N \in \{100, 500, 1000, 5000\}$.

-   :material-lightning-bolt: **[Acceleration](acceleration.md)**

    Pure Python vs Numba vs CuPy vs JAX on Bootstrap PF at $N \in \{10^3, 10^4, 10^5\}$. Speedups and memory usage.

-   :material-sync: **[PMCMC](pmcmc.md)**

    PMMH vs Particle Gibbs vs PG-AS on SV. Effective sample size **per second**, mixing time, acceptance rate, $N$-sensitivity.

-   :material-scale-balance: **[Library Comparison](comparison.md)**

    particlefilterbox vs [`particles`](https://github.com/nchopin/particles) (Python), SMC.jl (Julia), and a hand-rolled reference implementation.

</div>

## Methodology

### Timing protocol

- Each measurement is the **median of 10 runs** after **3 warm-up runs** (to allow JIT compilation, cache warm-up, and first-call overheads to stabilize).
- Reported error bars are interquartile ranges across the 10 timed runs, not confidence intervals over datasets.
- Runs with a cold Numba cache are labeled separately when the first-call latency is material.
- GPU measurements use `cuda.synchronize()` around the timed region.

### Statistical metrics

For a ground-truth state path $x_{1:T}$ (from simulation) and filter mean $\hat{x}_t$:

- **RMSE** — $\sqrt{\tfrac{1}{T}\sum_t (x_t - \hat{x}_t)^2}$.
- **ESS (mean)** — time-average of the effective sample size, $\overline{\text{ESS}} = \tfrac{1}{T}\sum_t \text{ESS}_t$.
- **Log-marginal-likelihood** — $\log \hat{p}(y_{1:T})$, estimated via the product of PF normalizers.
- **MAP** — maximum *a-posteriori* state at each $t$, using particle-weighted mode.
- **ESS/sec** (PMCMC) — posterior ESS of $\theta$ divided by wall-time.

Each metric is averaged over **20 independent datasets** simulated from the data-generating process. We report the mean across datasets and the 5–95% inter-dataset range when relevant.

### Canonical model

Most filter and PMCMC benchmarks use the **univariate stochastic volatility (SV) model**:

$$
\begin{aligned}
x_t &= \mu + \phi (x_{t-1} - \mu) + \sigma_\eta \, \eta_t, & \eta_t &\sim \mathcal{N}(0, 1) \\
y_t &= \exp(x_t / 2) \, \varepsilon_t, & \varepsilon_t &\sim \mathcal{N}(0, 1)
\end{aligned}
$$

with true parameters $\mu = 0$, $\phi = 0.97$, $\sigma_\eta = 0.15$, $T = 500$. This is the workhorse used in Chopin & Papaspiliopoulos (2020) and most SMC benchmarks in the literature.

### Hardware normalization

To allow comparison across machines, key numbers on each page are also reported **normalized to a single-threaded NumPy Bootstrap PF** on the same hardware. This isolates *relative* speedups from absolute hardware differences. For example, a "12×" Numba speedup means 12× the baseline on *your* machine, regardless of whether your baseline itself is fast or slow.

## How to Reproduce

All benchmark code lives under `benchmarks/` in the repository.

```bash
git clone https://github.com/nodesecon/particlefilterbox.git
cd particlefilterbox
pip install -e ".[all]"

# quick: filter-suite only, 3 replicates
pytest benchmarks/filters.py --benchmark-warmup=on --benchmark-only

# full: all suites, 10 replicates (takes ~2h on a workstation)
python benchmarks/run_all.py --output results.json --n-replicates 10

# render reports to HTML
python benchmarks/render.py results.json --out benchmarks-html/
```

Results are written to `benchmarks/results/<hardware_id>/` and ingested into a table renderer that produces exactly the tables shown on the individual pages. Raw JSON for each run is committed under `benchmarks/results/reference/` so you can compare against the reference machine's numbers.

??? tip "Tips for clean benchmarks"
    - **Disable CPU frequency scaling** — on Linux, `cpupower frequency-set -g performance`.
    - **Pin to a single NUMA node** — `numactl --cpunodebind=0 --membind=0 python ...`.
    - **Close other processes**, especially anything with a live Python REPL (holding the GIL in another thread still affects Numba-parallel code).
    - **Warm the cache** — run each benchmark at least once before timing.
    - **Use `-m pytest benchmarks/`** with the `pytest-benchmark` plugin when investigating a single scenario — it handles warm-ups and repeat counts for you.

## Relation to Upstream Libraries

particlefilterbox is built on NumPy/SciPy with optional Numba / CuPy / JAX backends. Comparisons with other libraries (page [Library Comparison](comparison.md)) follow the principle that **apples match apples**: same model, same $N$, same seed, same output metric. When a competing library's default differs (e.g. different default resampling scheme), the benchmark uses *its* default and notes the difference.

!!! tip "Cross-platform note"
    For linear-Gaussian models, `particlefilterbox` will always be slower than `kalmanbox` — the Kalman filter runs in $O(d^3)$ per step versus the particle filter's $O(N \cdot d)$. Use `particlefilterbox` only when the model is nonlinear or non-Gaussian. The [Library Comparison](comparison.md) page includes a "linear-Gaussian regression test" showing when each is preferred.

## Interpreting the Numbers

- **Runtime scales roughly linearly in $N$** for all filters — a Bootstrap PF at $N = 10^4$ is ~10× slower than at $N = 10^3$.
- **RMSE scales as $\mathcal{O}(N^{-1/2})$** — to halve the error you need 4× the particles.
- **Numba speedup is fixed overhead** — Python call dispatch, memory allocation, indexing — so benefits grow with $N$. Below $N \approx 200$, NumPy wins.
- **GPU wins start around $N \gtrsim 10^4$**, because kernel launch overhead dominates for smaller $N$. Don't go GPU for toy problems.
- **PMCMC efficiency depends on the product** of per-iteration cost *and* mixing. A faster filter that makes PMMH reject more may be worse per effective sample.

---

## See Also

- [Acceleration Guide](../acceleration/index.md) — when and how to use each backend.
- [Choosing a Filter](../getting-started/choosing-filter.md) — statistical criteria, not just timing.
- [Tuning PMCMC](../user-guide/pmcmc/tuning.md) — the control variables that drive ESS/sec.
