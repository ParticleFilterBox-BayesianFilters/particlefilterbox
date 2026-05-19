# Acceleration

## Overview

particlefilterbox provides a **layered acceleration architecture** that lets you scale from rapid prototyping on a laptop to production-grade inference on GPU clusters --- without changing your model code.

```
Pure Python  ──▶  Numba JIT  ──▶  GPU (CuPy / JAX)
   1×              10–50×            50–500×
```

!!! tip "Rule of thumb"
    Start with the default **Pure Python** backend for development and debugging.
    Switch to **Numba** when $N \geq 1\,000$ and wall-clock time matters.
    Move to **GPU** when $N \geq 10\,000$ and you have CUDA hardware available.

---

## Acceleration Hierarchy

| Level | Backend | Typical Speedup | Best For | Requirements |
|-------|---------|-----------------|----------|--------------|
| 0 | `'python'` (default) | 1× | Prototyping, debugging, small $N$ | None |
| 1 | `'numba'` | 10--50× | Medium $N$ ($10^3$--$10^4$), CPU-bound workloads | `numba` package |
| 2 | `'cupy'` | 50--500× | Large $N$ ($10^4$--$10^6$), NVIDIA GPUs | CUDA toolkit, `cupy` |
| 2 | `'jax'` | 50--500× | Large $N$, GPU/TPU, differentiable models | `jax`, `jaxlib` |

All backends share the **same user-facing API** --- only the `backend` parameter changes:

```python
from particlefilterbox import BootstrapFilter

# Pure Python (default)
bpf = BootstrapFilter(model, n_particles=1000)

# Numba JIT
bpf = BootstrapFilter(model, n_particles=10000, backend='numba')

# GPU via CuPy
bpf = BootstrapFilter(model, n_particles=100000, backend='cupy')

# GPU/TPU via JAX
bpf = BootstrapFilter(model, n_particles=100000, backend='jax')
```

---

## Filter × Backend Compatibility

The table below summarises which filters support which acceleration backends.

| Filter / Algorithm | `python` | `numba` | `cupy` | `jax` |
|--------------------|:--------:|:-------:|:------:|:-----:|
| **BootstrapFilter** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **GuidedFilter (APF)** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **UncentedPF** | :white_check_mark: | :white_check_mark: | :material-minus: | :white_check_mark: |
| **EnKF** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **PMMH** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **SMC²** | :white_check_mark: | :white_check_mark: | :material-minus: | :material-minus: |
| **Particle Smoother** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Two-Filter Smoother** | :white_check_mark: | :white_check_mark: | :material-minus: | :material-minus: |

!!! info "Legend"
    :white_check_mark: Full support  
    :material-minus: Partial or experimental support

---

## When to Use Each Level

### Pure Python

- **Model development** --- full traceability, easy breakpoints.
- **Small datasets** --- $T < 200$, $N < 500$.
- **Complex model logic** --- arbitrary Python objects in the state.

### Numba

- **Medium-scale inference** --- $N \in [10^3, 10^4]$.
- **CPU-only environments** --- cloud VMs without GPU.
- **Repeated runs** --- JIT cache amortises compilation cost.

### GPU (CuPy / JAX)

- **Large-scale inference** --- $N > 10^4$.
- **Real-time applications** --- streaming data, tight latency budgets.
- **Differentiable inference** --- JAX backend enables gradient-based parameter learning.

---

## Additional Acceleration Features

Beyond backend selection, particlefilterbox provides complementary acceleration strategies:

| Feature | Description | Page |
|---------|-------------|------|
| **Parallel execution** | Run independent filter replicas across CPU cores or Dask clusters | [Parallel](parallel.md) |
| **Adaptive $N$** | Dynamically adjust particle count based on ESS, reducing total cost | [Adaptive N](adaptive-n.md) |

---

## Quick Benchmark

Stochastic volatility model, $T = 1\,000$ observations, single run:

| Backend | $N = 1\,000$ | $N = 10\,000$ | $N = 100\,000$ |
|---------|:-----------:|:------------:|:-------------:|
| `python` | 1.2 s | 12 s | 120 s |
| `numba` | 0.08 s | 0.5 s | 4.8 s |
| `cupy` | 0.15 s | 0.12 s | 0.35 s |
| `jax` | 0.20 s | 0.14 s | 0.38 s |

!!! warning "Benchmark caveat"
    Timings are indicative and depend on hardware. GPU timings exclude first-call compilation / transfer overhead. Run your own benchmarks with `particlefilterbox.benchmark()`.

---

## Performance Tips

!!! tip "Performance Tips"

    1. **Profile before optimising** --- use the [Experiment Framework](../user-guide/experiment.md) to benchmark your specific model before committing to a backend.
    2. **Start with Numba** --- it is zero-effort and gives 10--50× speedup. Only move to GPU when Numba is not enough.
    3. **Match $N$ to the problem** --- run an [N-study](../diagnostics/convergence.md) first. If $N = 500$ suffices, no backend switch is needed.
    4. **Combine strategies** --- [Adaptive N](adaptive-n.md) + Numba often outperforms fixed-$N$ + GPU at lower cost.
    5. **Monitor ESS** --- use [ESS diagnostics](../diagnostics/ess-diagnostic.md) to verify that acceleration does not silently degrade filtering quality.
    6. **Watch for degeneracy** --- when scaling up $N$ with GPU, check [weight diagnostics](../diagnostics/weight-diagnostic.md) to ensure the extra particles are actually useful.

---

## Quick Backend Reference

!!! abstract "Backend → Speedup → When to Use"

    | Backend | Typical Speedup | Best For | Limitations |
    |---------|:--------------:|----------|-------------|
    | `'python'` | 1× | Prototyping, debugging, $N < 500$ | Slow for production |
    | `'numba'` | 10--50× | CPU-only, $N \in [10^3, 10^4]$, repeated runs | First-call compilation overhead; model must be [Numba-compatible](numba.md#writing-numba-compatible-models) |
    | `'cupy'` | 50--500× | $N > 10^4$, NVIDIA GPUs, batch processing | Requires CUDA; [GPU memory constraints](gpu.md#memory-management) |
    | `'jax'` | 50--500× | $N > 10^4$, differentiable inference, TPU | Requires [functional style](gpu.md#jax-backend); harder to debug |
    | [Parallel](parallel.md) | Linear in cores | Independent replicas, [SMC²](../user-guide/smc/smc-squared.md), sweeps | Communication overhead at high core counts |
    | [Adaptive N](adaptive-n.md) | 3--10× cost reduction | Variable-difficulty time series | Slight overhead per step; [PMMH compatibility](adaptive-n.md#trade-offs-and-limitations) needs correction |

---

## Next Steps

- [Numba JIT Compilation](numba.md) --- CPU-level speedups with minimal code changes.
- [GPU Acceleration](gpu.md) --- Massive parallelism with CuPy or JAX.
- [Parallel Execution](parallel.md) --- Multi-core and distributed computing.
- [Adaptive N](adaptive-n.md) --- Smart particle budget allocation.

---

## See Also

- **Diagnostics**: [ESS](../diagnostics/ess-diagnostic.md) · [Convergence / N-study](../diagnostics/convergence.md) · [Weight Analysis](../diagnostics/weight-diagnostic.md) --- verify filtering quality after switching backends
- **User Guide**: [Filters](../user-guide/filters/index.md) · [PMCMC](../user-guide/pmcmc/index.md) · [Experiment Framework](../user-guide/experiment.md)
- **Theory**: [Convergence Theory](../theory/convergence-theory.md) --- why $N$ matters and how error scales
