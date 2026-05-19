# Numba JIT Compilation

## Overview

The **Numba backend** applies Just-In-Time (JIT) compilation to the numerical core of particle filters --- resampling, weight computation, and state propagation --- delivering **10--50× speedups** over pure Python with zero changes to your model specification.

```python
from particlefilterbox import BootstrapFilter

bpf = BootstrapFilter(model, n_particles=10000, backend='numba')
result = bpf.filter(observations)
```

!!! tip "When to use Numba"
    Numba is the **best first step** for acceleration. It requires no GPU hardware, works on any platform with a C compiler, and provides substantial speedups for $N \in [10^3, 10^4]$.

---

## How It Works

When `backend='numba'` is set, particlefilterbox wraps the performance-critical inner loops with Numba's `@njit` decorator:

```python
from numba import njit

@njit
def _systematic_resample(weights, u):
    """Systematic resampling — compiled to machine code."""
    N = weights.shape[0]
    positions = (u + np.arange(N)) / N
    indices = np.empty(N, dtype=np.int64)
    cumsum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices
```

The key operations accelerated by Numba are:

| Operation | Description | Typical Speedup |
|-----------|-------------|:--------------:|
| Systematic resampling | Index computation from normalised weights | 20--40× |
| Multinomial resampling | Inverse-CDF sampling | 15--30× |
| Weight computation | Log-likelihood evaluation across particles | 10--25× |
| State propagation | Transition density sampling | 10--50× |
| ESS computation | Effective sample size from weights | 30--50× |

---

## Enabling Numba

### Per-filter activation

```python
from particlefilterbox import BootstrapFilter, GuidedFilter, PMMH

# Bootstrap filter with Numba
bpf = BootstrapFilter(model, n_particles=10000, backend='numba')

# Auxiliary particle filter with Numba
apf = GuidedFilter(model, n_particles=5000, backend='numba')

# PMMH with Numba-accelerated likelihood
pmmh = PMMH(model, n_particles=5000, backend='numba')
```

### Global default

```python
import particlefilterbox as pfb

pfb.set_backend('numba')  # All subsequent filters use Numba

bpf = pfb.BootstrapFilter(model, n_particles=10000)  # Uses Numba
```

---

## Writing Numba-Compatible Models

For the Numba backend to compile your model's transition and observation functions, they must be **Numba-compatible** --- i.e., use only supported types and operations.

### Supported patterns

=== "Numba-Compatible Model"

    ```python
    import numpy as np
    from particlefilterbox import StateSpaceModel

    class StochasticVolatility(StateSpaceModel):
        params = ['mu', 'phi', 'sigma']

        def transition(self, x, t, rng):
            """State transition — Numba-compatible."""
            mu, phi, sigma = self.mu, self.phi, self.sigma
            return mu + phi * (x - mu) + sigma * rng.standard_normal(x.shape)

        def observation(self, x, t):
            """Observation density — Numba-compatible."""
            return np.exp(x / 2.0)
    ```

=== "NOT Numba-Compatible"

    ```python
    class BadModel(StateSpaceModel):
        def transition(self, x, t, rng):
            # ❌ Python dicts not supported in nopython mode
            params = {'mu': 0.0, 'phi': 0.95}

            # ❌ String operations not supported
            label = f"state_{t}"

            # ❌ List comprehensions with complex objects
            return [scipy.stats.norm.rvs() for _ in range(len(x))]
    ```

### Rules for Numba compatibility

| Allowed | Not Allowed |
|---------|-------------|
| NumPy arrays and scalar types | Python dicts, sets, custom classes |
| NumPy mathematical functions | SciPy functions (most) |
| Basic control flow (`if`, `for`, `while`) | String formatting, f-strings |
| Tuple creation and indexing | List comprehensions with complex logic |
| `np.random` via Numba's RNG | `scipy.stats` distributions |
| In-place array operations | Dynamic object creation |

!!! info "Automatic fallback"
    If a model function cannot be compiled, particlefilterbox issues a warning and falls back to pure Python for that function while keeping Numba for the rest of the pipeline.

---

## Benchmarks

### Stochastic Volatility Model

$T = 1\,000$ time steps, averaged over 50 runs (excluding first-call compilation):

| $N$ (particles) | Python (s) | Numba (s) | Speedup |
|:---------------:|:----------:|:---------:|:-------:|
| 500 | 0.6 | 0.05 | **12×** |
| 1,000 | 1.2 | 0.08 | **15×** |
| 5,000 | 5.8 | 0.28 | **21×** |
| 10,000 | 12.0 | 0.50 | **24×** |
| 50,000 | 58.0 | 1.80 | **32×** |

### Linear Gaussian Model (Comparison with kalmanbox)

| Method | Time (s) | Accuracy (RMSE) |
|--------|:--------:|:---------------:|
| Kalman filter (kalmanbox) | 0.002 | Exact |
| BPF, $N=1\,000$, Python | 1.2 | 0.15 |
| BPF, $N=1\,000$, Numba | 0.08 | 0.15 |
| BPF, $N=10\,000$, Numba | 0.50 | 0.05 |

### Scaling Behaviour

```
Speedup vs Pure Python (Stochastic Volatility, T=1000)
  50× ┤
      │                                          ●
  40× ┤                                  ●
      │
  30× ┤                          ●
      │
  20× ┤                  ●
      │          ●
  10× ┤  ●
      │
   0× ┼──────────────────────────────────────────
      500  1K   5K   10K  50K  100K
                  N (particles)
```

!!! info "Scaling insight"
    Numba speedup **increases with $N$** because the overhead of the JIT dispatch becomes negligible relative to the vectorised inner loops.

---

## First-Call Overhead and Caching

### Compilation overhead

The **first call** with `backend='numba'` triggers JIT compilation, which takes 2--10 seconds depending on model complexity. Subsequent calls use the cached compiled code.

```python
import time

bpf = BootstrapFilter(model, n_particles=10000, backend='numba')

# First call — includes compilation
t0 = time.time()
result1 = bpf.filter(observations)
print(f"First call: {time.time() - t0:.2f}s")   # ~3.5s

# Second call — uses cache
t0 = time.time()
result2 = bpf.filter(observations)
print(f"Second call: {time.time() - t0:.2f}s")   # ~0.5s
```

### Persistent cache

Numba stores compiled functions in a persistent cache directory. To control caching:

```python
import particlefilterbox as pfb

# Enable persistent cache (default)
pfb.config.numba_cache = True

# Set custom cache directory
pfb.config.numba_cache_dir = '/tmp/pfb_numba_cache'

# Clear cache (useful after model changes)
pfb.clear_numba_cache()
```

!!! warning "Cache invalidation"
    The cache is keyed on function signatures and Numba version. Upgrading Numba or changing function parameter types will trigger recompilation.

---

## Troubleshooting

### Common compilation errors

**`TypingError: Failed in nopython mode`**

```
numba.core.errors.TypingError: Failed in nopython mode pipeline
  ... Cannot resolve function type: scipy.stats.norm.pdf
```

**Cause**: Using a function not supported by Numba's nopython mode.

**Fix**: Replace with NumPy equivalents:

```python
# ❌ SciPy (not Numba-compatible)
log_lik = scipy.stats.norm.logpdf(y, loc=x, scale=sigma)

# ✅ NumPy equivalent
log_lik = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y - x) / sigma)**2
```

---

**`UnsupportedError: Use of unknown opcode DICT_MERGE`**

**Cause**: Using Python dicts in a compiled function.

**Fix**: Use tuples or arrays for parameter passing:

```python
# ❌ Dict
params = {'mu': 0.0, 'phi': 0.95, 'sigma': 0.2}

# ✅ Tuple or individual attributes
mu, phi, sigma = 0.0, 0.95, 0.2
```

---

**`LoweringError: Failed at nopython mode ... np.random.normal`**

**Cause**: Numba uses its own RNG, which may differ from NumPy's.

**Fix**: Use the `rng` object passed by particlefilterbox:

```python
# ❌ Global NumPy RNG
noise = np.random.normal(0, sigma, size=x.shape)

# ✅ Numba-compatible RNG from particlefilterbox
noise = sigma * rng.standard_normal(x.shape)
```

---

### Type inference issues

If Numba cannot infer types, provide explicit type hints:

```python
from numba import float64

@njit(float64[:](float64[:], float64, float64, float64))
def transition_jit(x, mu, phi, sigma):
    return mu + phi * (x - mu) + sigma * np.random.randn(x.shape[0])
```

---

### Performance checklist

!!! abstract "Numba Performance Checklist"
    - [ ] Avoid Python objects (dicts, lists of mixed types) inside hot loops
    - [ ] Use contiguous NumPy arrays (`np.ascontiguousarray`)
    - [ ] Prefer in-place operations to reduce allocations
    - [ ] Keep arrays in `float64` (Numba's best-optimised type)
    - [ ] Profile with `numba.core.types` to verify type inference
    - [ ] Set `cache=True` for persistent compilation caching

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `'python'` | Set to `'numba'` to enable JIT compilation |
| `numba_cache` | `bool` | `True` | Enable persistent caching of compiled functions |
| `numba_cache_dir` | `str` | `None` | Custom cache directory (default: Numba's own) |
| `numba_parallel` | `bool` | `False` | Enable Numba's automatic parallelisation (`prange`) |
| `numba_fastmath` | `bool` | `False` | Allow fast-math optimisations (trades accuracy for speed) |

```python
bpf = BootstrapFilter(
    model,
    n_particles=10000,
    backend='numba',
    numba_parallel=True,    # Use prange for particle loops
    numba_fastmath=True,    # ~10% extra speedup, slight precision loss
)
```

---

## See Also

- [Acceleration Overview](index.md) --- Comparison of all backends
- [GPU Acceleration](gpu.md) --- For $N > 10\,000$ with GPU hardware
- [Adaptive N](adaptive-n.md) --- Reduce $N$ dynamically to save compute
- [Convergence Diagnostic](../diagnostics/convergence.md) --- N-study to determine the right $N$ for your model
- [ESS Diagnostic](../diagnostics/ess-diagnostic.md) --- monitor filtering quality with Numba backend
- [Filters Overview](../user-guide/filters/index.md) --- all filters support the Numba backend
- [Experiment Framework](../user-guide/experiment.md) --- benchmark Numba vs Python in systematic experiments
