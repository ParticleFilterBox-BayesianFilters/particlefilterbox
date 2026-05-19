---
title: "Acceleration API"
description: "API reference for particlefilterbox.acceleration — Numba JIT, CuPy/JAX GPU backends, parallel runners, and adaptive particle sizing"
---

# Acceleration API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.acceleration import NumbaBackend, CuPyBackend, JAXBackend, ParallelRunner, AdaptiveN`
    **Source**: `particlefilterbox/acceleration/`

## Overview

The acceleration module provides optional backends for accelerating particle filters via JIT compilation (Numba), GPU execution (CuPy, JAX), multi-core parallelism, and adaptive sizing of the particle cloud driven by ESS targets.

| Class | Description | Hardware |
|-------|-------------|----------|
| `NumbaBackend` | JIT-compiled CPU kernels for propagation, weighting, and resampling | CPU |
| `CuPyBackend` | GPU arrays and kernels via CuPy (CUDA) | NVIDIA GPU |
| `JAXBackend` | XLA-compiled kernels via JAX | CPU / GPU / TPU |
| `ParallelRunner` | Embarrassingly parallel execution of filters or experiments | CPU cluster |
| `AdaptiveN` | Adjusts particle count $N_t$ to hit an ESS target | Any |

!!! tip "Optional dependencies"
    Each backend is optional. Install with:
    ```bash
    pip install particlefilterbox[numba]   # NumbaBackend
    pip install particlefilterbox[cupy]    # CuPyBackend
    pip install particlefilterbox[jax]     # JAXBackend
    pip install particlefilterbox[all]     # all backends
    ```

---

## NumbaBackend

JIT compilation backend based on [Numba](https://numba.pydata.org/). Compiles hot inner loops (propagation, log-likelihood evaluation, resampling) to machine code via LLVM.

### Constructor

```python
class NumbaBackend(
    parallel: bool = True,
    cache: bool = True,
    fastmath: bool = False,
    nopython: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel` | `bool` | `True` | Enable `prange` for parallel loops |
| `cache` | `bool` | `True` | Cache compiled kernels across sessions |
| `fastmath` | `bool` | `False` | Allow IEEE-754 reordering (faster, less precise) |
| `nopython` | `bool` | `True` | Reject Python fallback paths |

### Methods

#### `compile()`

Compile a model's propagation and likelihood kernels.

```python
def compile(
    self,
    model: ParticleFilterModel,
) -> ParticleFilterModel
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `ParticleFilterModel` | Model whose kernels will be JIT-compiled |

**Returns**: A wrapped model with compiled `propagate`, `log_likelihood`, and (if available) `sample_proposal`.

#### `is_available()`

```python
@staticmethod
def is_available() -> bool
```

**Returns**: `True` if Numba is installed and imports correctly.

### Example

```python
from particlefilterbox.acceleration import NumbaBackend
from particlefilterbox.models import StochasticVolatility

backend = NumbaBackend(parallel=True, fastmath=True)
assert backend.is_available()

model = backend.compile(StochasticVolatility(variant='basic'))
```

---

## CuPyBackend

GPU backend based on [CuPy](https://cupy.dev/). Moves particle clouds and weights to the device and executes kernels on CUDA.

### Constructor

```python
class CuPyBackend(
    device: int = 0,
    dtype: str = "float32",
    mempool: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `int` | `0` | CUDA device ordinal |
| `dtype` | `str` | `"float32"` | Array dtype. `"float64"` for double precision |
| `mempool` | `bool` | `True` | Enable CuPy memory pool for allocation reuse |

### Methods

#### `to_gpu()`

Move a CPU `ParticleCloud` to the GPU.

```python
def to_gpu(
    self,
    cloud: ParticleCloud,
) -> GPUParticleCloud
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cloud` | `ParticleCloud` | CPU particle cloud |

**Returns**: `GPUParticleCloud` with `cupy.ndarray` storage.

#### `from_gpu()`

Copy a GPU cloud back to host memory.

```python
def from_gpu(
    self,
    cloud: GPUParticleCloud,
) -> ParticleCloud
```

#### `is_available()`

```python
@staticmethod
def is_available() -> bool
```

**Returns**: `True` if CuPy is installed and at least one CUDA device is visible.

### Example

```python
from particlefilterbox.acceleration import CuPyBackend

backend = CuPyBackend(device=0, dtype="float32")
gpu_cloud = backend.to_gpu(cpu_cloud)

# run kernels on GPU ...

cpu_cloud = backend.from_gpu(gpu_cloud)
```

---

## JAXBackend

XLA-compiled backend based on [JAX](https://jax.readthedocs.io/). Provides `jit`, `vmap`, and `pmap` transformations for CPU, GPU, and TPU.

### Constructor

```python
class JAXBackend(
    platform: str = "auto",
    dtype: str = "float32",
    jit: bool = True,
    vmap_batch: int | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `platform` | `str` | `"auto"` | `"cpu"`, `"gpu"`, `"tpu"`, or `"auto"` |
| `dtype` | `str` | `"float32"` | Array dtype |
| `jit` | `bool` | `True` | Apply `jax.jit` to kernels |
| `vmap_batch` | `int \| None` | `None` | Batch dimension for `jax.vmap` |

### Methods

#### `compile()`

```python
def compile(
    self,
    model: ParticleFilterModel,
) -> ParticleFilterModel
```

Returns a JAX-compatible model whose kernels run under `jit`/`vmap`.

#### `is_available()`

```python
@staticmethod
def is_available() -> bool
```

### Example

```python
from particlefilterbox.acceleration import JAXBackend

backend = JAXBackend(platform="gpu", jit=True)
model = backend.compile(sv_model)

pf = BootstrapPF(model, PFConfig(n_particles=100_000))
result = pf.filter(y)
```

---

## ParallelRunner

Embarrassingly parallel execution of filters or experiments across cores. Uses `joblib` or `multiprocessing` under the hood.

### Constructor

```python
class ParallelRunner(
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
    batch_size: str | int = "auto",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_jobs` | `int` | `-1` | Workers (`-1` = all cores) |
| `backend` | `str` | `"loky"` | `"loky"`, `"multiprocessing"`, `"threading"`, `"dask"` |
| `verbose` | `int` | `0` | Progress verbosity |
| `batch_size` | `str \| int` | `"auto"` | Tasks per batch |

### Methods

#### `run_filters()`

Run multiple filter configurations over the same observation sequence.

```python
def run_filters(
    self,
    filters: list[BaseParticleFilter],
    observations: NDArray[np.float64],
    seeds: list[int] | None = None,
) -> list[ParticleFilterResults]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `filters` | `list[BaseParticleFilter]` | Pre-constructed filter instances |
| `observations` | `NDArray[np.float64]` | Observation sequence shared across runs |
| `seeds` | `list[int] \| None` | Per-run RNG seeds for reproducibility |

**Returns**: A list of `ParticleFilterResults`, one per filter.

#### `run_experiments()`

```python
def run_experiments(
    self,
    configs: list[ExperimentConfig],
) -> list[ExperimentResult]
```

**Returns**: A list of `ExperimentResult` objects aligned with `configs`.

### Example

```python
from particlefilterbox.acceleration import ParallelRunner
from particlefilterbox.filters import BootstrapPF, AuxiliaryPF

runner = ParallelRunner(n_jobs=8, backend="loky")
results = runner.run_filters(
    filters=[BootstrapPF(m, cfg), AuxiliaryPF(m, cfg)],
    observations=y,
    seeds=[42, 43],
)
```

---

## AdaptiveN

Dynamically adjusts the particle count $N_t$ over time to keep the effective sample size near a target.

### Constructor

```python
class AdaptiveN(
    n_min: int = 500,
    n_max: int = 50_000,
    ess_target: float = 0.5,
    strategy: str = "ess_ratio",
    growth: float = 1.5,
    shrink: float = 0.8,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_min` | `int` | `500` | Lower bound on particle count |
| `n_max` | `int` | `50000` | Upper bound on particle count |
| `ess_target` | `float` | `0.5` | Target ESS ratio $\text{ESS}/N \in (0, 1)$ |
| `strategy` | `str` | `"ess_ratio"` | One of `"ess_ratio"`, `"variance"`, `"cv"` |
| `growth` | `float` | `1.5` | Multiplicative increase when below target |
| `shrink` | `float` | `0.8` | Multiplicative decrease when above target |

### Strategy

At each time $t$, given the current particle count $N_{t-1}$ and ESS ratio $\rho_t = \text{ESS}_t / N_{t-1}$:

$$
N_t = \begin{cases}
\min(N_{\max},\ \lceil \text{growth} \cdot N_{t-1} \rceil) & \rho_t < \text{ess\_target} \\
\max(N_{\min},\ \lceil \text{shrink} \cdot N_{t-1} \rceil) & \rho_t > \text{ess\_target} + \delta \\
N_{t-1} & \text{otherwise}
\end{cases}
$$

### Example

```python
from particlefilterbox.acceleration import AdaptiveN

adaptive = AdaptiveN(n_min=1000, n_max=20_000, ess_target=0.5)
config = PFConfig(n_particles=adaptive, resampling="systematic")
pf = BootstrapPF(model, config)
```

---

## See Also

- [Core API](core.md) — `PFConfig`, `ParticleCloud`
- [Experiment API](experiment.md) — uses `ParallelRunner` for batch runs
- [Diagnostics API](diagnostics.md) — ESS monitoring paired with `AdaptiveN`
