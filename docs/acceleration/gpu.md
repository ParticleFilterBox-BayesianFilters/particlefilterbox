# GPU Acceleration (CuPy / JAX)

## Overview

For large particle counts ($N > 10\,000$), GPU acceleration unlocks **50--500× speedups** by executing particle operations in massively parallel fashion on NVIDIA GPUs or Google TPUs. particlefilterbox supports two GPU backends:

- **CuPy** --- Drop-in NumPy replacement for NVIDIA CUDA GPUs.
- **JAX** --- Composable transformations (JIT, grad, vmap) on GPU/TPU.

```python
from particlefilterbox import BootstrapFilter

# CuPy backend (NVIDIA GPU)
bpf = BootstrapFilter(model, n_particles=100000, backend='cupy')

# JAX backend (GPU or TPU)
bpf = BootstrapFilter(model, n_particles=100000, backend='jax')
```

!!! tip "When to use GPU"
    GPU acceleration is most effective when $N \geq 10\,000$. Below this threshold, **CPU-GPU data transfer overhead** can negate the parallelism gains. Use [Numba](numba.md) for moderate $N$.

---

## GPU-Accelerated Operations

| Operation | CuPy | JAX | Description |
|-----------|:----:|:---:|-------------|
| State propagation | :white_check_mark: | :white_check_mark: | Vectorised transition across all particles |
| Weight computation | :white_check_mark: | :white_check_mark: | Parallel log-likelihood evaluation |
| Systematic resampling | :white_check_mark: | :white_check_mark: | Prefix-sum based parallel resampling |
| Multinomial resampling | :white_check_mark: | :white_check_mark: | GPU-parallel inverse CDF |
| ESS computation | :white_check_mark: | :white_check_mark: | Parallel reduction |
| Stratified resampling | :white_check_mark: | :white_check_mark: | Stratified index generation on GPU |
| Smoothing (FFBSi) | :white_check_mark: | :white_check_mark: | Backward sampling with GPU kernels |

---

## CuPy Backend

### Setup

=== "pip"

    ```bash
    # For CUDA 11.x
    pip install cupy-cuda11x

    # For CUDA 12.x
    pip install cupy-cuda12x
    ```

=== "conda"

    ```bash
    conda install -c conda-forge cupy cudatoolkit
    ```

### Verify installation

```python
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # Should be >= 1
print(cp.cuda.Device(0).name)            # e.g., 'NVIDIA A100-SXM4-40GB'
```

### Usage

```python
from particlefilterbox import BootstrapFilter, GuidedFilter

# Bootstrap filter on GPU
bpf = BootstrapFilter(model, n_particles=100000, backend='cupy')
result = bpf.filter(observations)

# Auxiliary particle filter on GPU
apf = GuidedFilter(model, n_particles=50000, backend='cupy')
result = apf.filter(observations)
```

!!! info "Automatic data transfer"
    Observations and model parameters are automatically transferred to GPU memory. Results are copied back to CPU NumPy arrays. No manual `cupy.asarray()` needed.

---

## JAX Backend

### Setup

=== "GPU (NVIDIA CUDA)"

    ```bash
    pip install jax[cuda12]
    ```

=== "TPU (Google Cloud)"

    ```bash
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    ```

=== "CPU only (for testing)"

    ```bash
    pip install jax[cpu]
    ```

### Verify installation

```python
import jax
print(jax.devices())          # [GpuDevice(id=0, ...)]
print(jax.default_backend())  # 'gpu' or 'tpu'
```

### Usage

```python
from particlefilterbox import BootstrapFilter

bpf = BootstrapFilter(model, n_particles=100000, backend='jax')
result = bpf.filter(observations)
```

### JAX-specific features

JAX enables **differentiable particle filtering**, which is useful for gradient-based parameter learning:

```python
import jax
import jax.numpy as jnp

def neg_log_likelihood(params, observations):
    model = StochasticVolatility(**params)
    bpf = BootstrapFilter(model, n_particles=10000, backend='jax')
    result = bpf.filter(observations)
    return -result.log_likelihood

# Gradient of log-likelihood w.r.t. parameters
grad_fn = jax.grad(neg_log_likelihood)
grads = grad_fn({'mu': 0.0, 'phi': 0.95, 'sigma': 0.2}, observations)
```

!!! warning "JAX functional constraints"
    JAX requires **pure functions** --- no side effects, no in-place mutation. Models using `jax` backend must avoid in-place array operations (`x[i] = val`) and use functional alternatives (`x = x.at[i].set(val)`).

---

## Benchmarks

### Stochastic Volatility Model

$T = 1\,000$ time steps, NVIDIA A100 GPU, averaged over 50 runs:

| $N$ (particles) | Python (s) | Numba (s) | CuPy (s) | JAX (s) |
|:---------------:|:----------:|:---------:|:---------:|:-------:|
| 1,000 | 1.2 | 0.08 | 0.15 | 0.20 |
| 10,000 | 12.0 | 0.50 | 0.12 | 0.14 |
| 100,000 | 120.0 | 4.80 | 0.35 | 0.38 |
| 1,000,000 | --- | 48.0 | 2.10 | 2.30 |

### Speedup vs Pure Python

```
Speedup (Stochastic Volatility, T=1000, NVIDIA A100)
 500× ┤
      │                                   ● CuPy
 400× ┤                                   ○ JAX
      │
 300× ┤                          ●
      │                          ○
 200× ┤
      │
 100× ┤                  ●
      │                  ○
   0× ┼──●───────────────────────────────────
      1K     10K     100K     1M
                  N (particles)
```

!!! info "Crossover point"
    GPU backends overtake Numba at approximately **$N = 5\,000$--$10\,000$**. Below this range, CPU-GPU transfer overhead dominates.

### Cost-Accuracy Trade-off

For a fixed time budget of 1 second ($T = 1\,000$):

| Backend | Max $N$ in 1s | Filter RMSE |
|---------|:------------:|:-----------:|
| Python | 800 | 0.18 |
| Numba | 20,000 | 0.04 |
| CuPy | 250,000 | 0.008 |
| JAX | 220,000 | 0.009 |

---

## Memory Management

### GPU memory usage

Each particle stores the state vector $x_t \in \mathbb{R}^d$ plus a weight scalar. The memory footprint is:

$$
\text{GPU memory} \approx N \times (d + 1) \times 8 \;\text{bytes (float64)}
$$

| $N$ | $d = 1$ | $d = 5$ | $d = 20$ |
|-----|:-------:|:-------:|:--------:|
| 10,000 | 0.15 MB | 0.46 MB | 1.6 MB |
| 100,000 | 1.5 MB | 4.6 MB | 16 MB |
| 1,000,000 | 15 MB | 46 MB | 160 MB |

### Batch processing for memory-constrained GPUs

When GPU memory is insufficient for the full particle set, use batch processing:

```python
bpf = BootstrapFilter(
    model,
    n_particles=1000000,
    backend='cupy',
    gpu_batch_size=100000,   # Process 100K particles at a time
)
result = bpf.filter(observations)
```

### Multi-GPU support

```python
bpf = BootstrapFilter(
    model,
    n_particles=1000000,
    backend='cupy',
    gpu_devices=[0, 1],   # Distribute across 2 GPUs
)
```

!!! warning "Multi-GPU overhead"
    Resampling across GPUs requires inter-device communication. For most workloads, a single large GPU outperforms two smaller ones.

---

## CuPy vs JAX: When to Choose Which

| Criterion | CuPy | JAX |
|-----------|------|-----|
| **Drop-in NumPy replacement** | :white_check_mark: Excellent | :material-minus: Requires functional style |
| **Gradient computation** | :material-minus: Not built-in | :white_check_mark: `jax.grad` |
| **TPU support** | :material-close: No | :white_check_mark: Yes |
| **Custom CUDA kernels** | :white_check_mark: `cp.RawKernel` | :material-minus: Via XLA |
| **Compilation model** | Eager (like NumPy) | JIT by default |
| **Ecosystem** | NVIDIA-centric | Google/DeepMind ecosystem |
| **Debugging** | Easier (eager mode) | Harder (traced execution) |

!!! abstract "Recommendation"
    - Choose **CuPy** if you want the simplest GPU migration path and don't need gradients.
    - Choose **JAX** if you need **differentiable filtering**, TPU support, or want to combine with Flax/Optax for deep learning.

---

## Setup Troubleshooting

### CUDA version mismatch

```
CUDADriverError: CUDA driver version is insufficient
```

**Fix**: Ensure your NVIDIA driver supports the CUDA version required by CuPy/JAX. Check with:

```bash
nvidia-smi   # Shows driver version and max CUDA version
nvcc --version   # Shows installed CUDA toolkit version
```

### Out of GPU memory

```
cupy.cuda.memory.OutOfMemoryError: Out of memory
```

**Fix**: Reduce `n_particles`, enable `gpu_batch_size`, or use `float32`:

```python
bpf = BootstrapFilter(
    model,
    n_particles=100000,
    backend='cupy',
    dtype='float32',         # Halves memory usage
    gpu_batch_size=50000,    # Process in batches
)
```

### JAX not detecting GPU

```python
import jax
print(jax.devices())  # [CpuDevice(id=0)] — GPU not found!
```

**Fix**: Install the correct `jaxlib` with CUDA support:

```bash
pip install jax[cuda12]
```

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `'python'` | Set to `'cupy'` or `'jax'` |
| `dtype` | `str` | `'float64'` | Use `'float32'` to halve GPU memory |
| `gpu_batch_size` | `int` | `None` | Process particles in batches (memory saving) |
| `gpu_devices` | `list[int]` | `None` | List of GPU device IDs for multi-GPU |
| `jax_jit` | `bool` | `True` | Enable JAX JIT compilation |
| `jax_platform` | `str` | `None` | Force platform: `'gpu'`, `'tpu'`, or `'cpu'` |

---

## See Also

- [Acceleration Overview](index.md) --- Backend comparison and compatibility table
- [Numba JIT](numba.md) --- CPU acceleration for moderate $N$
- [Parallel Execution](parallel.md) --- Combine GPU with multi-process runs
- [Adaptive N](adaptive-n.md) --- Dynamically reduce particle count
- [Convergence Diagnostic](../diagnostics/convergence.md) --- N-study to determine if GPU-scale $N$ is necessary
- [ESS Diagnostic](../diagnostics/ess-diagnostic.md) --- verify GPU runs maintain filtering quality
- [Filters Overview](../user-guide/filters/index.md) --- which filters support GPU backends
- [Experiment Framework](../user-guide/experiment.md) --- benchmark GPU vs CPU in a systematic experiment
