---
title: ParticleCloud
description: "The central data structure for weighted particle sets in particlefilterbox"
---

# ParticleCloud

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `ParticleCloud` |
    | **Import** | `from particlefilterbox.core import ParticleCloud` |
    | **Key params** | `n_particles`, `k_states` |
    | **Role** | Stores $N$ weighted particles in $\mathbb{R}^k$ and computes statistics |

## Overview

`ParticleCloud` is the fundamental data structure in particlefilterbox. It represents a discrete approximation to a probability distribution using $N$ weighted particles:

$$
\hat{p}(x) = \sum_{i=1}^{N} w_t^{(i)} \, \delta_{x_t^{(i)}}(x)
$$

where $x_t^{(i)} \in \mathbb{R}^k$ are particle positions and $w_t^{(i)}$ are normalized importance weights satisfying $\sum_i w_t^{(i)} = 1$.

Every filter, smoother, and SMC method in particlefilterbox operates on `ParticleCloud` instances internally.

---

## Creating a ParticleCloud

```python
from particlefilterbox.core import ParticleCloud
import numpy as np

# Create a cloud with 1000 particles in 2D state space
cloud = ParticleCloud(n_particles=1000, k_states=2)

print(cloud)
# ParticleCloud(n_particles=1000, k_states=2)
```

On creation, particles are initialized to zero and weights are set to uniform ($1/N$).

```python
print(cloud.particles.shape)   # (1000, 2)
print(cloud.log_weights.shape) # (1000,)
print(cloud.normalized_weights[:5])
# [0.001, 0.001, 0.001, 0.001, 0.001]
```

### Setting Initial Particles

Typically, particles are initialized from a prior distribution:

```python
rng = np.random.default_rng(42)

# Sample from a prior: x_0 ~ N(mu_0, Sigma_0)
mu_0 = np.array([0.0, 1.0])
sigma_0 = np.array([[1.0, 0.0], [0.0, 0.5]])
cloud.particles = rng.multivariate_normal(mu_0, sigma_0, size=1000)
```

---

## Properties

### Particles and Dimensions

| Property | Type | Description |
|----------|------|-------------|
| `n_particles` | `int` | Number of particles $N$ |
| `k_states` | `int` | State-space dimension $k$ |
| `particles` | `ndarray (N, k)` | Particle positions |
| `ancestors` | `ndarray (N,)` | Ancestor indices from last resampling |

### Weights

Weights are stored in log-space to avoid numerical underflow. The normalized weights are computed on-the-fly using the log-sum-exp trick.

| Property | Type | Description |
|----------|------|-------------|
| `log_weights` | `ndarray (N,)` | Unnormalized log-weights $\log \tilde{w}^{(i)}$ |
| `normalized_weights` | `ndarray (N,)` | Normalized weights $w^{(i)} = \tilde{w}^{(i)} / \sum_j \tilde{w}^{(j)}$ |
| `ess` | `float` | Effective Sample Size (see [ESS](ess.md)) |
| `log_likelihood_increment` | `float` | $\log \hat{p}(y_t \mid y_{1:t-1})$ |

!!! note "Log-space arithmetic"
    Particle weights can span many orders of magnitude. Storing and manipulating them in log-space prevents underflow when $w^{(i)} \approx 0$ for most particles. Normalization uses the identity:

    $$
    w^{(i)} = \frac{\exp(\log \tilde{w}^{(i)})}{\sum_j \exp(\log \tilde{w}^{(j)})} = \text{softmax}(\log \tilde{w})_i
    $$

---

## Weight Manipulation

### Setting Weights

```python
# Reset to uniform weights
cloud.set_uniform_weights()

# Set log-weights directly
log_w = -np.abs(rng.standard_normal(1000))
cloud.set_log_weights(log_w)

print(f"Sum of normalized weights: {cloud.normalized_weights.sum():.6f}")
# Sum of normalized weights: 1.000000
```

### Incrementing Weights

During filtering, weights are updated by adding log-likelihood increments:

$$
\log \tilde{w}_t^{(i)} = \log \tilde{w}_{t-1}^{(i)} + \log p(y_t \mid x_t^{(i)})
$$

```python
# Simulate log-likelihood values
log_likelihoods = rng.standard_normal(1000) * 2

# Update weights (adds to existing log-weights)
cloud.add_log_weights(log_likelihoods)

print(f"ESS: {cloud.ess:.1f} / {cloud.n_particles}")
# ESS: 287.3 / 1000
```

---

## Weighted Statistics

`ParticleCloud` computes statistics weighted by the current normalized weights.

### Mean

The weighted mean provides the point estimate of the state:

$$
\hat{x}_t = \sum_{i=1}^{N} w_t^{(i)} \, x_t^{(i)}
$$

```python
mean = cloud.weighted_mean()
print(f"Weighted mean: {mean}")
# Weighted mean: [0.0312  1.0087]
```

### Covariance

The weighted covariance matrix quantifies estimation uncertainty:

$$
\hat{\Sigma}_t = \sum_{i=1}^{N} w_t^{(i)} \, (x_t^{(i)} - \hat{x}_t)(x_t^{(i)} - \hat{x}_t)^\top
$$

```python
cov = cloud.weighted_cov()
print(f"Weighted covariance:\n{cov}")
# Weighted covariance:
# [[1.023  0.012]
#  [0.012  0.498]]
```

### Quantiles

Weighted quantiles are useful for credible intervals:

```python
# Median
median = cloud.weighted_quantile(0.5)

# 90% credible interval
q05, q95 = cloud.weighted_quantile([0.05, 0.95])

print(f"Median: {median}")
print(f"90% CI: [{q05}, {q95}]")
```

---

## Resampling

When weights become degenerate (few particles carry most of the weight), resampling redistributes particles. See [Resampling](resampling.md) for algorithm details.

```python
from particlefilterbox.resampling import systematic_resample

# Check if resampling is needed
if cloud.ess < 0.5 * cloud.n_particles:
    indices = systematic_resample(cloud.normalized_weights)
    cloud.resample(indices)
    
    print(f"After resampling - ESS: {cloud.ess:.1f}")
    # After resampling - ESS: 1000.0  (weights are now uniform)
```

After resampling:

- Particles are rearranged according to `indices`
- Weights are reset to uniform ($\log w^{(i)} = -\log N$)
- `ancestors` stores the resampling indices for genealogy tracking

---

## Cloning

Create an independent deep copy:

```python
cloud_copy = cloud.clone()

# Modify the copy without affecting the original
cloud_copy.particles[0] = [999.0, 999.0]
print(cloud.particles[0])  # Original unchanged
```

!!! tip "When to clone"
    Clone before resampling if you need to preserve the pre-resampling state (e.g., for smoothing algorithms like FFBSi that need the full particle history).

---

## Complete Example

A full example showing `ParticleCloud` in a manual filtering step:

```python
import numpy as np
from particlefilterbox.core import ParticleCloud
from particlefilterbox.resampling import systematic_resample

rng = np.random.default_rng(42)

# --- Setup ---
N = 2000
cloud = ParticleCloud(n_particles=N, k_states=1)

# Initialize from prior: x_0 ~ N(0, 1)
cloud.particles = rng.standard_normal((N, 1))

# --- Predict step: x_t = 0.9 * x_{t-1} + noise ---
phi = 0.9
sigma_x = 0.5
cloud.particles = phi * cloud.particles + sigma_x * rng.standard_normal((N, 1))

# --- Update step: y_t ~ N(x_t, 0.1) ---
y_obs = 1.5  # observed value
sigma_y = np.sqrt(0.1)
log_lik = -0.5 * ((y_obs - cloud.particles[:, 0]) / sigma_y) ** 2
cloud.add_log_weights(log_lik)

print(f"Before resampling:")
print(f"  ESS: {cloud.ess:.1f} / {N}")
print(f"  Weighted mean: {cloud.weighted_mean()[0]:.4f}")
print(f"  Log-likelihood increment: {cloud.log_likelihood_increment:.4f}")

# --- Resample ---
if cloud.ess < 0.5 * N:
    indices = systematic_resample(cloud.normalized_weights)
    cloud.resample(indices)

print(f"\nAfter resampling:")
print(f"  ESS: {cloud.ess:.1f} / {N}")
print(f"  Weighted mean: {cloud.weighted_mean()[0]:.4f}")
```

Expected output:

```
Before resampling:
  ESS: 423.7 / 2000
  Weighted mean: 1.1284
  Log-likelihood increment: -3.2156

After resampling:
  ESS: 2000.0 / 2000
  Weighted mean: 1.1284
```

---

## Memory Management for Large $N$

For large particle counts ($N > 10^5$), keep in mind:

!!! warning "Memory usage"
    A `ParticleCloud` with $N$ particles in $k$ dimensions uses approximately $8 \times N \times (k + 1)$ bytes (64-bit floats for particles plus log-weights). For example:

    | $N$ | $k$ | Memory |
    |-----|-----|--------|
    | 10,000 | 2 | ~234 KB |
    | 100,000 | 2 | ~2.3 MB |
    | 1,000,000 | 10 | ~84 MB |
    | 10,000,000 | 10 | ~838 MB |

**Tips for large-scale runs:**

- Use `PFConfig(store_particles=False)` to avoid storing the full $T \times N \times k$ history
- Prefer systematic or stratified resampling ($O(N)$) over multinomial ($O(N \log N)$)
- Consider the [Acceleration](../../acceleration/index.md) module for Numba JIT and GPU support

---

## See Also

- [Resampling](resampling.md) --- algorithms for redistributing particles
- [ESS](ess.md) --- monitoring weight degeneracy
- [API Reference: Core](../../api/core.md) --- full API documentation
- [Core Concepts](../../getting-started/core-concepts.md) --- introductory overview
