---
title: Regularized Particle Filter
description: "The Regularized PF — kernel smoothing to combat sample impoverishment"
---

# Regularized Particle Filter

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `RegularizedPF` |
    | **Import** | `from particlefilterbox.filters import RegularizedPF` |
    | **Key idea** | Apply kernel jittering after resampling to maintain particle diversity |
    | **Complexity** | $O(N)$ per time step |
    | **Reference** | Musso, Oudjane & Le Gland (2001) |

## Overview

The Regularized Particle Filter (RPF) addresses a fundamental problem of standard particle filters: **sample impoverishment**. After resampling, many particles collapse to the same values, reducing the effective support of the approximation. The RPF restores diversity by applying **kernel smoothing** — after resampling, each particle is jittered by a small random perturbation drawn from a kernel density estimate of the current particle distribution.

This is especially important for:

- **Continuous state spaces** where the posterior is smooth
- **Long time series** where repeated resampling erodes diversity
- **Parameter estimation** where fixed parameters would collapse to a point mass

**Advantages:**

- Simple modification to any existing particle filter
- Prevents particle collapse for continuous distributions
- Effective for multimodal posteriors — kernel can maintain distinct modes
- Same $O(N)$ complexity as Bootstrap

**Disadvantages:**

- Introduces a bias (the kernel oversmooths the true posterior)
- Bandwidth selection is nontrivial — too large = over-smoothed, too small = no effect
- Not appropriate for discrete or mixed-type state spaces
- The jittering can push particles into low-likelihood regions

---

## Algorithm

The Regularized PF modifies the resampling step by adding kernel noise:

$$
\boxed{
\begin{aligned}
&\textbf{Regularized Particle Filter} \\[6pt]
&\text{1. } \textbf{Initialize: } x_0^{(i)} \sim p(x_0), \quad w_0^{(i)} = \tfrac{1}{N} \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T: \\
&\qquad \text{a. } \textbf{Propagate: } x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)}) \\
&\qquad \text{b. } \textbf{Weight: } \tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot p(y_t \mid x_t^{(i)}) \\
&\qquad \text{c. } \textbf{Normalize: } w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_j \tilde{w}_t^{(j)}} \\
&\qquad \text{d. } \textbf{Resample: } \text{If } \text{ESS} < \tau \cdot N: \\
&\qquad \qquad \text{(i) Resample indices } \{a_t^{(i)}\} \\
&\qquad \qquad \text{(ii) } \textbf{Regularize: } x_t^{(i)} \leftarrow x_t^{(a_t^{(i)})} + h_t \cdot \epsilon^{(i)}, \quad \epsilon^{(i)} \sim K \\
&\qquad \qquad \text{(iii) Set } w_t^{(i)} = \tfrac{1}{N}
\end{aligned}
}
$$

where $K$ is a kernel function and $h_t$ is the bandwidth.

### Kernel Density Estimation

The regularized posterior at time $t$ is approximated by:

$$
\hat{p}(x_t \mid y_{1:t}) = \sum_{i=1}^{N} w_t^{(i)} \cdot \frac{1}{h_t^k} K\!\left(\frac{x_t - x_t^{(i)}}{h_t}\right)
$$

After resampling, we draw from this KDE rather than using point masses.

### Kernel Functions

The RPF supports several kernel shapes:

=== "Epanechnikov"

    $$K(u) = \frac{k+2}{2 c_k} (1 - \|u\|^2) \cdot \mathbf{1}(\|u\| \leq 1)$$

    - Optimal in the MSE sense for density estimation
    - Compact support — jittering is bounded
    - Default choice in particlefilterbox

=== "Gaussian"

    $$K(u) = (2\pi)^{-k/2} \exp\!\left(-\tfrac{1}{2} \|u\|^2\right)$$

    - Smooth, unbounded support
    - More aggressive smoothing
    - Useful when the posterior is very smooth

=== "Uniform"

    $$K(u) = \frac{1}{c_k} \cdot \mathbf{1}(\|u\| \leq 1)$$

    - Simplest kernel
    - Uniform jittering within a hypersphere
    - Fast to sample

### Bandwidth Selection

The bandwidth $h_t$ controls how much jittering is applied. Too large and the filter oversmooths; too small and particles remain degenerate.

**Silverman's rule** (default):

$$
h_t = \left(\frac{4}{N(k+2)}\right)^{1/(k+4)} \cdot \hat{\sigma}_t
$$

where $\hat{\sigma}_t$ is the weighted standard deviation of the particles and $k$ is the state dimension.

**Optimal bandwidth** (Musso et al., 2001):

$$
h_t^{\text{opt}} = A(K, k) \cdot N^{-1/(k+4)}
$$

where $A(K, k)$ depends on the kernel and dimensionality.

---

## API Reference

### Constructor

```python
from particlefilterbox.filters import RegularizedPF
from particlefilterbox.core.config import PFConfig

config = PFConfig(
    n_particles=1000,
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
)

rpf = RegularizedPF(model=my_model, config=config)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `1000` | Number of particles $N$ |
| `resampling` | `str` | `"systematic"` | Resampling scheme |
| `ess_threshold` | `float` | `0.5` | Resample when $\text{ESS} < \tau \cdot N$ |
| `seed` | `int \| None` | `None` | Random seed |
| `kernel` | `str` | `"epanechnikov"` | Kernel function: `"epanechnikov"`, `"gaussian"`, `"uniform"` |
| `bandwidth` | `str \| float` | `"silverman"` | Bandwidth: `"silverman"`, `"optimal"`, or a fixed float value |
| `bandwidth_scale` | `float` | `1.0` | Multiplicative scaling factor for automatic bandwidth |

### Batch Filtering

```python
result = rpf.filter(observations)
```

Returns the same `ParticleFilterResults` as Bootstrap PF. See [Bootstrap PF — Batch Filtering](bootstrap.md#batch-filtering).

---

## Examples

### Example 1: Multimodal Target Distribution

A model where the posterior is bimodal — the Regularized PF maintains both modes while the Bootstrap PF collapses to one.

```python
import numpy as np
from particlefilterbox.filters import RegularizedPF, BootstrapPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class BimodalModel(ParticleFilterModel):
    """
    x_t = x_{t-1} + eta_t,    eta_t ~ N(0, 0.5^2)
    y_t ~ 0.5 * N(x_t - 3, 0.1^2) + 0.5 * N(x_t + 3, 0.1^2)

    The observation likelihood is bimodal, creating two clusters
    of plausible states for each observation.
    """
    k_states = 1
    k_obs = 1

    def initial_distribution(self, n_particles, rng):
        return rng.normal(0.0, 2.0, size=(n_particles, 1))

    def transition(self, particles, t, rng):
        return particles + rng.normal(0.0, 0.5, size=particles.shape)

    def log_observation_likelihood(self, particles, y_t, t):
        x = particles[:, 0]
        ll1 = -0.5 * ((y_t[0] - (x - 3)) / 0.1)**2
        ll2 = -0.5 * ((y_t[0] - (x + 3)) / 0.1)**2
        # log-sum-exp for mixture
        max_ll = np.maximum(ll1, ll2)
        return max_ll + np.log(np.exp(ll1 - max_ll) + np.exp(ll2 - max_ll)) - np.log(2)

# --- Simulate ---
rng = np.random.default_rng(123)
T = 100
x_true = np.zeros(T)
y_obs = np.zeros(T)

x_true[0] = 0.0
for t in range(T):
    if t > 0:
        x_true[t] = x_true[t - 1] + rng.normal(0.0, 0.5)
    # Observation from one of the two modes
    mode = rng.choice([-3, 3])
    y_obs[t] = x_true[t] + mode + rng.normal(0.0, 0.1)

# --- Compare ---
config = PFConfig(n_particles=2000, resampling="systematic", seed=42)

rpf = RegularizedPF(model=BimodalModel(), config=config, kernel="epanechnikov")
bpf = BootstrapPF(model=BimodalModel(), config=config)

result_rpf = rpf.filter(y_obs)
result_bpf = bpf.filter(y_obs)

print(f"{'Metric':<25} {'Bootstrap':>12} {'Regularized':>12}")
print("-" * 50)
print(f"{'Log-likelihood':<25} {result_bpf.log_likelihood:>12.2f} {result_rpf.log_likelihood:>12.2f}")
print(f"{'Mean ESS':<25} {result_bpf.ess_history.mean():>12.0f} {result_rpf.ess_history.mean():>12.0f}")
```

!!! tip "What to expect"
    The Regularized PF should maintain higher particle diversity and better capture the bimodal structure. Check the particle cloud at various time steps — the RPF should show particles in both modes, while the Bootstrap PF may collapse to a single mode after several resampling events.

### Example 2: Parameter Estimation with Kernel Jittering

When estimating static parameters via particle filtering, standard resampling causes the parameter particles to collapse to a point mass. The RPF prevents this.

```python
import numpy as np
from particlefilterbox.filters import RegularizedPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class UnknownVarianceModel(ParticleFilterModel):
    """
    Estimate observation noise variance alongside the state.
    State: [x_t, log_sigma^2]
    x_t = 0.9 * x_{t-1} + eta_t,   eta_t ~ N(0, 1)
    y_t = x_t + eps_t,              eps_t ~ N(0, exp(log_sigma^2))
    """
    k_states = 2  # [x, log_sigma2]
    k_obs = 1

    def initial_distribution(self, n_particles, rng):
        x = rng.normal(0.0, 1.0, size=(n_particles, 1))
        log_sigma2 = rng.uniform(-2.0, 2.0, size=(n_particles, 1))
        return np.hstack([x, log_sigma2])

    def transition(self, particles, t, rng):
        new = particles.copy()
        new[:, 0] = 0.9 * particles[:, 0] + rng.normal(0.0, 1.0, size=particles.shape[0])
        # Static parameter — no transition (RPF jittering provides diversity)
        return new

    def log_observation_likelihood(self, particles, y_t, t):
        x = particles[:, 0]
        sigma2 = np.exp(particles[:, 1])
        residual = y_t[0] - x
        return -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * residual**2 / sigma2

# --- Simulate ---
rng = np.random.default_rng(456)
T = 200
true_sigma2 = 0.25  # true observation variance

x_true = np.zeros(T)
y_obs = np.zeros(T)
x_true[0] = rng.normal(0, 1)
y_obs[0] = x_true[0] + rng.normal(0, np.sqrt(true_sigma2))
for t in range(1, T):
    x_true[t] = 0.9 * x_true[t-1] + rng.normal(0, 1)
    y_obs[t] = x_true[t] + rng.normal(0, np.sqrt(true_sigma2))

# --- Filter ---
config = PFConfig(n_particles=5000, resampling="systematic", ess_threshold=0.5, seed=42)
rpf = RegularizedPF(
    model=UnknownVarianceModel(),
    config=config,
    kernel="gaussian",
    bandwidth="silverman",
    bandwidth_scale=0.5,
)
result = rpf.filter(y_obs)

# Estimated log(sigma^2)
est_log_sigma2 = result.filtered_means[-1, 1]
print(f"True log(sigma^2): {np.log(true_sigma2):.3f}")
print(f"Estimated log(sigma^2): {est_log_sigma2:.3f}")
print(f"Mean ESS: {result.ess_history.mean():.0f} / {config.n_particles}")
```

!!! tip "What to expect"
    The regularized filter should converge toward $\log(\sigma^2) \approx -1.386$ (i.e., $\sigma^2 = 0.25$). Without kernel jittering, the parameter particles would collapse to a point mass early in the sequence, losing the ability to explore.

---

## Tuning Guide

### Bandwidth Selection

The bandwidth is the most critical tuning parameter:

| Method | Formula | Characteristics |
|--------|---------|----------------|
| **Silverman** (default) | $h = (4/(N(k+2)))^{1/(k+4)} \hat{\sigma}$ | Data-adaptive, conservative |
| **Optimal** | $h = A(K,k) \cdot N^{-1/(k+4)}$ | Minimizes MISE, requires kernel constants |
| **Fixed** | User-specified $h$ | Full control, requires experimentation |

!!! warning "Bandwidth too large"
    If the bandwidth is too large, the filter introduces significant bias — the filtered distribution will be over-smoothed. Monitor the log-likelihood: if it decreases substantially compared to the Bootstrap PF, reduce the bandwidth via `bandwidth_scale`.

### Kernel Selection

| Kernel | Support | Smoothness | Recommendation |
|--------|---------|------------|----------------|
| **Epanechnikov** | Compact | $C^0$ | **Default** — optimal MISE, bounded jittering |
| **Gaussian** | Unbounded | $C^\infty$ | Smooth posteriors, parameter estimation |
| **Uniform** | Compact | $C^{-1}$ | Fast sampling, less smooth |

### When to Use the Regularized PF

| Scenario | Recommendation |
|----------|---------------|
| Continuous state, repeated resampling | **Use RPF** — prevents impoverishment |
| Joint state + parameter estimation | **Use RPF** — jittering prevents parameter collapse |
| Multimodal continuous posterior | **Use RPF** — maintains mode diversity |
| Discrete or integer-valued states | Use [Bootstrap PF](bootstrap.md) — kernel jittering is inappropriate |
| Already using a good proposal (SIR, UPF) | Less benefit from regularization |

### Computational Complexity

| Operation | Cost |
|-----------|------|
| Propagation | $O(N)$ |
| Weighting | $O(N)$ |
| Resampling | $O(N)$ |
| Kernel jittering | $O(N \cdot k)$ — sample from $k$-dimensional kernel |
| **Total per step** | **$O(N \cdot k)$** |

The kernel jittering adds negligible overhead for low-dimensional states.

---

## References

- Musso, C., Oudjane, N. & Le Gland, F. (2001). Improving regularised particle filters. In *Sequential Monte Carlo Methods in Practice*, Springer, 247–271.
- Le Gland, F. & Oudjane, N. (2004). Stability and uniform approximation of nonlinear filters using the Hilbert metric and application to particle filters. *Annals of Applied Probability*, 14(1), 144–187.
- Silverman, B.W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall.
