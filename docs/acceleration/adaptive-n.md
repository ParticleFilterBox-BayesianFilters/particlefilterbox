# Adaptive Number of Particles

## Overview

Standard particle filters use a **fixed** number of particles $N$ throughout the entire time series. This is wasteful: at some time steps the filter is well-concentrated and few particles suffice, while at others (e.g., regime changes, outliers) many more are needed to maintain accuracy.

**Adaptive $N$** dynamically adjusts the particle count at each time step based on filtering quality metrics --- primarily the **Effective Sample Size (ESS)**. This delivers the **same accuracy as a large fixed $N$** at a fraction of the computational cost.

```python
from particlefilterbox import BootstrapFilter

bpf = BootstrapFilter(
    model,
    n_particles='adaptive',
    n_min=100,
    n_max=10000,
    ess_target=0.7,
)
result = bpf.filter(observations)
```

---

## The Problem with Fixed $N$

Consider a stochastic volatility model with $T = 1\,000$ observations. A fixed $N = 10\,000$ is needed to handle the most challenging time steps, but for 80% of the series, $N = 500$ would be perfectly adequate:

```
ESS / N ratio over time (fixed N = 10,000)
  1.0 ┤ ●●●●●●●●●●                    ●●●●●●●●
      │           ●●●●●            ●●●●
  0.7 ┤── ── ── ── ── ── ── target ── ── ── ── ──
      │                ●●●●    ●●●
  0.4 ┤                    ●●
      │                      ●  ← Regime change
  0.1 ┤
      ┼──────────────────────────────────────────
      0         250        500        750       1000
                          Time
```

With fixed $N$, you pay for 10,000 particles at **every** time step even when 500 would suffice.

---

## Adaptive Algorithm

### Core idea

At each time step $t$, after resampling:

1. **Compute ESS**: $\text{ESS}_t = \left(\sum_{i=1}^{N_t} (w_t^{(i)})^2\right)^{-1}$
2. **Compare to target**: If $\text{ESS}_t / N_t < \rho_{\text{target}}$, increase $N_{t+1}$; if above, decrease.
3. **Clamp**: $N_{t+1} = \text{clamp}(N_{t+1}, N_{\min}, N_{\max})$

### Adaptation rule

The default adaptation uses a **multiplicative controller**:

$$
N_{t+1} = \text{clamp}\!\left( N_t \cdot \left(\frac{\rho_{\text{target}}}{\text{ESS}_t / N_t}\right)^\alpha, \; N_{\min}, \; N_{\max} \right)
$$

where $\alpha \in (0, 1]$ is a **smoothing parameter** that prevents oscillations. Default: $\alpha = 0.5$.

!!! info "Intuition"
    When ESS ratio drops below target, the fraction $\rho_{\text{target}} / (\text{ESS}_t / N_t) > 1$, so $N$ increases. When ESS ratio is above target, $N$ decreases. The exponent $\alpha$ dampens the response to avoid over-reaction.

---

## API

### Basic adaptive filtering

```python
from particlefilterbox import BootstrapFilter

bpf = BootstrapFilter(
    model,
    n_particles='adaptive',
    n_min=100,           # Minimum particle count
    n_max=10000,         # Maximum particle count
    ess_target=0.7,      # Target ESS/N ratio
)
result = bpf.filter(observations)

# Inspect particle count over time
print(result.n_particles_history)   # Array of N_t values
print(f"Mean N: {result.n_particles_history.mean():.0f}")
print(f"Max  N: {result.n_particles_history.max()}")
```

### With custom adaptation parameters

```python
bpf = BootstrapFilter(
    model,
    n_particles='adaptive',
    n_min=200,
    n_max=50000,
    ess_target=0.5,
    adapt_smoothing=0.3,     # Lower α = smoother N trajectory
    adapt_strategy='multiplicative',  # or 'additive'
)
```

### With any backend

```python
# Adaptive N works with all backends
bpf = BootstrapFilter(
    model,
    n_particles='adaptive',
    n_min=500,
    n_max=100000,
    ess_target=0.7,
    backend='numba',     # or 'cupy', 'jax'
)
```

---

## Alive Particle Filter

The **alive particle filter** (APF variant) takes a different approach: instead of adjusting $N$ globally, it tracks which particles are "alive" (have non-negligible weight) and removes dead particles at each step.

```python
from particlefilterbox import AliveParticleFilter

apf = AliveParticleFilter(
    model,
    n_particles=10000,
    alive_threshold=1e-8,    # Minimum weight to be "alive"
)
result = apf.filter(observations)

# Track alive particle counts
print(result.n_alive_history)   # Array of alive counts per step
```

### How it works

At each time step:

1. Propagate all $N$ particles.
2. Compute weights.
3. **Remove** particles with $w_t^{(i)} < \epsilon_{\text{alive}}$.
4. Renormalise weights of surviving particles.
5. Optionally **replenish** from high-weight particles if alive count drops below $N_{\min}$.

$$
N_t^{\text{alive}} = \#\{i : w_t^{(i)} \geq \epsilon_{\text{alive}}\}
$$

!!! tip "When to use"
    The alive particle filter is particularly effective for models with **multi-modal** posteriors or **heavy-tailed** observation distributions, where many particles can land in low-probability regions.

---

## Budget Allocation

For applications with a **fixed total computational budget**, particlefilterbox can optimise the distribution of particles across time steps:

### Fixed-budget adaptive filtering

```python
bpf = BootstrapFilter(
    model,
    n_particles='adaptive',
    total_budget=500000,     # Total particle-steps budget
    ess_target=0.7,
)
result = bpf.filter(observations)

print(f"Total particles used: {result.n_particles_history.sum()}")
# ≈ 500,000 ± small overshoot
```

### How budget allocation works

Given a budget $B$ and $T$ time steps:

1. **Initial allocation**: $N_1 = B / T$ (uniform).
2. **Re-allocation**: After observing ESS at each step, shift budget from "easy" steps to "hard" steps.
3. **Look-ahead** (optional): Use pilot run with small $N$ to pre-estimate difficulty across the series.

$$
N_t^* = \frac{B \cdot d_t}{\sum_{s=1}^T d_s}
$$

where $d_t$ is the estimated **difficulty** at time $t$ (inverse of ESS ratio from the pilot run).

```python
bpf = BootstrapFilter(
    model,
    n_particles='adaptive',
    total_budget=500000,
    budget_strategy='pilot',     # Use pilot run for allocation
    pilot_n=200,                 # Pilot run with 200 particles
)
result = bpf.filter(observations)
```

!!! abstract "Budget strategies"
    | Strategy | Description | Overhead |
    |----------|-------------|----------|
    | `'online'` | Adapt step-by-step based on current ESS (default) | None |
    | `'pilot'` | Pre-allocate using a cheap pilot run | 1 extra pass |
    | `'uniform'` | Fixed $N = B/T$ at all steps (baseline) | None |

---

## Benchmarks

### Adaptive vs Fixed (Stochastic Volatility, $T = 1\,000$)

| Method | Mean $N$ | Total Particles | RMSE | Time (s) |
|--------|:--------:|:--------------:|:----:|:--------:|
| Fixed $N = 500$ | 500 | 500,000 | 0.18 | 0.60 |
| Fixed $N = 5,000$ | 5,000 | 5,000,000 | 0.04 | 5.80 |
| Fixed $N = 10,000$ | 10,000 | 10,000,000 | 0.03 | 12.0 |
| **Adaptive** ($N_{\min}=100$, $N_{\max}=10\,000$) | **1,200** | **1,200,000** | **0.035** | **1.45** |

!!! tip "Key insight"
    Adaptive filtering achieves accuracy comparable to $N = 10\,000$ fixed while using only **12%** of the total particle budget --- an **8× cost reduction**.

### Budget allocation comparison

Fixed budget $B = 1\,000\,000$ particle-steps, $T = 1\,000$:

| Strategy | RMSE | Max ESS Drop | Failures |
|----------|:----:|:------------:|:--------:|
| Uniform ($N = 1\,000$) | 0.12 | 0.15 | 3 |
| Online adaptive | 0.06 | 0.45 | 0 |
| Pilot-based | 0.05 | 0.52 | 0 |

### Scaling with $T$

| $T$ | Fixed ($N=5\,000$) | Adaptive (mean $N$) | Speedup |
|:---:|:------------------:|:-------------------:|:-------:|
| 100 | 0.5 s | 0.12 s (mean $N$=850) | 4.2× |
| 500 | 2.9 s | 0.55 s (mean $N$=920) | 5.3× |
| 1,000 | 5.8 s | 1.05 s (mean $N$=1,100) | 5.5× |
| 5,000 | 29.0 s | 5.80 s (mean $N$=1,250) | 5.0× |

---

## Trade-offs and Limitations

### Advantages

- **Significant cost reduction** (3--10× fewer particles on average).
- **Same or better accuracy** vs fixed $N$ at difficult time steps.
- **Automatic** --- no manual tuning of $N$ per dataset.
- **Compatible** with all backends and filter types.

### Limitations

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Overhead per step** | ESS computation + adaptation logic adds ~5% overhead | Negligible for $N > 100$ |
| **Memory pre-allocation** | Must allocate arrays for $N_{\max}$ | Set reasonable $N_{\max}$ |
| **Log-likelihood estimation** | Variable $N_t$ complicates unbiased $\hat{p}(y_{1:T})$ | Use corrected estimator (automatic) |
| **Reproducibility** | Different seeds may yield different $N_t$ trajectories | Set seed for reproducibility |
| **Not ideal for PMMH** | PMMH requires unbiased likelihood; variable $N$ needs correction | Use fixed $N$ for PMMH unless correction is enabled |

!!! warning "PMMH compatibility"
    When using adaptive $N$ with PMMH, enable the **bias correction** to maintain the correct stationary distribution:

    ```python
    pmmh = PMMH(
        model, n_particles='adaptive',
        n_min=500, n_max=5000,
        adaptive_correction=True,   # Required for valid MCMC
    )
    ```

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` or `str` | --- | Set to `'adaptive'` for dynamic $N$ |
| `n_min` | `int` | `100` | Minimum number of particles |
| `n_max` | `int` | `10000` | Maximum number of particles |
| `ess_target` | `float` | `0.7` | Target ESS/N ratio in $[0, 1]$ |
| `adapt_smoothing` | `float` | `0.5` | Smoothing parameter $\alpha \in (0, 1]$ |
| `adapt_strategy` | `str` | `'multiplicative'` | `'multiplicative'` or `'additive'` |
| `total_budget` | `int` | `None` | Fixed total particle-step budget |
| `budget_strategy` | `str` | `'online'` | `'online'`, `'pilot'`, or `'uniform'` |
| `pilot_n` | `int` | `200` | Particles for pilot run (if `budget_strategy='pilot'`) |
| `alive_threshold` | `float` | `1e-8` | Minimum weight for alive particle filter |
| `adaptive_correction` | `bool` | `False` | Bias correction for PMMH compatibility |

---

## See Also

- [Acceleration Overview](index.md) --- Backend comparison
- [Numba JIT](numba.md) --- Combine adaptive $N$ with Numba for fast CPU inference
- [Parallel Execution](parallel.md) --- Run adaptive filter replicas in parallel
- [ESS Diagnostic](../diagnostics/ess-diagnostic.md) --- understanding ESS, the metric that drives adaptive $N$
- [Weight Diagnostic](../diagnostics/weight-diagnostic.md) --- verify that adaptive $N$ maintains weight quality
- [Convergence Diagnostic](../diagnostics/convergence.md) --- N-study to set $N_{\min}$ and $N_{\max}$ bounds
- [PMMH](../user-guide/pmcmc/pmmh.md) --- adaptive $N$ with PMMH requires bias correction
- [Filters Overview](../user-guide/filters/index.md) --- all filter variants support adaptive $N$
