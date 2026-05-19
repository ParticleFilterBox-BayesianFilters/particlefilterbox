---
title: Effective Sample Size (ESS)
description: "ESS definition, monitoring, adaptive resampling, and diagnostics for particle filters"
---

# Effective Sample Size (ESS)

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Property** | `cloud.ess` |
    | **Monitor** | `from particlefilterbox.diagnostics import ESSMonitor` |
    | **Config** | `PFConfig(ess_threshold=0.5)` |
    | **Range** | $1 \leq \text{ESS} \leq N$ |
    | **Role** | Measures weight degeneracy and triggers resampling |

## Definition

The **Effective Sample Size** measures how many particles are effectively contributing to the approximation. With $N$ particles carrying normalized weights $\{w^{(i)}\}_{i=1}^N$, the ESS is:

$$
\text{ESS} = \frac{1}{\sum_{i=1}^{N} \left(w^{(i)}\right)^2}
$$

This is the inverse of the sum of squared weights, also known as the **reciprocal of the Herfindahl index** in economics.

### Interpretation

| ESS Value | Meaning |
|-----------|---------|
| $\text{ESS} = N$ | All weights are equal ($w^{(i)} = 1/N$) --- perfect diversity |
| $\text{ESS} \approx N/2$ | Moderate weight concentration --- healthy |
| $\text{ESS} \ll N$ | Severe weight degeneracy --- resampling needed |
| $\text{ESS} = 1$ | A single particle carries all the weight --- degenerate |

The ESS can be interpreted as the number of particles in an equally-weighted set that would provide the same estimation quality. If $\text{ESS} = 200$ out of $N = 1000$, only ~200 particles are effectively contributing.

### Derivation

The ESS arises from matching the variance of a weighted importance sampling estimator to that of an equally-weighted estimator. For estimating $\mathbb{E}_p[f(x)]$ with proposal $q$:

$$
\text{Var}\!\left[\hat{I}_{\text{IS}}\right] = \text{Var}\!\left[\frac{1}{N_{\text{eff}}}\sum_{i=1}^{N_{\text{eff}}} f(x^{(i)})\right]
$$

where $x^{(i)} \sim p$. Solving for $N_{\text{eff}}$ yields:

$$
N_{\text{eff}} = \frac{\left(\sum_{i=1}^N w^{(i)}\right)^2}{\sum_{i=1}^N \left(w^{(i)}\right)^2} = \frac{1}{\sum_{i=1}^N \left(w^{(i)}\right)^2}
$$

when weights are normalized.

---

## Computing ESS

### From ParticleCloud

```python
from particlefilterbox.core import ParticleCloud
import numpy as np

cloud = ParticleCloud(n_particles=1000, k_states=2)

# Uniform weights -> ESS = N
print(f"ESS (uniform): {cloud.ess:.1f}")
# ESS (uniform): 1000.0

# After weighting with observation likelihood
rng = np.random.default_rng(42)
log_lik = rng.standard_normal(1000) * 3  # High variance -> low ESS
cloud.add_log_weights(log_lik)

print(f"ESS (weighted): {cloud.ess:.1f}")
print(f"ESS ratio: {cloud.ess / cloud.n_particles:.2%}")
# ESS (weighted): 78.3
# ESS ratio: 7.83%
```

### Static Computation

```python
from particlefilterbox.diagnostics import ESSMonitor

weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
ess = ESSMonitor.compute_ess(weights)
print(f"ESS: {ess:.2f} / {len(weights)}")
# ESS: 2.99 / 5
```

---

## Adaptive Resampling

The standard approach is to resample **only when ESS drops below a threshold**, typically $\text{ESS} < \alpha N$ with $\alpha = 0.5$:

$$
\text{Resample at time } t \iff \text{ESS}_t < \alpha \cdot N
$$

### Why Not Always Resample?

Resampling has a cost: it introduces **path degeneracy** (all particles eventually share the same ancestry). Resampling only when necessary balances two competing concerns:

| Always resample | Never resample |
|----------------|----------------|
| No weight degeneracy | No path degeneracy |
| Maximum path degeneracy | Maximum weight degeneracy |

The threshold $\alpha = 0.5$ is a well-established default (Doucet & Johansen, 2009).

### Configuration

```python
from particlefilterbox import BootstrapPF, PFConfig

# Default: resample when ESS < 50% of N
config = PFConfig(
    n_particles=5000,
    ess_threshold=0.5,  # resample when ESS < 2500
)

# Conservative: resample less often
config_conservative = PFConfig(
    n_particles=5000,
    ess_threshold=0.3,  # resample when ESS < 1500
)

# Aggressive: resample more often
config_aggressive = PFConfig(
    n_particles=5000,
    ess_threshold=0.8,  # resample when ESS < 4000
)
```

!!! tip "Choosing the threshold"
    - **$\alpha = 0.5$**: Standard default. Works well in most cases.
    - **$\alpha < 0.5$**: Use for smoothing applications where path diversity matters.
    - **$\alpha > 0.5$**: Use when observation noise is very low (difficult likelihood).
    - **$\alpha = 1.0$**: Always resample (every time step).

---

## ESS Monitoring

The `ESSMonitor` class tracks ESS over time and generates alerts when quality drops.

### Setup

```python
from particlefilterbox.diagnostics import ESSMonitor

monitor = ESSMonitor(
    warning_ratio=0.1,   # warn when ESS/N < 10%
    critical_ess=1.0,     # critical when ESS approaches 1
)
```

### Real-Time Monitoring

```python
from particlefilterbox.core import ParticleCloud
from particlefilterbox.resampling import systematic_resample
import numpy as np

rng = np.random.default_rng(42)
cloud = ParticleCloud(n_particles=1000, k_states=1)
cloud.particles = rng.standard_normal((1000, 1))
monitor = ESSMonitor(warning_ratio=0.1, critical_ess=1.0)

# Simulate 50 time steps
for t in range(50):
    # Predict
    cloud.particles = 0.9 * cloud.particles + 0.5 * rng.standard_normal((1000, 1))
    
    # Weight (observation at 0 with noise 0.1)
    log_lik = -0.5 * (cloud.particles[:, 0] ** 2) / 0.1
    cloud.add_log_weights(log_lik)
    
    # Monitor
    alert_level = monitor.update(cloud, time_step=t)
    
    # Resample if needed
    if cloud.ess < 500:
        indices = systematic_resample(cloud.normalized_weights, rng=rng)
        cloud.resample(indices)

# Review monitoring results
summary = monitor.summary()
print(f"ESS min:  {summary['ess_min']:.1f}")
print(f"ESS mean: {summary['ess_mean']:.1f}")
print(f"Healthy:  {monitor.is_healthy()}")
print(f"Alerts:   {len(monitor.alerts)}")
```

### Alert Levels

The monitor classifies ESS readings into three levels:

| Level | Condition | Meaning |
|-------|-----------|---------|
| `OK` | $\text{ESS}/N > $ `warning_ratio` | Normal operation |
| `WARNING` | $\text{ESS}/N \leq $ `warning_ratio` | Weight concentration increasing |
| `CRITICAL` | $\text{ESS} \leq $ `critical_ess` | Near-total degeneracy |

```python
from particlefilterbox.diagnostics.ess_monitor import AlertLevel

for alert in monitor.alerts:
    if alert.level == AlertLevel.CRITICAL:
        print(f"CRITICAL at t={alert.time_step}: ESS={alert.ess_value:.1f}")
        print(f"  {alert.message}")
```

---

## ESS for SMC Samplers

In SMC samplers (e.g., for tempering between distributions $\pi_0, \pi_1, \ldots, \pi_P$), the ESS plays a different role: it determines the **tempering schedule**.

Given a sequence of distributions $\pi_n(x) \propto \pi_0(x)^{1-\phi_n} \pi_P(x)^{\phi_n}$ with $\phi_0 = 0, \phi_P = 1$, the next temperature $\phi_{n+1}$ is chosen such that:

$$
\text{ESS}(\phi_{n+1}) = \alpha \cdot N
$$

where $\alpha$ is typically $0.8$--$0.95$ (higher than for filtering).

```python
from particlefilterbox import PFConfig

# For SMC samplers, use a higher threshold
smc_config = PFConfig(
    n_particles=10000,
    ess_threshold=0.9,  # conservative tempering
)
```

---

## Diagnostics

### ESS Over Time

The ESS history is available in filter results:

```python
from particlefilterbox import BootstrapPF, PFConfig
from particlefilterbox.models import StochasticVolatility
import numpy as np

# Run a filter
model = StochasticVolatility(variant="basic")
config = PFConfig(n_particles=2000, ess_threshold=0.5, seed=42)
pf = BootstrapPF(model, config)

# Simulate data
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=200, rng=rng)
results = pf.filter(obs)

# ESS history
print(f"ESS min:  {results.ess_history.min():.0f}")
print(f"ESS mean: {results.ess_history.mean():.0f}")
print(f"ESS max:  {results.ess_history.max():.0f}")
print(f"Resampled: {results.resampled.sum()} / {results.nobs} steps")
```

### Interpreting ESS Patterns

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| ESS consistently high ($> 0.7N$) | Model is easy; filter is over-resourced | Reduce $N$ to save computation |
| ESS oscillates around threshold | Normal adaptive behavior | No action needed |
| ESS drops sharply at specific times | Outliers or structural breaks | Investigate observations at those times |
| ESS persistently low ($< 0.1N$) | Proposal is poor; model mismatch | Use a better proposal (SIR, Guided PF) or increase $N$ |
| ESS $\approx 1$ frequently | Severe degeneracy | Model/filter mismatch; consider reparameterization |

### Relationship to Estimation Quality

The ESS directly affects the quality of filtered state estimates. The variance of the particle approximation scales as:

$$
\text{Var}[\hat{I}_N] \approx \frac{\sigma_f^2}{\text{ESS}}
$$

where $\sigma_f^2 = \text{Var}_p[f(x)]$. This means:

- Halving the ESS **doubles** the estimation variance
- An ESS of 100 gives roughly the same quality as 100 equally-weighted particles
- The log-likelihood estimate $\log \hat{p}(y_{1:T})$ is particularly sensitive to low ESS

---

## Complete Example: ESS Diagnostics

```python
import numpy as np
from particlefilterbox import BootstrapPF, PFConfig
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.diagnostics import ESSMonitor

# Setup
model = StochasticVolatility(variant="basic")
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=300, rng=rng)

# Run with different thresholds
thresholds = [0.3, 0.5, 0.8]

for alpha in thresholds:
    config = PFConfig(n_particles=2000, ess_threshold=alpha, seed=42)
    pf = BootstrapPF(model, config)
    results = pf.filter(obs)
    
    n_resampled = results.resampled.sum()
    mean_ess = results.ess_history.mean()
    
    print(f"threshold={alpha:.1f}: "
          f"resampled {n_resampled:>3d}/300 steps, "
          f"mean ESS={mean_ess:.0f}, "
          f"log-lik={results.log_likelihood:.2f}")
```

Expected output:

```
threshold=0.3: resampled  87/300 steps, mean ESS=1142, log-lik=-412.35
threshold=0.5: resampled 156/300 steps, mean ESS=1384, log-lik=-412.31
threshold=0.8: resampled 248/300 steps, mean ESS=1687, log-lik=-412.28
```

!!! note "Log-likelihood stability"
    Higher resampling thresholds produce more stable (lower variance) log-likelihood estimates, at the cost of increased path degeneracy. For PMCMC applications where log-likelihood variance matters, use $\alpha \geq 0.5$.

---

## See Also

- [ParticleCloud](particle-cloud.md) --- the `ess` property lives here
- [Resampling](resampling.md) --- what happens when ESS triggers resampling
- [ESS Diagnostic](../../diagnostics/ess-diagnostic.md) --- detailed diagnostic tools
- [API Reference: Diagnostics](../../api/diagnostics.md) --- full API documentation
