---
title: Convergence Diagnostic
description: "Particle filter convergence diagnostics: N-study, inter-run variance, and asymptotic variance estimation"
---

# Convergence Diagnostic

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `ConvergenceDiagnostic` |
    | **Import** | `from particlefilterbox.diagnostics import ConvergenceDiagnostic` |
    | **Input** | Model + observations (runs multiple filters internally) |
    | **Key method** | `.n_study(n_values)` |
    | **Goal** | Determine the minimum $N$ for reliable inference |

## Overview

Particle filters are **consistent** --- as $N \to \infty$, the particle approximation converges to the true filtering distribution. But how large does $N$ need to be in practice? The convergence diagnostic answers this question empirically by:

1. Running the filter with increasing particle counts ($N$, $2N$, $4N$, ...)
2. Comparing estimates across runs to measure inter-run variability
3. Estimating the asymptotic variance to predict how estimates improve with $N$

This is the most important diagnostic to run **before** using a particle filter for production inference.

---

## The N-Study

The core tool is the **N-study**: run the same filter with different particle counts and compare the results.

### Basic Usage

```python
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.diagnostics import ConvergenceDiagnostic
import numpy as np

# Setup
model = StochasticVolatility(variant="basic")
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=200, rng=rng)

# Create convergence diagnostic
conv = ConvergenceDiagnostic(model, obs)

# Run N-study
results = conv.n_study(
    n_values=[100, 250, 500, 1000, 2000, 5000],
    n_runs=10,       # independent runs per N value
    seed=42,
)

print(results)
```

```text
=== N-Study Results ===
     N   | Mean log-lik |  Std log-lik | Mean ESS ratio | RMSE vs N=5000
---------+--------------+--------------+----------------+---------------
     100 |    -415.82   |     3.421    |     0.312      |    2.847
     250 |    -413.15   |     1.287    |     0.458      |    1.123
     500 |    -412.56   |     0.634    |     0.587      |    0.521
    1000 |    -412.38   |     0.312    |     0.672      |    0.248
    2000 |    -412.32   |     0.158    |     0.721      |    0.119
    5000 |    -412.29   |     0.063    |     0.784      |    0.000
```

### Convergence Plot

```python
# Plot convergence of filtered state estimates
conv.plot_convergence()

# Customized
conv.plot_convergence(
    metric="log_likelihood",   # or "state_mean", "state_std"
    show_ci=True,              # confidence intervals across runs
    show_rate=True,            # fitted convergence rate
    figsize=(12, 6),
)
```

The convergence plot shows:

- **Point estimates** (mean across runs) at each $N$ value
- **Error bars** (standard deviation across runs)
- **Fitted rate** line: $\text{Var} \propto 1/N$ (the theoretical rate)

---

## Convergence Theory

### Central Limit Theorem for Particle Filters

For a test function $f$ and the particle approximation $\hat{\pi}_t^N$:

$$
\sqrt{N}\left(\hat{\pi}_t^N(f) - \pi_t(f)\right) \xrightarrow{d} \mathcal{N}\!\left(0, \sigma_t^2(f)\right)
$$

where $\sigma_t^2(f)$ is the **asymptotic variance** that depends on the model, the proposal, and the resampling scheme.

This means:

- The error decreases as $O(1/\sqrt{N})$
- To halve the error, you need to **quadruple** $N$
- The asymptotic variance $\sigma_t^2$ determines the constant --- a better proposal reduces it

### Log-Likelihood Convergence

The log-likelihood estimate $\log \hat{p}(y_{1:T})$ converges at the same rate:

$$
\text{Var}\!\left[\log \hat{p}(y_{1:T})\right] = O(1/N)
$$

!!! note "Variance vs. bias"
    The particle filter estimate of $\log p(y_{1:T})$ is **biased downward** --- the expected value is always less than or equal to the true log-likelihood. The bias is $O(1/N)$ and disappears faster than the variance. For PMCMC applications, the bias does not affect the MCMC target distribution (it cancels in the Metropolis-Hastings ratio), but the variance affects mixing.

---

## Inter-Run Variance

Running the filter multiple times with different random seeds reveals the **Monte Carlo variance** of the estimates:

```python
# Run multiple independent filters
variance_report = conv.inter_run_variance(
    n_particles=2000,
    n_runs=20,
    seed=42,
)

print(variance_report)
```

```text
=== Inter-Run Variance (N=2000, 20 runs) ===

Log-likelihood:
  Mean:     -412.32
  Std:       0.158
  Range:    [-412.68, -412.01]

Filtered state (averaged over time):
  Mean RMSE:  0.142
  Std RMSE:   0.008

State mean at t=100:
  Mean:      0.847
  Std:       0.031
  95% CI:   [0.786, 0.908]
```

### Interpreting Inter-Run Variance

| Metric | Acceptable Range | Too High |
|--------|-----------------|----------|
| Std of $\log \hat{p}(y_{1:T})$ | $< 1.0$ | $> 3.0$ |
| Std of filtered state mean | $< 10\%$ of posterior std | $> 50\%$ |
| Max difference across runs | $< 2\sigma$ | $> 5\sigma$ |

!!! tip "For PMCMC applications"
    Pitt et al. (2012) recommend that the standard deviation of the log-likelihood estimate should be between **1.0 and 1.7** for optimal PMMH performance. Use the inter-run variance to calibrate $N$.
    
    ```python
    # Target: std(log-lik) ≈ 1.0-1.7 for PMCMC
    for n in [500, 1000, 2000, 5000]:
        report = conv.inter_run_variance(n_particles=n, n_runs=20, seed=42)
        print(f"N={n:5d}: std(log-lik) = {report['ll_std']:.3f}")
    ```

---

## Asymptotic Variance Estimation

The convergence diagnostic can estimate the asymptotic variance $\sigma^2$ from the N-study data, using the relationship $\text{Var} = \sigma^2 / N$:

```python
# Estimate asymptotic variance
asym = conv.asymptotic_variance(
    n_values=[500, 1000, 2000, 5000],
    n_runs=20,
    seed=42,
)

print(f"Estimated asymptotic variance: {asym['sigma2']:.2f}")
print(f"Predicted std at N=10000: {asym['predict_std'](10000):.4f}")
print(f"N needed for std < 0.1: {asym['n_for_std'](0.1):.0f}")
```

```text
Estimated asymptotic variance: 49.82
Predicted std at N=10000: 0.071
N needed for std < 0.1: 4982
```

### Using the Estimate

The asymptotic variance lets you **predict** performance at particle counts you haven't tested:

$$
\text{Std}[\hat{\theta}(N)] \approx \frac{\sigma}{\sqrt{N}}
$$

```python
# Convergence rate plot with prediction
conv.plot_convergence_rate(
    n_values=[500, 1000, 2000, 5000],
    n_runs=20,
    predict_to=20000,   # extrapolate prediction
    seed=42,
)
```

---

## Practical Rules for Choosing N

!!! tip "When to stop increasing N"
    Increase $N$ until the quantity you care about stabilizes. The right $N$ depends on the application:

| Application | Target | Typical $N$ |
|-------------|--------|-------------|
| State filtering (point estimates) | State std $< 5\%$ of posterior std | 500--2000 |
| State filtering (uncertainty) | ESS ratio $> 0.3$ | 1000--5000 |
| Log-likelihood for PMCMC | $\text{Std}[\log \hat{p}] \approx 1.0\text{--}1.7$ | 500--5000 |
| SMC$^2$ (online) | Low inter-run variance | 1000--10000 |
| Smoothing (fixed-lag) | Unique trajectory ratio $> 0.5$ | 2000--10000 |
| Model comparison (Bayes factors) | $\text{Std}[\log \hat{p}] < 0.5$ | 5000--50000 |

### Diminishing Returns

```python
# Quick check: is doubling N worth it?
conv.diminishing_returns(
    n_values=[500, 1000, 2000, 4000, 8000],
    n_runs=10,
    seed=42,
)
```

```text
=== Diminishing Returns Analysis ===
     N   |   Std(log-lik)  |  Improvement over previous
---------+-----------------+---------------------------
     500 |     0.634       |     ---
    1000 |     0.312       |     50.8% reduction
    2000 |     0.158       |     49.4% reduction
    4000 |     0.081       |     48.7% reduction
    8000 |     0.040       |     50.6% reduction

Convergence rate: O(N^{-0.49}) ≈ theoretical O(N^{-0.50})
```

!!! note "The $\sqrt{N}$ rule"
    Halving the standard deviation requires quadrupling $N$. If you're at $N = 1000$ and the estimates aren't precise enough, going to $N = 2000$ only reduces the std by ~30%. Going to $N = 4000$ reduces it by ~50%. Consider whether a better proposal might be more cost-effective than brute-force particle count increases.

---

## Complete Example

```python
import numpy as np
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.diagnostics import ConvergenceDiagnostic

# Setup
model = StochasticVolatility(variant="basic")
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=200, rng=rng)

# Full convergence analysis
conv = ConvergenceDiagnostic(model, obs)

# 1. N-study
results = conv.n_study(
    n_values=[100, 500, 1000, 5000],
    n_runs=10,
    seed=42,
)

# 2. Convergence plot
conv.plot_convergence(metric="log_likelihood", show_rate=True)

# 3. Find optimal N for PMCMC
asym = conv.asymptotic_variance(
    n_values=[500, 1000, 2000, 5000],
    n_runs=20,
    seed=42,
)
n_optimal = asym["n_for_std"](1.5)  # target std = 1.5 for PMMH
print(f"Recommended N for PMMH: {n_optimal:.0f}")

# 4. Verify with inter-run variance
report = conv.inter_run_variance(
    n_particles=int(n_optimal),
    n_runs=20,
    seed=42,
)
print(f"Verified std(log-lik) at N={int(n_optimal)}: {report['ll_std']:.3f}")
```

---

## API Summary

| Method | Description |
|--------|-------------|
| `ConvergenceDiagnostic(model, obs)` | Create diagnostic from model and observations |
| `.n_study(n_values, n_runs, seed)` | Run filter at multiple $N$ values |
| `.plot_convergence(metric, **kwargs)` | Plot convergence with $N$ |
| `.inter_run_variance(n_particles, n_runs)` | Variance across independent runs |
| `.asymptotic_variance(n_values, n_runs)` | Estimate $\sigma^2$ and predict performance |
| `.plot_convergence_rate(**kwargs)` | Plot fitted convergence rate |
| `.diminishing_returns(n_values, n_runs)` | Analyze cost-benefit of increasing $N$ |

---

## See Also

- [Theory: Convergence](../theory/convergence-theory.md) --- theoretical convergence results and CLT
- [ESS Diagnostic](ess-diagnostic.md) --- ESS as a per-step convergence indicator
- [Degeneracy Diagnostic](degeneracy.md) --- what prevents convergence in practice
- [PMCMC Tuning](../user-guide/pmcmc/tuning.md) --- choosing $N$ for PMCMC applications
- [Filters Overview](../user-guide/filters/index.md) --- better proposals can reduce the $N$ required for convergence
- [Experiment Framework](../user-guide/experiment.md) --- automate N-studies and filter comparisons
- [Acceleration: GPU](../acceleration/gpu.md) --- when the N-study demands large $N$, GPU acceleration enables it
- [Acceleration: Numba](../acceleration/numba.md) --- speed up individual filter runs during the N-study
