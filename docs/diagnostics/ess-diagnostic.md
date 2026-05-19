---
title: ESS Diagnostic
description: "Effective Sample Size diagnostics: time series analysis, alarm rates, and interpretation for particle filters"
---

# ESS Diagnostic

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `ESSDiagnostic` |
    | **Import** | `from particlefilterbox.diagnostics import ESSDiagnostic` |
    | **Input** | `FilterResult` from any particle filter |
    | **Key metric** | $\text{ESS}_t / N$ ratio over time |
    | **Healthy range** | Mean ESS ratio $> 0.3$, alarm rate $< 0.10$ |

## Overview

The ESS Diagnostic provides a comprehensive view of how the Effective Sample Size evolves throughout a filter run. While the [ESS core module](../user-guide/core/ess.md) covers the definition and computation, this diagnostic focuses on **interpretation**, **alarming**, and **actionable guidance**.

The key insight: a single ESS value at one time step tells you little. The **pattern** of ESS over time reveals whether the filter is working well, struggling with specific observations, or systematically failing.

---

## Basic Usage

```python
from particlefilterbox import BootstrapPF, PFConfig
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.diagnostics import ESSDiagnostic
import numpy as np

# Run a filter
model = StochasticVolatility(variant="basic")
config = PFConfig(n_particles=2000, ess_threshold=0.5, seed=42)
pf = BootstrapPF(model, config)

rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=300, rng=rng)
result = pf.filter(obs)

# Create diagnostic
diag = ESSDiagnostic(result)

# Summary statistics
print(diag.summary())
```

```text
=== ESS Diagnostic Summary ===
Particles (N):     2000
Time steps (T):    300

ESS Statistics:
  Mean:            1384.2  (69.2% of N)
  Median:          1456.8  (72.8% of N)
  Min:             312.1   (15.6% of N)  at t=147
  Max:             1987.3  (99.4% of N)
  Std:             298.4

Alarm Rates:
  ESS/N < 0.50:   0.156  (47/300 steps)
  ESS/N < 0.25:   0.023  (7/300 steps)
  ESS/N < 0.10:   0.003  (1/300 steps)

Verdict: HEALTHY
```

---

## ESS Over Time

### Plotting

```python
# Basic ESS time series plot
diag.plot()

# Customized plot
diag.plot(
    show_threshold=True,     # horizontal line at resampling threshold
    show_resampling=True,    # markers where resampling occurred
    show_percentiles=True,   # shaded bands for rolling percentiles
    figsize=(14, 5),
)
```

The ESS plot shows:

- **Blue line**: ESS at each time step
- **Red dashed line**: Resampling threshold ($\alpha N$)
- **Gray triangles**: Time steps where resampling was triggered
- **Shaded band**: Rolling 10th--90th percentile (window of 20 steps)

### Interpreting the Plot

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| ESS stays near $N$ | Observations are uninformative or model is easy | Consider reducing $N$ |
| ESS oscillates around $\alpha N$ | Normal adaptive behavior | No action needed |
| ESS drops sharply at isolated points | Outliers or regime changes in data | Investigate those time points |
| ESS persistently below $0.2N$ | Proposal is poorly matched to likelihood | Switch to SIR, Auxiliary, or Guided PF |
| ESS frequently hits 1 | Severe degeneracy | Fundamental problem --- see [Degeneracy](degeneracy.md) |

---

## Summary Statistics

### ESS Mean, Minimum, and Percentiles

```python
summary = diag.summary()

# Access individual statistics
print(f"Mean ESS:     {diag.mean_ess:.1f}")
print(f"Min ESS:      {diag.min_ess:.1f}")
print(f"Min ESS time: {diag.min_ess_time}")
print(f"Percentiles:  {diag.percentiles([5, 25, 50, 75, 95])}")
```

!!! tip "Rules of thumb for ESS statistics"
    - **Mean ESS / N > 0.5**: The filter is working well on average.
    - **Min ESS / N > 0.1**: Even the worst time step has reasonable particle diversity.
    - **Min ESS / N < 0.05**: At least one time step was nearly degenerate --- investigate.

### ESS Ratio Distribution

```python
# Histogram of ESS/N ratios across time
diag.plot_ess_distribution(bins=30)
```

This shows the distribution of $\text{ESS}_t / N$ across all time steps. A healthy filter produces a distribution concentrated above $0.5$. A struggling filter shows a heavy left tail or bimodal distribution.

---

## Alarm Rate

The **alarm rate** is the fraction of time steps where the ESS ratio falls below a given threshold:

$$
\text{AlarmRate}(\tau) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\!\left[\frac{\text{ESS}_t}{N} < \tau\right]
$$

```python
# Default threshold: 50% of N
rate = diag.alarm_rate(threshold=0.5)
print(f"Alarm rate (50%): {rate:.3f}")

# Multiple thresholds
for tau in [0.5, 0.25, 0.10, 0.05]:
    rate = diag.alarm_rate(threshold=tau)
    print(f"  ESS/N < {tau:.2f}: {rate:.3f}")
```

### Interpreting Alarm Rates

| Threshold | Acceptable Rate | Meaning |
|-----------|----------------|---------|
| $\tau = 0.50$ | $< 0.50$ | Up to half the steps can trigger resampling |
| $\tau = 0.25$ | $< 0.10$ | Rarely should ESS drop below 25% |
| $\tau = 0.10$ | $< 0.02$ | Near-degeneracy should be very rare |
| $\tau = 0.05$ | $0.00$ | Should essentially never happen |

!!! warning "Alarm rates depend on the resampling threshold"
    If adaptive resampling is enabled (the default), the alarm rate at $\tau = \alpha$ (the resampling threshold) is approximately equal to the resampling frequency. High alarm rates at thresholds *below* $\alpha$ are the real cause for concern --- they indicate that even after resampling, the ESS quickly degrades.

---

## ESS and Estimation Quality

The ESS directly controls the variance of particle estimates. For any test function $f$:

$$
\text{Var}\!\left[\hat{\mathbb{E}}[f(x_t) \mid y_{1:t}]\right] \approx \frac{\text{Var}_{\pi_t}[f(x_t)]}{\text{ESS}_t}
$$

This means:

- **ESS = 1000**: Estimation variance equivalent to 1000 independent samples
- **ESS = 100**: 10x higher variance than ESS = 1000
- **ESS = 10**: Estimates are essentially meaningless

### Log-Likelihood Sensitivity

The log-likelihood estimate $\log \hat{p}(y_{1:T})$ is particularly sensitive to low ESS. A single time step with very low ESS can dominate the total variance:

$$
\text{Var}\!\left[\log \hat{p}(y_{1:T})\right] \approx \sum_{t=1}^{T} \frac{1}{\text{ESS}_t}
$$

```python
# Check log-likelihood reliability
ll_variance_proxy = diag.log_likelihood_variance_proxy()
print(f"Log-lik variance proxy: {ll_variance_proxy:.4f}")
```

!!! note "For PMCMC users"
    If you are using the log-likelihood in PMMH or SMC$^2$, the variance should be below $\approx 1.0$. Higher variance leads to sticky MCMC chains and poor mixing. Use the ESS diagnostic to identify which time steps contribute most to the variance, then consider increasing $N$ or improving the proposal.

---

## Comparing Filters via ESS

One of the most useful applications of the ESS diagnostic is comparing different filter configurations:

```python
from particlefilterbox import BootstrapPF, SIRPF, AuxiliaryPF, PFConfig
from particlefilterbox.diagnostics import ESSDiagnostic

model = StochasticVolatility(variant="basic")
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=300, rng=rng)

filters = {
    "Bootstrap": BootstrapPF(model, PFConfig(n_particles=2000, seed=42)),
    "SIR": SIRPF(model, PFConfig(n_particles=2000, seed=42)),
    "Auxiliary": AuxiliaryPF(model, PFConfig(n_particles=2000, seed=42)),
}

for name, pf in filters.items():
    result = pf.filter(obs)
    diag = ESSDiagnostic(result)
    print(f"{name:12s}: mean_ESS={diag.mean_ess:7.1f}, "
          f"min_ESS={diag.min_ess:7.1f}, "
          f"alarm_rate={diag.alarm_rate(0.25):.3f}")
```

```text
Bootstrap   : mean_ESS= 1384.2, min_ESS=  312.1, alarm_rate=0.023
SIR         : mean_ESS= 1612.5, min_ESS=  687.3, alarm_rate=0.000
Auxiliary   : mean_ESS= 1743.8, min_ESS=  892.4, alarm_rate=0.000
```

### Comparative Plot

```python
# Side-by-side ESS comparison
ESSDiagnostic.compare(
    results={"Bootstrap": result_bpf, "SIR": result_sir, "Auxiliary": result_apf},
    figsize=(14, 6),
)
```

---

## What To Do When ESS Is Low

!!! warning "Low ESS is a symptom, not a root cause"
    Low ESS means the proposal distribution is poorly matched to the filtering distribution. The remedy depends on *why* it is low.

### Diagnosis Tree

| Symptom | Likely Cause | Remedy |
|---------|-------------|--------|
| ESS low everywhere | Proposal too diffuse relative to likelihood | Use SIR or Guided PF with informed proposal |
| ESS low at specific times | Outliers or structural breaks | Robustify the observation model (heavier tails) |
| ESS low despite many particles | High-dimensional state space | Rao-Blackwellized PF or Ensemble PF |
| ESS low and worsening over time | Accumulating model mismatch | Check model specification; consider regime-switching |

### Step-by-Step Remediation

1. **Increase $N$**: The simplest fix. ESS scales linearly with $N$, but computation does too.
2. **Improve the proposal**: Switch from Bootstrap to SIR (uses an EKF/UKF proposal) or Auxiliary PF.
3. **Add tempering**: For very informative observations, temper the likelihood across sub-steps.
4. **Rao-Blackwellize**: If part of the state is conditionally linear-Gaussian, marginalize it out.
5. **Regularize**: The Regularized PF adds a small kernel move after resampling to maintain diversity.

```python
# Example: improving ESS by switching proposal
from particlefilterbox import SIRPF, PFConfig

# SIR uses an EKF-based proposal that incorporates y_t
config = PFConfig(n_particles=2000, seed=42)
sir = SIRPF(model, config)
result_sir = sir.filter(obs)

diag_sir = ESSDiagnostic(result_sir)
print(f"SIR mean ESS: {diag_sir.mean_ess:.1f} vs Bootstrap: {diag.mean_ess:.1f}")
```

---

## API Summary

| Method | Description |
|--------|-------------|
| `ESSDiagnostic(result)` | Create diagnostic from a `FilterResult` |
| `.summary()` | Print comprehensive ESS summary |
| `.plot(**kwargs)` | Plot ESS over time with options |
| `.plot_ess_distribution(bins)` | Histogram of ESS/N ratios |
| `.alarm_rate(threshold)` | Fraction of steps below threshold |
| `.mean_ess` | Mean ESS across time |
| `.min_ess` | Minimum ESS value |
| `.min_ess_time` | Time step of minimum ESS |
| `.percentiles(q)` | ESS percentiles |
| `.log_likelihood_variance_proxy()` | Proxy for log-likelihood variance |
| `.compare(results, **kwargs)` | Compare ESS across multiple filters |

---

## See Also

- [Core: ESS](../user-guide/core/ess.md) --- definition, derivation, and adaptive resampling
- [Weight Diagnostic](weight-diagnostic.md) --- complementary weight distribution analysis
- [Degeneracy Diagnostic](degeneracy.md) --- what happens when ESS stays low
- [Convergence Diagnostic](convergence.md) --- how many particles do you really need?
- [Theory: Convergence](../theory/convergence-theory.md) --- theoretical relationship between ESS and estimation variance
- [Filters Overview](../user-guide/filters/index.md) --- choosing a filter with better ESS behaviour
- [Acceleration: Adaptive N](../acceleration/adaptive-n.md) --- dynamically adjust $N$ based on ESS
- [Acceleration Overview](../acceleration/index.md) --- scale up $N$ when ESS diagnostics demand more particles
