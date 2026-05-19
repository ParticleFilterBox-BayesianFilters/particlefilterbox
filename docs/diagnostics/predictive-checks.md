---
title: Posterior Predictive Checks
description: "Posterior predictive checks for particle filter models: data generation, visual comparison, Bayesian p-values, and model misspecification detection"
---

# Posterior Predictive Checks

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `PredictiveCheck` |
    | **Import** | `from particlefilterbox.diagnostics import PredictiveCheck` |
    | **Input** | Model, MCMC chain, and observed data |
    | **Key method** | `.generate(n_samples)` → `.plot_comparison()` |
    | **Goal** | Assess whether the model can reproduce features of the observed data |

## Overview

Posterior predictive checks (PPCs) are a fundamental tool for **model validation** in Bayesian inference. The idea is simple:

1. Draw parameter samples $\theta^{(s)}$ from the posterior
2. For each $\theta^{(s)}$, simulate a full dataset $y^{(s)}_{1:T}$ from the model
3. Compare the simulated datasets to the observed data $y_{1:T}$

If the model is well-specified, the observed data should look like a **typical draw** from the posterior predictive distribution:

$$
p(y^{\text{rep}} \mid y) = \int p(y^{\text{rep}} \mid \theta)\, p(\theta \mid y)\, d\theta
$$

Systematic discrepancies between the observed and simulated data indicate **model misspecification**.

---

## Basic Usage

```python
from particlefilterbox.diagnostics import PredictiveCheck

# model: the state-space model
# chain: posterior samples from PMMH/PGAS (n_iter, n_params)
# obs: observed data (T,) or (T, d)
ppc = PredictiveCheck(model, chain, obs)

# Generate replicated datasets
ppc.generate(n_samples=1000, seed=42)

# Visual comparison
ppc.plot_comparison()
```

---

## Generating Predictive Samples

### How It Works

For each of the `n_samples` posterior draws $\theta^{(s)}$:

1. Set model parameters to $\theta^{(s)}$
2. Simulate the state trajectory: $x^{(s)}_{0:T} \sim p(x_{0:T} \mid \theta^{(s)})$
3. Simulate observations: $y^{(s)}_t \sim p(y_t \mid x^{(s)}_t, \theta^{(s)})$ for $t = 1, \ldots, T$

```python
# Generate with options
ppc.generate(
    n_samples=1000,      # number of replicated datasets
    thin=5,              # use every 5th posterior sample
    seed=42,
)

# Access the generated data
print(f"Shape of replicated data: {ppc.y_rep.shape}")  # (1000, T)
print(f"Shape of observed data:   {ppc.y_obs.shape}")   # (T,)
```

```text
Shape of replicated data: (1000, 200)
Shape of observed data:   (200,)
```

---

## Visual Comparisons

### Time Series Overlay

```python
# Overlay replicated data on observed data
ppc.plot_comparison(
    kind="timeseries",
    ci_levels=[0.50, 0.90, 0.99],  # credible intervals
    figsize=(14, 5),
)
```

The time series plot shows:

- **Black line**: Observed data $y_{1:T}$
- **Shaded bands**: Posterior predictive credible intervals (50%, 90%, 99%)
- **Blue line**: Median of replicated data

!!! tip "What to look for"
    - Observed data should mostly fall within the 90% band
    - Systematic excursions outside the band indicate model misspecification
    - Observations consistently at the edge of the band suggest the model under-predicts variability

### Distribution Comparison

```python
# Compare marginal distributions
ppc.plot_comparison(kind="density")
```

This overlays the histogram of the observed data with the posterior predictive density. A well-specified model produces a density that closely matches the observed distribution.

### Scatter Plot (Observed vs Replicated)

```python
# Observed vs replicated quantiles
ppc.plot_comparison(kind="qq")
```

A QQ-plot of observed data quantiles against posterior predictive quantiles. Points on the diagonal indicate good fit.

---

## Test Statistics

### Built-in Statistics

PPCs become most powerful when you focus on **specific features** of the data. Compare test statistics computed on the observed data $T(y)$ with the distribution of $T(y^{\text{rep}})$:

```python
# Compute test statistics
stats = ppc.test_statistics()
print(stats)
```

```text
=== Posterior Predictive Test Statistics ===
Statistic       | Observed | Pred. Mean | Pred. Std |  p-value
----------------+----------+------------+-----------+---------
Mean            |   0.032  |    0.028   |   0.041   |  0.462
Variance        |   1.847  |    1.723   |   0.198   |  0.267
Skewness        |   0.214  |    0.015   |   0.142   |  0.080
Kurtosis        |   4.823  |    3.412   |   0.387   |  0.001  ⚠
Max |y_t|       |   5.421  |    3.872   |   0.521   |  0.002  ⚠
ACF(lag=1)      |   0.048  |    0.035   |   0.067   |  0.421
ACF(lag=5)      |   0.112  |    0.008   |   0.064   |  0.052
```

### Custom Test Statistics

```python
# Define custom test statistics
def max_drawdown(y):
    """Maximum peak-to-trough decline."""
    cummax = np.maximum.accumulate(y)
    return np.max(cummax - y)

def tail_ratio(y):
    """Ratio of extreme observations beyond 2 std."""
    return np.mean(np.abs(y) > 2 * np.std(y))

# Compute with custom statistics
stats = ppc.test_statistics(
    custom={
        "Max drawdown": max_drawdown,
        "Tail ratio": tail_ratio,
    }
)
```

### Visualizing Test Statistics

```python
# Histogram of T(y_rep) with observed T(y) marked
ppc.plot_statistic("Kurtosis")
ppc.plot_statistic("Max |y_t|")
```

The histogram shows the distribution of $T(y^{\text{rep}})$ across replicated datasets, with a vertical line at the observed value $T(y)$. If the observed value falls in the tails, the model fails to reproduce that feature.

---

## Bayesian p-values

### Definition

The Bayesian p-value for a test statistic $T$ is:

$$
p_B = P\!\left(T(y^{\text{rep}}) \geq T(y) \mid y\right) \approx \frac{1}{S}\sum_{s=1}^{S} \mathbb{1}\!\left[T(y^{(s)}) \geq T(y)\right]
$$

Values near 0 or 1 indicate that the observed data is extreme relative to the posterior predictive.

### Computing p-values

```python
# p-values for all default statistics
p_values = ppc.p_values()
print(p_values)
```

```text
{'Mean': 0.462, 'Variance': 0.267, 'Skewness': 0.080,
 'Kurtosis': 0.001, 'Max |y_t|': 0.002, 'ACF(lag=1)': 0.421}
```

```python
# p-value for a custom statistic
p_kurtosis = ppc.p_value(statistic="Kurtosis")
print(f"Bayesian p-value for kurtosis: {p_kurtosis:.4f}")
```

### Interpreting p-values

!!! warning "Bayesian p-values are not frequentist p-values"
    Bayesian p-values measure how extreme the observed data is relative to the model's predictions. They are **calibration diagnostics**, not hypothesis tests.

    | p-value | Interpretation |
    |---------|---------------|
    | $0.05 < p < 0.95$ | Model consistent with data for this statistic |
    | $p < 0.05$ or $p > 0.95$ | Model struggles to reproduce this feature |
    | $p < 0.01$ or $p > 0.99$ | Strong evidence of model misspecification for this feature |

!!! note "Multiple testing"
    When checking many test statistics, some p-values will be extreme by chance. Focus on statistics that are substantively important for your application rather than fishing for significant p-values.

---

## Detecting Model Misspecification

### Common Patterns

| Observed discrepancy | Possible misspecification | Remedy |
|---------------------|--------------------------|--------|
| Kurtosis too high | Observation noise too light-tailed | Use Student-$t$ observations |
| Variance too high | State volatility underestimated | Add stochastic volatility or regime switching |
| ACF mismatch | Wrong state dynamics order | Increase AR order or add latent factors |
| Extreme values unexplained | No mechanism for outliers | Add outlier component or heavy tails |
| Seasonal pattern in residuals | Missing seasonal component | Add seasonal state variable |

### Targeted Checks

```python
# Check specific model assumptions

# 1. Are observation errors Gaussian?
ppc.check_normality(component="observations")

# 2. Are state transitions correct?
ppc.check_dynamics(statistic="acf", max_lag=20)

# 3. Does the model capture volatility clustering?
ppc.check_volatility_clustering()
```

---

## Complete Example

```python
import numpy as np
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.pmcmc import PMMH
from particlefilterbox.diagnostics import PredictiveCheck

# Setup and run PMMH
model = StochasticVolatility(variant="basic")
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=200, rng=rng)

pmmh = PMMH(model, obs, n_particles=1000, adaptive=True)
chain = pmmh.run(n_iterations=10000, seed=42)

# Posterior predictive checks (discard burn-in)
ppc = PredictiveCheck(model, chain[2000:], obs)
ppc.generate(n_samples=1000, seed=42)

# 1. Visual comparison
ppc.plot_comparison(kind="timeseries", ci_levels=[0.50, 0.90])
ppc.plot_comparison(kind="density")

# 2. Test statistics and p-values
stats = ppc.test_statistics()
print(stats)

# 3. Identify problematic statistics
p_values = ppc.p_values()
for name, pval in p_values.items():
    if pval < 0.05 or pval > 0.95:
        print(f"⚠ {name}: p = {pval:.4f} — model may not capture this feature")

# 4. Detailed plot for any problematic statistic
ppc.plot_statistic("Kurtosis")
```

---

## API Summary

| Method | Description |
|--------|-------------|
| `PredictiveCheck(model, chain, obs)` | Create PPC from model, posterior chain, and data |
| `.generate(n_samples, thin, seed)` | Generate replicated datasets from posterior predictive |
| `.plot_comparison(kind, **kwargs)` | Visual comparison (`"timeseries"`, `"density"`, `"qq"`) |
| `.test_statistics(custom)` | Compute test statistics on observed and replicated data |
| `.p_values()` | Bayesian p-values for all test statistics |
| `.p_value(statistic)` | Bayesian p-value for a specific statistic |
| `.plot_statistic(name, **kwargs)` | Histogram of $T(y^{\text{rep}})$ with observed $T(y)$ |
| `.check_normality(component)` | Normality check for observations or state transitions |
| `.check_dynamics(statistic, max_lag)` | Check state dynamics assumptions via ACF |
| `.y_rep` | Array of replicated datasets `(n_samples, T)` |
| `.y_obs` | Observed data array |

---

## See Also

- [Marginal Likelihood](marginal-likelihood.md) --- model comparison via Bayes factors
- [MCMC Convergence](mcmc-convergence.md) --- verify chains before running PPCs
- [Mixing Diagnostics](mixing.md) --- ensure adequate posterior samples
- [PMMH](../user-guide/pmcmc/pmmh.md) --- generate posterior chains for predictive checks
- [PGAS](../user-guide/pmcmc/pgas.md) --- alternative PMCMC sampler for chain generation
- [Models Overview](../user-guide/models/index.md) --- pre-built models including stochastic volatility, DSGE, and regime-switching
- [Stochastic Volatility](../user-guide/models/stochastic-volatility.md) --- commonly used model for PPCs in financial applications
