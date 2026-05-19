---
title: PMCMC Plots
description: Trace plots, posterior distributions, autocorrelation, and convergence diagnostics for PMCMC chains
---

# PMCMC Plots

PMCMC plots provide standard MCMC diagnostics tailored for Particle MCMC methods (PMMH, Particle Gibbs, PG-AS). These plots help assess convergence, mixing, and the quality of posterior inference.

```python
from particlefilterbox.viz import (
    plot_trace,
    plot_posterior,
    plot_posterior_2d,
    plot_acf,
    plot_running_mean,
    plot_acceptance_rate,
    plot_prior_vs_posterior,
)
```

---

## `plot_trace` { #plot_trace }

Trace plots of MCMC chain parameters, the primary visual diagnostic for convergence and mixing.

### API

```python
plot_trace(
    chain,                    # PMCMCResult or dict of arrays
    params=None,              # Parameter names (None = all)
    burnin=0,                 # Number of burn-in samples to shade
    true_values=None,         # Dict of true parameter values
    colors=None,              # List of colors per parameter
    burnin_color="#FFCDD2",   # Background color for burn-in region
    ax=None,
    figsize=(12, 3),          # Per-parameter subplot height
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain` | `PMCMCResult \| dict` | required | MCMC chain output |
| `params` | `list[str] \| None` | `None` | Parameters to plot; `None` plots all |
| `burnin` | `int` | `0` | Number of burn-in iterations to highlight |
| `true_values` | `dict[str, float] \| None` | `None` | True values to mark with horizontal lines |

### Example

```python
from particlefilterbox import PMMH
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.viz import plot_trace

model = StochasticVolatility(phi=0.97, sigma=0.15, beta=0.65)
true_states, observations = model.simulate(T=500, seed=42)

pmmh = PMMH(model, n_particles=200)
chain = pmmh.sample(observations, n_iterations=5000, burnin=1000)

fig, axes = plot_trace(
    chain,
    params=["phi", "sigma", "beta"],
    burnin=1000,
    true_values={"phi": 0.97, "sigma": 0.15, "beta": 0.65},
)
```

!!! note "Output"
    One subplot per parameter. Each subplot shows the parameter value at each iteration as a time series. The burn-in region is shaded in light red. Horizontal dashed lines mark the true parameter values. Good mixing appears as a "hairy caterpillar" pattern with rapid oscillation around the posterior mean. Slow trends or long flat stretches indicate poor mixing.

---

## `plot_posterior` { #plot_posterior }

Histogram and/or KDE of the marginal posterior distribution for each parameter.

### API

```python
plot_posterior(
    chain,                    # PMCMCResult or dict
    params=None,              # Parameter names
    burnin=0,                 # Samples to discard
    bins=50,
    kde=True,                 # Overlay KDE
    true_values=None,         # True parameter values
    hdi=0.95,                 # Highest Density Interval to shade
    color="#4051B5",
    kde_color="#E91E63",
    hdi_color="#C5CAE9",
    ax=None,
    figsize=(5, 4),           # Per-parameter subplot
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain` | `PMCMCResult \| dict` | required | MCMC chain output |
| `burnin` | `int` | `0` | Number of burn-in samples to discard |
| `kde` | `bool` | `True` | Overlay smooth KDE on histogram |
| `hdi` | `float` | `0.95` | Highest Density Interval level to highlight |
| `true_values` | `dict[str, float] \| None` | `None` | True values to mark with vertical lines |

### Example

```python
from particlefilterbox.viz import plot_posterior

fig, axes = plot_posterior(
    chain,
    params=["phi", "sigma", "beta"],
    burnin=1000,
    hdi=0.95,
    true_values={"phi": 0.97, "sigma": 0.15, "beta": 0.65},
)
```

!!! note "Output"
    One subplot per parameter. Each shows a histogram overlaid with a smooth KDE curve. The $95\%$ Highest Density Interval (HDI) is shaded, and a vertical dashed line marks the true value. A well-identified parameter shows a unimodal posterior with the true value inside the HDI.

---

## `plot_posterior_2d` { #plot_posterior_2d }

Bivariate posterior distribution as a scatter plot with contour overlay, revealing correlations between parameters.

### API

```python
plot_posterior_2d(
    chain,                    # PMCMCResult or dict
    params=("phi", "sigma"),  # Pair of parameter names
    burnin=0,
    scatter=True,             # Show scatter points
    contour=True,             # Overlay KDE contours
    n_levels=6,               # Number of contour levels
    scatter_alpha=0.15,
    scatter_size=5,
    cmap="Blues",
    true_values=None,         # True parameter values (marked as ×)
    ax=None,
    figsize=(7, 7),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain` | `PMCMCResult \| dict` | required | MCMC chain output |
| `params` | `tuple[str, str]` | `("phi", "sigma")` | Pair of parameter names |
| `scatter` | `bool` | `True` | Show individual MCMC samples |
| `contour` | `bool` | `True` | Overlay bivariate KDE contours |
| `n_levels` | `int` | `6` | Number of contour levels |

### Example

```python
from particlefilterbox.viz import plot_posterior_2d

fig, ax = plot_posterior_2d(
    chain,
    params=("phi", "sigma"),
    burnin=1000,
    scatter_alpha=0.1,
    cmap="Blues",
    true_values={"phi": 0.97, "sigma": 0.15},
)
```

!!! note "Output"
    A scatter plot of MCMC samples in 2D parameter space, overlaid with KDE contour lines representing posterior density levels. The true parameter values appear as a red $\times$ marker. Elongated contours indicate strong correlation between parameters. Round contours suggest near-independence.

---

## `plot_acf` { #plot_acf }

Autocorrelation function (ACF) of the MCMC chain, quantifying how quickly the chain forgets its past.

### API

```python
plot_acf(
    chain,                    # PMCMCResult or dict
    params=None,              # Parameter names
    burnin=0,
    max_lag=100,              # Maximum lag to compute
    color="#4051B5",
    ci_color="#BBDEFB",       # Color for confidence interval band
    ax=None,
    figsize=(8, 3),           # Per-parameter subplot
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain` | `PMCMCResult \| dict` | required | MCMC chain output |
| `max_lag` | `int` | `100` | Maximum lag for ACF computation |
| `burnin` | `int` | `0` | Samples to discard before computing ACF |

### Example

```python
from particlefilterbox.viz import plot_acf

fig, axes = plot_acf(
    chain,
    params=["phi", "sigma", "beta"],
    burnin=1000,
    max_lag=80,
)
```

!!! note "Output"
    One subplot per parameter. Each shows vertical bars of the autocorrelation $\rho(k)$ at lags $k = 0, 1, \ldots, K$. A light blue shaded band marks the $95\%$ confidence interval under the null of white noise. ACF that decays rapidly (within $10$--$20$ lags) indicates good mixing. Slow decay suggests high autocorrelation and the need for thinning or reparameterization.

!!! tip "Effective sample size"
    The ACF is directly related to the effective sample size:

    $$
    \text{ESS} = \frac{M}{1 + 2 \sum_{k=1}^{K} \rho(k)}
    $$

    where $M$ is the chain length after burn-in. Slow ACF decay means low ESS.

---

## `plot_running_mean` { #plot_running_mean }

Running (cumulative) mean of each parameter over iterations, used to assess convergence to the posterior mean.

### API

```python
plot_running_mean(
    chain,                    # PMCMCResult or dict
    params=None,
    burnin=0,
    true_values=None,
    colors=None,
    ax=None,
    figsize=(10, 3),          # Per-parameter subplot
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain` | `PMCMCResult \| dict` | required | MCMC chain output |
| `burnin` | `int` | `0` | Samples to discard |
| `true_values` | `dict[str, float] \| None` | `None` | True values for reference |

### Example

```python
from particlefilterbox.viz import plot_running_mean

fig, axes = plot_running_mean(
    chain,
    params=["phi", "sigma", "beta"],
    burnin=1000,
    true_values={"phi": 0.97, "sigma": 0.15, "beta": 0.65},
)
```

!!! note "Output"
    One subplot per parameter showing the cumulative average $\bar{\theta}_n = \frac{1}{n} \sum_{i=1}^{n} \theta^{(i)}$ over iterations. A horizontal dashed line marks the true value. Convergence is indicated by the running mean stabilizing (flattening) around the posterior mean. Persistent drift suggests the chain has not converged.

---

## `plot_acceptance_rate` { #plot_acceptance_rate }

Acceptance rate of the Metropolis-Hastings step over iterations, using a rolling window.

### API

```python
plot_acceptance_rate(
    chain,                    # PMCMCResult
    window=100,               # Rolling window size
    target_range=(0.15, 0.40),# Optimal acceptance rate range
    color="#4051B5",
    target_color="#C8E6C9",   # Background color for target range
    ax=None,
    figsize=(10, 4),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain` | `PMCMCResult` | required | MCMC chain with acceptance information |
| `window` | `int` | `100` | Rolling window size for smoothing |
| `target_range` | `tuple[float, float]` | `(0.15, 0.40)` | Optimal acceptance rate band |

### Example

```python
from particlefilterbox.viz import plot_acceptance_rate

fig, ax = plot_acceptance_rate(
    chain,
    window=200,
    target_range=(0.15, 0.40),
)
```

!!! note "Output"
    A line plot of the rolling acceptance rate over iterations. A green shaded band highlights the optimal range (typically $15\%$--$40\%$ for PMMH). The acceptance rate should stabilize within this band after burn-in.

!!! warning "Tuning the proposal"
    - **Acceptance rate too high** ($> 50\%$): proposal variance is too small -- the chain makes small, correlated steps. Increase the proposal scale.
    - **Acceptance rate too low** ($< 10\%$): proposal variance is too large -- most proposals are rejected. Decrease the proposal scale.
    - **Optimal range for PMMH**: $15\%$--$35\%$, depending on the number of particles and parameter dimension.

---

## `plot_prior_vs_posterior` { #plot_prior_vs_posterior }

Overlay of prior and posterior densities for each parameter, showing how much the data updates prior beliefs.

### API

```python
plot_prior_vs_posterior(
    chain,                    # PMCMCResult or dict
    priors,                   # Dict of scipy.stats distributions
    params=None,
    burnin=0,
    bins=50,
    prior_color="#9E9E9E",
    posterior_color="#4051B5",
    prior_linestyle="--",
    true_values=None,
    ax=None,
    figsize=(5, 4),           # Per-parameter subplot
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain` | `PMCMCResult \| dict` | required | MCMC chain output |
| `priors` | `dict[str, rv_continuous]` | required | Dictionary mapping parameter names to `scipy.stats` distributions |
| `burnin` | `int` | `0` | Samples to discard |
| `true_values` | `dict[str, float] \| None` | `None` | True parameter values |

### Example

```python
from scipy import stats
from particlefilterbox.viz import plot_prior_vs_posterior

priors = {
    "phi": stats.beta(a=20, b=1.5),
    "sigma": stats.halfnorm(scale=0.5),
    "beta": stats.halfnorm(scale=1.0),
}

fig, axes = plot_prior_vs_posterior(
    chain,
    priors=priors,
    burnin=1000,
    true_values={"phi": 0.97, "sigma": 0.15, "beta": 0.65},
)
```

!!! note "Output"
    One subplot per parameter. Each shows the prior density as a gray dashed line and the posterior (from MCMC samples) as a solid blue histogram with KDE overlay. A vertical dashed line marks the true value. When the posterior is substantially different from the prior, the data is informative. When they overlap, the data provides little information about that parameter.

!!! tip "Prior sensitivity"
    Run the same PMCMC with different priors and compare the resulting posteriors:

    ```python
    import matplotlib.pyplot as plt
    from particlefilterbox.viz import plot_prior_vs_posterior

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Tight priors
    plot_prior_vs_posterior(chain_tight, priors_tight,
                           params=["phi"], ax=axes[0], show=False)
    axes[0].set_title("Tight Priors")

    # Diffuse priors
    plot_prior_vs_posterior(chain_diffuse, priors_diffuse,
                           params=["phi"], ax=axes[1], show=False)
    axes[1].set_title("Diffuse Priors")

    # Misspecified priors
    plot_prior_vs_posterior(chain_mis, priors_mis,
                           params=["phi"], ax=axes[2], show=False)
    axes[2].set_title("Misspecified Priors")

    fig.suptitle(r"Prior Sensitivity for $\phi$", fontsize=14)
    fig.tight_layout()
    plt.show()
    ```

---

## Customization

### Full Diagnostic Panel

```python
import matplotlib.pyplot as plt
from particlefilterbox.viz import (
    plot_trace,
    plot_posterior,
    plot_acf,
    plot_running_mean,
)

# Create a comprehensive diagnostic figure for a single parameter
param = "phi"
true_val = 0.97
burnin = 1000

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 2, 1)
plot_trace(chain, params=[param], burnin=burnin,
           true_values={param: true_val}, ax=ax1, show=False)

ax2 = fig.add_subplot(2, 2, 2)
plot_posterior(chain, params=[param], burnin=burnin,
              true_values={param: true_val}, ax=ax2, show=False)

ax3 = fig.add_subplot(2, 2, 3)
plot_acf(chain, params=[param], burnin=burnin,
         max_lag=80, ax=ax3, show=False)

ax4 = fig.add_subplot(2, 2, 4)
plot_running_mean(chain, params=[param], burnin=burnin,
                  true_values={param: true_val}, ax=ax4, show=False)

fig.suptitle(rf"PMCMC Diagnostics: $\phi$", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig("pmcmc_diagnostics_phi.pdf", dpi=300, bbox_inches="tight")
plt.show()
```

### Multi-Chain Comparison

```python
import matplotlib.pyplot as plt
from particlefilterbox.viz import plot_trace

fig, ax = plt.subplots(figsize=(12, 4))

# Overlay multiple chains for the same parameter
for i, chain_i in enumerate(chains):
    samples = chain_i.samples["phi"][burnin:]
    ax.plot(samples, alpha=0.6, linewidth=0.5, label=f"Chain {i+1}")

ax.axhline(y=0.97, color="black", linestyle="--", label="True value")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\phi$")
ax.legend(loc="upper right")
ax.set_title("Multi-Chain Trace Plot")
plt.tight_layout()
plt.show()
```

### Publication-Ready Settings

```python
# Journal-quality posterior plot
plot_posterior(
    chain,
    params=["phi", "sigma", "beta"],
    burnin=1000,
    theme="paper",
    figsize=(4.5, 3.5),
    bins=40,
    hdi=0.95,
    true_values={"phi": 0.97, "sigma": 0.15, "beta": 0.65},
)
```
