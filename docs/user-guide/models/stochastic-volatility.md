---
title: Stochastic Volatility
description: "Stochastic Volatility model with 4 variants: basic, leverage, jumps, and factor"
---

# Stochastic Volatility

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `StochasticVolatility` |
    | **Import** | `from particlefilterbox.models import StochasticVolatility` |
    | **Variants** | `basic`, `leverage`, `jumps`, `factor` |
    | **State** | Log-volatility $h_t$ (continuous, 1D--4D) |
    | **Observation** | Returns $y_t$ (continuous) |
    | **Recommended filter** | [Bootstrap PF](../filters/bootstrap.md) / [Auxiliary PF](../filters/auxiliary.md) |
    | **References** | Kim, Shephard & Chib (1998); Omori et al. (2007); Eraker et al. (2003) |

---

## Overview

The **Stochastic Volatility (SV)** model is the standard framework for capturing time-varying volatility in financial returns. Unlike GARCH models, SV treats volatility as a *latent state* that evolves stochastically, making it a natural fit for particle filtering.

particlefilterbox provides **4 variants** of the SV model, from the classic single-factor specification to multivariate factor models.

---

## Mathematical Framework

### Basic SV Model

The canonical SV model (Kim, Shephard & Chib, 1998):

$$
\begin{aligned}
h_t &= \mu + \phi (h_{t-1} - \mu) + \sigma_\eta \, \eta_t, \qquad \eta_t \sim \mathcal{N}(0, 1) \\[6pt]
y_t &= \exp\!\left(\frac{h_t}{2}\right) \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1)
\end{aligned}
$$

where:

| Parameter | Symbol | Description | Typical Range |
|:----------|:-------|:------------|:--------------|
| Level | $\mu$ | Long-run mean of log-volatility | $[-3, 1]$ |
| Persistence | $\phi$ | Autoregressive coefficient | $[0.9, 0.999]$ |
| Vol-of-vol | $\sigma_\eta$ | Volatility of log-volatility | $[0.05, 0.5]$ |

The stationary distribution of $h_t$ is:

$$
h_t \sim \mathcal{N}\!\left(\mu, \; \frac{\sigma_\eta^2}{1 - \phi^2}\right)
$$

!!! note "Why log-volatility?"
    Working in log-volatility space ($h_t$ instead of $\sigma_t^2$) ensures that the volatility process remains positive without requiring constrained sampling. The observation equation $y_t = \exp(h_t / 2) \, \varepsilon_t$ maps back to the natural scale.

### SV with Leverage

The leverage effect (Omori et al., 2007) introduces correlation between return shocks and volatility innovations:

$$
\begin{aligned}
h_t &= \mu + \phi (h_{t-1} - \mu) + \sigma_\eta \, \eta_t \\[4pt]
y_t &= \exp\!\left(\frac{h_t}{2}\right) \varepsilon_t
\end{aligned}
$$

with $\text{Corr}(\varepsilon_t, \eta_{t+1}) = \rho$. The leverage parameter $\rho < 0$ captures the empirical observation that negative returns tend to increase future volatility.

In practice, the joint innovation is constructed as:

$$
\eta_{t+1} = \rho \, \varepsilon_t + \sqrt{1 - \rho^2} \, \nu_t, \qquad \nu_t \sim \mathcal{N}(0, 1)
$$

### SV with Jumps

The SV-J model (Eraker, Johannes & Polson, 2003) adds rare large movements:

$$
\begin{aligned}
h_t &= \mu + \phi (h_{t-1} - \mu) + \sigma_\eta \, \eta_t \\[4pt]
y_t &= \exp\!\left(\frac{h_t}{2}\right) \varepsilon_t + q_t \, J_t
\end{aligned}
$$

where:

$$
q_t \sim \text{Bernoulli}(\lambda), \qquad J_t \sim \mathcal{N}(\mu_J, \sigma_J^2)
$$

The state is augmented to $(h_t, q_t)$ with `k_states = 2`.

| Parameter | Symbol | Default | Description |
|:----------|:-------|:--------|:------------|
| Jump intensity | $\lambda$ | $0.05$ | Probability of jump at each step |
| Jump mean | $\mu_J$ | $-0.5$ | Average jump size (negative = crashes) |
| Jump std | $\sigma_J$ | $1.0$ | Jump size variability |

### Factor SV

The multivariate factor SV model (Chib, Nardari & Shephard, 2006) for $K$ observed series:

$$
\begin{aligned}
h_t^{(0)} &= \mu + \phi (h_{t-1}^{(0)} - \mu) + \sigma_\eta \, \eta_t^{(0)} && \text{(common factor)} \\[4pt]
h_t^{(k)} &= \phi_k \, h_{t-1}^{(k)} + \sigma_k \, \eta_t^{(k)} && \text{(idiosyncratic, } k = 1, \ldots, K\text{)} \\[4pt]
y_t^{(k)} &= \exp\!\left(\frac{\beta_k \, h_t^{(0)} + h_t^{(k)}}{2}\right) \varepsilon_t^{(k)}
\end{aligned}
$$

State dimension: $1 + K$. Observation dimension: $K$.

---

## API

### Constructor

```python
from particlefilterbox.models import StochasticVolatility

# Basic SV with default parameters
sv = StochasticVolatility()

# Basic SV with custom parameters
sv = StochasticVolatility(
    variant="basic",
    params={"mu": 0.0, "phi": 0.97, "sigma": 0.15}
)

# SV with leverage
sv_lev = StochasticVolatility(
    variant="leverage",
    params={"mu": -1.0, "phi": 0.97, "sigma": 0.15, "rho": -0.5}
)

# SV with jumps
sv_j = StochasticVolatility(
    variant="jumps",
    params={
        "mu": -1.0, "phi": 0.97, "sigma": 0.15,
        "lambda_jump": 0.05, "mu_jump": -0.5, "sigma_jump": 1.0,
    }
)

# Factor SV for 3 series
sv_factor = StochasticVolatility(variant="factor", k_factor_series=3)
```

### Parameters by Variant

=== "basic"

    | Parameter | Key | Default | Prior |
    |:----------|:----|:--------|:------|
    | $\mu$ | `mu` | $-1.0$ | $\mathcal{N}(0, 5)$ |
    | $\phi$ | `phi` | $0.97$ | $\text{Beta}(20, 1.5)$ |
    | $\sigma_\eta$ | `sigma` | $0.15$ | $\text{InvGamma}(2.5, 0.025)$ |

=== "leverage"

    | Parameter | Key | Default | Prior |
    |:----------|:----|:--------|:------|
    | $\mu$ | `mu` | $-1.0$ | $\mathcal{N}(0, 5)$ |
    | $\phi$ | `phi` | $0.97$ | $\text{Beta}(20, 1.5)$ |
    | $\sigma_\eta$ | `sigma` | $0.15$ | $\text{InvGamma}(2.5, 0.025)$ |
    | $\rho$ | `rho` | $-0.5$ | $\text{Uniform}(-1, 1)$ |

=== "jumps"

    | Parameter | Key | Default | Prior |
    |:----------|:----|:--------|:------|
    | $\mu$ | `mu` | $-1.0$ | $\mathcal{N}(0, 5)$ |
    | $\phi$ | `phi` | $0.97$ | $\text{Beta}(20, 1.5)$ |
    | $\sigma_\eta$ | `sigma` | $0.15$ | $\text{InvGamma}(2.5, 0.025)$ |
    | $\lambda$ | `lambda_jump` | $0.05$ | $\text{Beta}(2, 40)$ |
    | $\mu_J$ | `mu_jump` | $-0.5$ | $\mathcal{N}(0, 2)$ |
    | $\sigma_J$ | `sigma_jump` | $1.0$ | $\text{InvGamma}(2.5, 1)$ |

=== "factor"

    | Parameter | Key | Default | Prior |
    |:----------|:----|:--------|:------|
    | $\mu$ | `mu` | $-1.0$ | $\mathcal{N}(0, 5)$ |
    | $\phi$ | `phi` | $0.97$ | $\text{Beta}(20, 1.5)$ |
    | $\sigma_\eta$ | `sigma` | $0.15$ | $\text{InvGamma}(2.5, 0.025)$ |
    | $\phi_k$ | `phi_k` | $0.95$ | $\text{Beta}(20, 1.5)$ |
    | $\sigma_k$ | `sigma_k` | $0.2$ | $\text{InvGamma}(2.5, 0.025)$ |
    | $\beta_k$ | `beta_k` | $1.0$ | $\mathcal{N}(1, 1)$ |

### Simulation

```python
# Simulate 1000 observations
sv = StochasticVolatility(variant="basic", params={"mu": 0.0, "phi": 0.97, "sigma": 0.15})
sim = sv.simulate(T=1000, seed=42)

# Returns dict with 'observations' and 'states'
returns = sim["observations"]  # shape (1000, 1)
log_vol = sim["states"]        # shape (1000, 1)
```

---

## Filtering

### Basic Filtering with Bootstrap PF

```python
import numpy as np
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig

# Model
sv = StochasticVolatility(
    variant="basic",
    params={"mu": 0.0, "phi": 0.97, "sigma": 0.15}
)

# Simulate test data
sim = sv.simulate(T=500, seed=42)
y = sim["observations"]
true_h = sim["states"]

# Filter
config = PFConfig(n_particles=1000, seed=42)
pf = BootstrapPF(model=sv, config=config)
result = pf.filter(y)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Mean ESS: {np.mean(result.ess_history):.0f}")
```

### Extracting Volatility Estimates

```python
import matplotlib.pyplot as plt

# Filtered log-volatility
h_filtered = result.filtered_states  # shape (T, n_particles, 1)
h_mean = h_filtered.mean(axis=1).squeeze()
h_q05 = np.quantile(h_filtered.squeeze(), 0.05, axis=1)
h_q95 = np.quantile(h_filtered.squeeze(), 0.95, axis=1)

# Convert to annualized volatility
vol_mean = np.exp(h_mean / 2) * np.sqrt(252) * 100  # percentage

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Returns
axes[0].plot(y, color="steelblue", alpha=0.6, linewidth=0.5)
axes[0].set_ylabel("Returns")
axes[0].set_title("Simulated Returns")

# Filtered volatility vs true
axes[1].fill_between(range(len(h_mean)), h_q05, h_q95, alpha=0.3, color="steelblue")
axes[1].plot(h_mean, color="steelblue", label="Filtered (posterior mean)")
axes[1].plot(true_h.squeeze(), color="red", alpha=0.7, linewidth=0.8, label="True")
axes[1].set_ylabel("Log-volatility $h_t$")
axes[1].set_xlabel("Time")
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## Parameter Estimation with PMMH

Use **Particle Marginal Metropolis-Hastings** ([PMMH](../pmcmc/pmmh.md)) for fully Bayesian estimation of SV parameters.

### Estimating $(\mu, \phi, \sigma_\eta)$

```python
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.pmcmc import PMMH
from particlefilterbox.core.config import PFConfig

# True model
sv_true = StochasticVolatility(
    variant="basic",
    params={"mu": -0.5, "phi": 0.98, "sigma": 0.12}
)
sim = sv_true.simulate(T=2000, seed=42)
y = sim["observations"]

# Model for estimation (initial guess)
sv = StochasticVolatility(variant="basic")

# PMMH configuration
pf_config = PFConfig(n_particles=500, seed=0)
priors = sv.default_prior()

pmmh = PMMH(
    model=sv,
    filter_cls=BootstrapPF,
    pf_config=pf_config,
    priors=priors,
    n_iterations=10000,
    burn_in=2000,
    seed=42,
)

# Run MCMC
chain = pmmh.run(y)

# Posterior summary
for param in ["mu", "phi", "sigma"]:
    samples = chain[param]
    print(f"{param}: mean={samples.mean():.4f}, std={samples.std():.4f}")
```

### Prior Customization

Override the default priors by passing your own:

```python
custom_priors = {
    "mu": {"distribution": "normal", "loc": -1.0, "scale": 2.0},
    "phi": {"distribution": "beta", "a": 50.0, "b": 2.0},     # tighter prior on persistence
    "sigma": {"distribution": "inverse_gamma", "a": 5.0, "b": 0.1},
}

pmmh = PMMH(
    model=sv,
    filter_cls=BootstrapPF,
    pf_config=pf_config,
    priors=custom_priors,
    n_iterations=10000,
    burn_in=2000,
    seed=42,
)
```

---

## Variants in Detail

### SV with Leverage: Asymmetric Volatility

The leverage effect is one of the most robust stylized facts in finance: negative returns increase future volatility more than positive returns of the same magnitude.

```python
sv_lev = StochasticVolatility(
    variant="leverage",
    params={"mu": -1.0, "phi": 0.97, "sigma": 0.15, "rho": -0.5}
)

sim = sv_lev.simulate(T=2000, seed=42)
y = sim["observations"]

# Filter
config = PFConfig(n_particles=1500, seed=42)
pf = BootstrapPF(model=sv_lev, config=config)
result = pf.filter(y)
```

!!! tip "Leverage and Filter Choice"
    The leverage variant introduces cross-correlation between $\varepsilon_t$ and $\eta_{t+1}$, which can reduce the efficiency of the bootstrap proposal. Consider using the [Auxiliary PF](../filters/auxiliary.md) for better performance.

### SV with Jumps: Crash Risk

The jump component captures sudden, large moves that the Gaussian SV model cannot explain:

```python
sv_j = StochasticVolatility(
    variant="jumps",
    params={
        "mu": -1.0, "phi": 0.97, "sigma": 0.15,
        "lambda_jump": 0.05, "mu_jump": -0.5, "sigma_jump": 1.0,
    }
)

sim = sv_j.simulate(T=2000, seed=42)

# The state now includes jump indicators
states = sim["states"]  # shape (2000, 2): [h_t, q_t]
jumps = states[:, 1]
print(f"Jumps detected in simulation: {int(jumps.sum())} / {len(jumps)}")
```

After filtering, you can extract **jump probabilities** from the posterior:

```python
config = PFConfig(n_particles=2000, seed=42)
pf = BootstrapPF(model=sv_j, config=config)
result = pf.filter(sim["observations"])

# Posterior jump probability at each time step
jump_probs = result.filtered_states[:, :, 1].mean(axis=1)  # P(q_t = 1 | y_{1:t})
```

### Factor SV: Multivariate Volatility

For modeling volatility co-movements across multiple assets:

```python
sv_factor = StochasticVolatility(
    variant="factor",
    k_factor_series=3,
    params={
        "mu": -1.0, "phi": 0.95, "sigma": 0.2,
        "phi_0": 0.93, "sigma_0": 0.15, "beta_0": 1.2,
        "phi_1": 0.90, "sigma_1": 0.18, "beta_1": 0.8,
        "phi_2": 0.92, "sigma_2": 0.20, "beta_2": 1.0,
    }
)

sim = sv_factor.simulate(T=1000, seed=42)
print(f"Observations shape: {sim['observations'].shape}")  # (1000, 3)
print(f"States shape: {sim['states'].shape}")              # (1000, 4)
```

---

## Example: Financial Returns Analysis

A complete workflow for analyzing daily equity returns:

```python
import numpy as np
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.pmcmc import PMMH
from particlefilterbox.core.config import PFConfig

# --- 1. Simulate realistic daily returns ---
sv_dgp = StochasticVolatility(
    variant="basic",
    params={"mu": -0.5, "phi": 0.98, "sigma": 0.12}
)
sim = sv_dgp.simulate(T=2520, seed=42)  # ~10 years of daily data
returns = sim["observations"]

# --- 2. Filter with known parameters ---
config = PFConfig(n_particles=1000, seed=42)
pf = BootstrapPF(model=sv_dgp, config=config)
result = pf.filter(returns)

h_hat = result.filtered_states.mean(axis=1).squeeze()
vol_annual = np.exp(h_hat / 2) * np.sqrt(252)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Average annualized vol: {vol_annual.mean() * 100:.1f}%")

# --- 3. Bayesian estimation with PMMH ---
sv_est = StochasticVolatility(variant="basic")
pf_config = PFConfig(n_particles=300, seed=0)

pmmh = PMMH(
    model=sv_est,
    filter_cls=BootstrapPF,
    pf_config=pf_config,
    priors=sv_est.default_prior(),
    n_iterations=15000,
    burn_in=5000,
    seed=42,
)
chain = pmmh.run(returns)

# --- 4. Posterior summary ---
print("\nPosterior estimates (true values: mu=-0.5, phi=0.98, sigma=0.12):")
for param in ["mu", "phi", "sigma"]:
    samples = chain[param]
    print(f"  {param}: {samples.mean():.4f} +/- {samples.std():.4f}")
```

---

## SV vs GARCH

| Feature | Stochastic Volatility | GARCH |
|:--------|:---------------------|:------|
| **Volatility** | Latent stochastic process | Deterministic function of past data |
| **Estimation** | PMCMC, SMC (particle-based) | MLE (closed-form likelihood) |
| **Flexibility** | Leverage, jumps, factors as natural extensions | Extensions exist but less modular |
| **Forecasting** | Full posterior predictive distribution | Point forecasts + analytic intervals |
| **Computational cost** | Higher (particle methods) | Lower (analytic) |
| **Multivariate** | Factor model scales naturally | DCC, BEKK --- challenging in high dimensions |

!!! tip "When to use SV over GARCH"
    Use SV when you need **uncertainty quantification** (full posterior of volatility path), **Bayesian estimation** of structural parameters, or when the model includes features like jumps, leverage, or latent factors. Use GARCH when computational speed is the priority and a point estimate of volatility is sufficient.

---

## See Also

- [Bootstrap PF](../filters/bootstrap.md) --- Default filter for SV models
- [Auxiliary PF](../filters/auxiliary.md) --- Better efficiency for SV with leverage
- [PMMH](../pmcmc/pmmh.md) --- Bayesian parameter estimation
- [Particle Gibbs](../pmcmc/particle-gibbs.md) --- Alternative PMCMC for joint state-parameter estimation
- [Jump-Diffusion](jump-diffusion.md) --- Related model for asset prices with discontinuities
