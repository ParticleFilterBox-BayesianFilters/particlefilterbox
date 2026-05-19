---
title: Count Data
description: "Poisson and Negative Binomial state-space models for discrete observation data"
---

# Count Data

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `CountModel` |
    | **Import** | `from particlefilterbox.models import CountModel` |
    | **Variants** | `poisson`, `negative-binomial`, `zero-inflated` |
    | **State** | Continuous log-intensity $x_t$ (1D) |
    | **Observation** | Discrete counts $y_t \in \{0, 1, 2, \ldots\}$ |
    | **Recommended filter** | [Bootstrap PF](../filters/bootstrap.md) / [Auxiliary PF](../filters/auxiliary.md) |
    | **References** | Davis, Dunsmuir & Streett (2003); Durbin & Koopman (2012); Frühwirth-Schnatter & Wagner (2006) |

---

## Overview

**Count Data state-space models** handle situations where observations are non-negative integers --- event counts, number of trades, disease cases, goal counts --- while the underlying intensity evolves as a continuous latent process.

The key challenge is the **mismatch between discrete observations and continuous state**: the observation likelihood is non-Gaussian (Poisson, Negative Binomial), so the Kalman filter cannot be applied directly. Particle filters provide exact inference without Gaussian approximations.

particlefilterbox provides three variants: Poisson, Negative Binomial (for overdispersed counts), and Zero-Inflated Poisson.

---

## Mathematical Framework

### Poisson State-Space Model

The canonical specification:

**State equation** (log-intensity follows a random walk or AR process):

$$
x_t = \mu + \phi (x_{t-1} - \mu) + \sigma_\eta \, \eta_t, \qquad \eta_t \sim \mathcal{N}(0, 1)
$$

**Observation equation** (Poisson with log-link):

$$
y_t \mid x_t \sim \text{Poisson}(\lambda_t), \qquad \lambda_t = \exp(x_t)
$$

The log-link ensures positivity of the intensity $\lambda_t$.

| Parameter | Symbol | Description | Typical Range |
|:----------|:-------|:------------|:--------------|
| Level | $\mu$ | Long-run mean of log-intensity | depends on application |
| Persistence | $\phi$ | AR coefficient | $[0.8, 0.999]$ |
| Volatility | $\sigma_\eta$ | Innovation std of log-intensity | $[0.05, 0.5]$ |

!!! note "Why log-intensity?"
    Working in the log space ($x_t = \log \lambda_t$) allows the latent state to take any real value while ensuring the Poisson rate $\lambda_t = \exp(x_t)$ is always positive. This is analogous to the log-volatility parameterization in SV models.

### Negative Binomial State-Space Model

For **overdispersed counts** (variance exceeds the mean), replace Poisson with Negative Binomial:

$$
y_t \mid x_t \sim \text{NegBin}(r, p_t), \qquad p_t = \frac{r}{r + \exp(x_t)}
$$

where $r > 0$ is the dispersion parameter. The mean and variance are:

$$
\mathbb{E}[y_t \mid x_t] = \exp(x_t), \qquad \text{Var}(y_t \mid x_t) = \exp(x_t) + \frac{\exp(2 x_t)}{r}
$$

As $r \to \infty$, the Negative Binomial converges to the Poisson.

| Parameter | Symbol | Default | Description |
|:----------|:-------|:--------|:------------|
| Dispersion | $r$ | $10.0$ | Controls overdispersion ($r \to \infty$ = Poisson) |

### Zero-Inflated Poisson

For data with excess zeros (e.g., rare events):

$$
y_t \mid x_t \sim
\begin{cases}
0 & \text{with probability } \pi \\
\text{Poisson}(\exp(x_t)) & \text{with probability } 1 - \pi
\end{cases}
$$

The zero-inflation parameter $\pi \in [0, 1)$ captures the fraction of "structural zeros" beyond what the Poisson distribution predicts.

---

## Particle Filtering for Count Data

### Why Particle Filters?

The Poisson (and NegBin) observation likelihood is non-Gaussian, so:

- The **Kalman filter** cannot be applied directly
- **Extended KF** linearization is poor for discrete observations
- **Particle filters** evaluate the exact likelihood $p(y_t \mid x_t)$ without approximation

### Weight Computation

For the Bootstrap PF, the importance weight at time $t$ is:

$$
w_t^{(i)} \propto p(y_t \mid x_t^{(i)}) = \frac{\lambda_t^{y_t} \, e^{-\lambda_t}}{y_t!}, \qquad \lambda_t = \exp(x_t^{(i)})
$$

!!! tip "Numerical Stability"
    Always compute log-weights to avoid overflow:
    $$
    \log w_t^{(i)} = y_t \, x_t^{(i)} - \exp(x_t^{(i)}) - \log(y_t!)
    $$
    particlefilterbox handles this automatically.

### Auxiliary PF for Count Data

The [Auxiliary PF](../filters/auxiliary.md) is particularly effective for count data because it pre-selects particles based on a first-stage approximation of $p(y_t \mid x_t)$, concentrating particles in regions of high likelihood:

```python
from particlefilterbox.filters import AuxiliaryPF

config = PFConfig(n_particles=1000, seed=42)
apf = AuxiliaryPF(model=count_model, config=config)
result = apf.filter(y)
```

---

## API

### Constructor

```python
from particlefilterbox.models import CountModel

# Poisson state-space model
poisson = CountModel(
    variant="poisson",
    params={"mu": 2.0, "phi": 0.95, "sigma_eta": 0.1},
)

# Negative Binomial for overdispersed counts
negbin = CountModel(
    variant="negative-binomial",
    params={"mu": 2.0, "phi": 0.95, "sigma_eta": 0.1, "r": 5.0},
)

# Zero-inflated Poisson
zip_model = CountModel(
    variant="zero-inflated",
    params={"mu": 1.5, "phi": 0.90, "sigma_eta": 0.15, "pi": 0.2},
)
```

### Parameters by Variant

=== "poisson"

    | Parameter | Key | Default | Prior |
    |:----------|:----|:--------|:------|
    | $\mu$ | `mu` | $2.0$ | $\mathcal{N}(0, 5)$ |
    | $\phi$ | `phi` | $0.95$ | $\text{Beta}(20, 1.5)$ |
    | $\sigma_\eta$ | `sigma_eta` | $0.1$ | $\text{InvGamma}(2.5, 0.025)$ |

=== "negative-binomial"

    | Parameter | Key | Default | Prior |
    |:----------|:----|:--------|:------|
    | $\mu$ | `mu` | $2.0$ | $\mathcal{N}(0, 5)$ |
    | $\phi$ | `phi` | $0.95$ | $\text{Beta}(20, 1.5)$ |
    | $\sigma_\eta$ | `sigma_eta` | $0.1$ | $\text{InvGamma}(2.5, 0.025)$ |
    | $r$ | `r` | $10.0$ | $\text{Gamma}(2, 5)$ |

=== "zero-inflated"

    | Parameter | Key | Default | Prior |
    |:----------|:----|:--------|:------|
    | $\mu$ | `mu` | $1.5$ | $\mathcal{N}(0, 5)$ |
    | $\phi$ | `phi` | $0.90$ | $\text{Beta}(20, 1.5)$ |
    | $\sigma_\eta$ | `sigma_eta` | $0.15$ | $\text{InvGamma}(2.5, 0.025)$ |
    | $\pi$ | `pi` | $0.2$ | $\text{Beta}(2, 8)$ |

### Simulation

```python
# Simulate count data
poisson = CountModel(
    variant="poisson",
    params={"mu": 2.0, "phi": 0.95, "sigma_eta": 0.1},
)
sim = poisson.simulate(T=500, seed=42)

counts = sim["observations"]   # shape (500, 1), integer-valued
log_intensity = sim["states"]  # shape (500, 1), continuous
```

---

## Filtering

### Basic Filtering

```python
import numpy as np
from particlefilterbox.models import CountModel
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig

# Poisson model
model = CountModel(
    variant="poisson",
    params={"mu": 2.0, "phi": 0.95, "sigma_eta": 0.1},
)

# Simulate
sim = model.simulate(T=300, seed=42)
y = sim["observations"]
true_x = sim["states"]

# Filter
config = PFConfig(n_particles=1000, seed=42)
pf = BootstrapPF(model=model, config=config)
result = pf.filter(y)

# Extract filtered intensity
x_filtered = result.filtered_states.mean(axis=1).squeeze()
lambda_filtered = np.exp(x_filtered)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Mean ESS: {np.mean(result.ess_history):.0f}")
```

### Visualizing Filtered Intensity

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Count observations
axes[0].bar(range(len(y)), y.squeeze(), color="steelblue", alpha=0.6, width=1.0)
axes[0].set_ylabel("Count $y_t$")
axes[0].set_title("Observed Counts")

# Filtered intensity vs true
true_lambda = np.exp(true_x.squeeze())
axes[1].plot(true_lambda, color="red", alpha=0.7, linewidth=0.8, label="True $\\lambda_t$")
axes[1].plot(lambda_filtered, color="steelblue", label="Filtered $\\lambda_t$")

# 90% credible interval
x_q05 = np.quantile(result.filtered_states.squeeze(), 0.05, axis=1)
x_q95 = np.quantile(result.filtered_states.squeeze(), 0.95, axis=1)
axes[1].fill_between(range(len(x_filtered)),
                     np.exp(x_q05), np.exp(x_q95),
                     alpha=0.2, color="steelblue")

axes[1].set_ylabel("Intensity $\\lambda_t$")
axes[1].set_xlabel("Time")
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## Parameter Estimation with PMMH

```python
from particlefilterbox.models import CountModel
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.pmcmc import PMMH
from particlefilterbox.core.config import PFConfig

# True model
true_model = CountModel(
    variant="poisson",
    params={"mu": 2.5, "phi": 0.97, "sigma_eta": 0.08},
)
sim = true_model.simulate(T=1000, seed=42)
y = sim["observations"]

# Estimation
model = CountModel(variant="poisson")
pf_config = PFConfig(n_particles=500, seed=0)

pmmh = PMMH(
    model=model,
    filter_cls=BootstrapPF,
    pf_config=pf_config,
    priors=model.default_prior(),
    n_iterations=12000,
    burn_in=4000,
    seed=42,
)

chain = pmmh.run(y)

print("Posterior estimates (true: mu=2.5, phi=0.97, sigma_eta=0.08):")
for param in ["mu", "phi", "sigma_eta"]:
    samples = chain[param]
    print(f"  {param}: {samples.mean():.4f} +/- {samples.std():.4f}")
```

---

## Example: Epidemiological Surveillance

Modeling weekly disease case counts with time-varying transmission intensity:

```python
import numpy as np
from particlefilterbox.models import CountModel
from particlefilterbox.filters import AuxiliaryPF
from particlefilterbox.pmcmc import PMMH
from particlefilterbox.core.config import PFConfig

# --- 1. Disease surveillance model ---
# Log-intensity follows an AR(1) process, cases are Poisson
epi_model = CountModel(
    variant="poisson",
    params={
        "mu": 3.0,         # baseline: ~exp(3) ≈ 20 cases/week
        "phi": 0.90,       # moderate persistence
        "sigma_eta": 0.15, # smooth intensity changes
    },
)

# --- 2. Simulate weekly case counts ---
sim = epi_model.simulate(T=200, seed=42)  # ~4 years of weekly data
cases = sim["observations"]

# --- 3. Filter with Auxiliary PF ---
config = PFConfig(n_particles=1500, seed=42)
apf = AuxiliaryPF(model=epi_model, config=config)
result = apf.filter(cases)

# --- 4. Estimate reproduction-like intensity ---
x_mean = result.filtered_states.mean(axis=1).squeeze()
intensity = np.exp(x_mean)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Average weekly cases (filtered): {intensity.mean():.1f}")
print(f"Peak intensity: {intensity.max():.1f} cases/week")

# --- 5. Bayesian estimation ---
model_est = CountModel(variant="poisson")
pf_config = PFConfig(n_particles=500, seed=0)

pmmh = PMMH(
    model=model_est,
    filter_cls=AuxiliaryPF,
    pf_config=pf_config,
    priors=model_est.default_prior(),
    n_iterations=10000,
    burn_in=3000,
    seed=42,
)
chain = pmmh.run(cases)

print("\nPosterior estimates:")
for param in ["mu", "phi", "sigma_eta"]:
    samples = chain[param]
    print(f"  {param}: {samples.mean():.4f} +/- {samples.std():.4f}")
```

---

## Example: Trade Counts in Finance

Modeling intraday trade counts with overdispersion:

```python
import numpy as np
from particlefilterbox.models import CountModel
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig

# Negative Binomial handles overdispersion in trade counts
trade_model = CountModel(
    variant="negative-binomial",
    params={
        "mu": 4.0,         # baseline: ~exp(4) ≈ 55 trades/interval
        "phi": 0.98,       # highly persistent activity
        "sigma_eta": 0.05, # smooth evolution
        "r": 3.0,          # substantial overdispersion
    },
)

sim = trade_model.simulate(T=500, seed=42)
trades = sim["observations"]

config = PFConfig(n_particles=1000, seed=42)
pf = BootstrapPF(model=trade_model, config=config)
result = pf.filter(trades)

# Trading intensity
intensity = np.exp(result.filtered_states.mean(axis=1).squeeze())
print(f"Mean filtered trades/interval: {intensity.mean():.1f}")
print(f"Overdispersion ratio: {trades.var() / trades.mean():.2f}")
```

---

## Filter Recommendations

| Scenario | Recommended Filter | Particles | Notes |
|:---------|:-------------------|:----------|:------|
| Poisson with smooth intensity | [Auxiliary PF](../filters/auxiliary.md) | 500--1000 | Pre-selection helps with discrete likelihood |
| High-count data ($\lambda > 50$) | [Bootstrap PF](../filters/bootstrap.md) | 500--1000 | Poisson approaches Gaussian for large $\lambda$ |
| Overdispersed (NegBin) | [Bootstrap PF](../filters/bootstrap.md) | 1000--2000 | Heavier tails need more particles |
| Zero-inflated | [Bootstrap PF](../filters/bootstrap.md) | 1000--2000 | Bimodal likelihood at zero |
| Parameter estimation | [PMMH](../pmcmc/pmmh.md) + Auxiliary PF | 300--500 | Auxiliary PF gives lower-variance likelihood |

---

## See Also

- [Bootstrap PF](../filters/bootstrap.md) --- Standard filter for count data models
- [Auxiliary PF](../filters/auxiliary.md) --- Improved efficiency via pre-selection
- [PMMH](../pmcmc/pmmh.md) --- Bayesian estimation of model parameters
- [Stochastic Volatility](stochastic-volatility.md) --- Related continuous-observation model with similar state dynamics
- [Bounded State-Space](bounded.md) --- Another non-Gaussian observation model
