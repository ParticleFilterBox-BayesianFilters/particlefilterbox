---
title: Mixture Models
description: "Sequential mixture models with Dirichlet process priors for online clustering and anomaly detection"
---

# Mixture Models

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `MixtureModel` |
    | **Import** | `from particlefilterbox.models import MixtureModel` |
    | **Variants** | `finite`, `dirichlet-process` |
    | **State** | Component assignments $c_t$ + component parameters $\theta_k$ |
    | **Observation** | Continuous $y_t$ from mixture of distributions |
    | **Recommended filter** | [Bootstrap PF](../filters/bootstrap.md) / [SMC Sampler](../smc/smc-sampler.md) |
    | **References** | Fearnhead (2004); Carvalho et al. (2010); Chopin & Pelgrin (2004) |

---

## Overview

**Mixture models** in a sequential setting arise when observations come from a mixture of distributions whose components evolve over time. The particle filter maintains a population of hypotheses about which component generated each observation and how the component parameters are changing.

Key applications include:

- **Online clustering**: assigning new data points to clusters in real time
- **Anomaly detection**: identifying observations that don't fit any existing component
- **Speaker diarization**: segmenting audio by speaker identity
- **Financial regime detection**: identifying distinct market behaviors

particlefilterbox supports both **finite mixtures** (fixed $K$ components) and **Dirichlet process mixtures** (infinite, data-driven number of components).

---

## Mathematical Framework

### Finite Mixture Model

At each time step, the observation is drawn from one of $K$ components:

$$
y_t \sim \sum_{k=1}^{K} \pi_k \, f(y_t \mid \theta_k)
$$

where $\pi_k$ are mixing weights ($\sum_k \pi_k = 1$) and $\theta_k$ are component parameters.

**Component assignment** (latent):

$$
c_t \sim \text{Categorical}(\pi_1, \ldots, \pi_K)
$$

**Observation given component**:

$$
y_t \mid c_t = k \sim f(\cdot \mid \theta_k)
$$

For Gaussian mixtures:

$$
f(y \mid \theta_k) = \mathcal{N}(y \mid \mu_k, \sigma_k^2)
$$

**Sequential evolution**: the parameters and/or weights evolve over time:

$$
\begin{aligned}
\pi_t &\sim \text{Dirichlet}(\alpha \, \pi_{t-1}) \\[4pt]
\mu_{k,t} &= \mu_{k,t-1} + \sigma_\mu \, \eta_{k,t} \\[4pt]
\log \sigma_{k,t} &= \log \sigma_{k,t-1} + \sigma_v \, \nu_{k,t}
\end{aligned}
$$

| Component | Symbol | Description |
|:----------|:-------|:------------|
| Mixing weights | $\pi_k$ | Probability of component $k$ |
| Component means | $\mu_k$ | Location of each component |
| Component variances | $\sigma_k^2$ | Spread of each component |
| Concentration | $\alpha$ | Controls weight evolution speed |
| Assignment | $c_t$ | Which component generated $y_t$ |

### Dirichlet Process Mixture

The **Dirichlet Process (DP)** mixture allows an unbounded number of components, with new components created as needed:

$$
G \sim \text{DP}(\alpha, G_0)
$$

where $\alpha > 0$ is the concentration parameter and $G_0$ is the base distribution.

The **Chinese Restaurant Process (CRP)** gives the predictive assignment rule:

$$
P(c_t = k \mid c_{1:t-1}) =
\begin{cases}
\dfrac{n_k}{t - 1 + \alpha} & \text{existing component } k \\[8pt]
\dfrac{\alpha}{t - 1 + \alpha} & \text{new component}
\end{cases}
$$

where $n_k = \sum_{s < t} \mathbf{1}(c_s = k)$ is the number of previous observations assigned to component $k$.

!!! note "Expected number of components"
    Under the CRP, the expected number of distinct components after $T$ observations is:
    $$
    \mathbb{E}[K_T] \approx \alpha \log\!\left(1 + \frac{T}{\alpha}\right)
    $$
    For $\alpha = 1$ and $T = 1000$, expect $\sim 7$ components.

### Particle Representation

Each particle $i$ carries a full clustering hypothesis:

$$
\text{Particle } i = \left(c_{1:t}^{(i)}, \; \{\theta_k^{(i)}\}_{k=1}^{K_t^{(i)}}, \; \pi^{(i)}\right)
$$

The particle weight is:

$$
w_t^{(i)} \propto \sum_{k=1}^{K_t^{(i)} + 1} P(c_t = k \mid c_{1:t-1}^{(i)}) \, f(y_t \mid \theta_k^{(i)})
$$

where the sum includes the possibility of a new component (for DP mixtures).

---

## SMC for Mixture Models

### Sequential Importance Sampling

At each time step $t$, for each particle $i$:

1. **Assign** $c_t^{(i)}$: sample from $P(c_t \mid c_{1:t-1}^{(i)}, y_t)$ (optimal allocation uses the posterior)
2. **Update** $\theta_{c_t}^{(i)}$: update the assigned component's sufficient statistics
3. **Create** new component if $c_t^{(i)} = K_t^{(i)} + 1$ (DP only): sample $\theta_{\text{new}} \sim G_0$
4. **Weight**: compute importance weight

### Sufficient Statistics (Conjugate Case)

For Gaussian components with a Normal-Inverse-Gamma prior, maintain sufficient statistics per component:

$$
\theta_k \mid y_{1:t} \propto \text{NIG}(\mu_n, \kappa_n, \alpha_n, \beta_n)
$$

where:

$$
\begin{aligned}
\kappa_n &= \kappa_0 + n_k, \qquad \mu_n = \frac{\kappa_0 \mu_0 + n_k \bar{y}_k}{\kappa_n} \\[4pt]
\alpha_n &= \alpha_0 + \frac{n_k}{2}, \qquad \beta_n = \beta_0 + \frac{1}{2} S_k + \frac{\kappa_0 n_k (\bar{y}_k - \mu_0)^2}{2 \kappa_n}
\end{aligned}
$$

This avoids storing all past data --- only sufficient statistics are needed.

!!! tip "Conjugacy for efficiency"
    When using conjugate priors (Normal-Inverse-Gamma for Gaussian mixtures), component parameters can be marginalized analytically. Each particle only stores the sufficient statistics, not point-estimate parameters. This dramatically reduces particle impoverishment.

---

## API

### Constructor

```python
from particlefilterbox.models import MixtureModel

# Finite Gaussian mixture with K=3 components
finite_mix = MixtureModel(
    variant="finite",
    n_components=3,
    component_dist="gaussian",
    params={
        "alpha": 1.0,       # Dirichlet concentration for weight evolution
        "sigma_mu": 0.01,   # component mean drift
    },
)

# Dirichlet process mixture (infinite components)
dp_mix = MixtureModel(
    variant="dirichlet-process",
    component_dist="gaussian",
    params={
        "alpha": 1.0,       # DP concentration parameter
        "mu_0": 0.0,        # base distribution mean
        "kappa_0": 0.1,     # base distribution precision scaling
        "alpha_0": 2.0,     # base distribution shape
        "beta_0": 1.0,      # base distribution rate
    },
)

# Finite mixture with Student-t components (heavier tails)
robust_mix = MixtureModel(
    variant="finite",
    n_components=3,
    component_dist="student-t",
    params={"alpha": 1.0, "df": 5.0},
)
```

### Parameters by Variant

=== "finite"

    | Parameter | Key | Default | Description |
    |:----------|:----|:--------|:------------|
    | $K$ | `n_components` | $3$ | Number of mixture components |
    | $\alpha$ | `alpha` | $1.0$ | Dirichlet concentration for weights |
    | $\sigma_\mu$ | `sigma_mu` | $0.01$ | Component mean random walk noise |
    | Distribution | `component_dist` | `gaussian` | Component distribution type |

=== "dirichlet-process"

    | Parameter | Key | Default | Description |
    |:----------|:----|:--------|:------------|
    | $\alpha$ | `alpha` | $1.0$ | DP concentration (controls new clusters) |
    | $\mu_0$ | `mu_0` | $0.0$ | Base distribution mean |
    | $\kappa_0$ | `kappa_0` | $0.1$ | Base distribution precision scaling |
    | $\alpha_0$ | `alpha_0` | $2.0$ | Base distribution shape (NIG) |
    | $\beta_0$ | `beta_0` | $1.0$ | Base distribution rate (NIG) |

### Simulation

```python
# Simulate from a 3-component Gaussian mixture
mix = MixtureModel(
    variant="finite",
    n_components=3,
    component_dist="gaussian",
    params={
        "weights": [0.4, 0.35, 0.25],
        "means": [-2.0, 0.0, 3.0],
        "stds": [0.5, 0.8, 0.4],
    },
)
sim = mix.simulate(T=500, seed=42)

observations = sim["observations"]   # shape (500, 1)
assignments = sim["assignments"]     # shape (500,), integer labels
```

---

## Filtering

### Sequential Clustering with Finite Mixture

```python
import numpy as np
from particlefilterbox.models import MixtureModel
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig

# 3-component Gaussian mixture
mix = MixtureModel(
    variant="finite",
    n_components=3,
    component_dist="gaussian",
    params={
        "weights": [0.4, 0.35, 0.25],
        "means": [-2.0, 0.0, 3.0],
        "stds": [0.5, 0.8, 0.4],
        "alpha": 10.0,
    },
)

sim = mix.simulate(T=300, seed=42)
y = sim["observations"]

config = PFConfig(n_particles=2000, seed=42)
pf = BootstrapPF(model=mix, config=config)
result = pf.filter(y)

# Posterior component probabilities at each time step
comp_probs = result.component_probabilities  # shape (T, K)

# Most likely assignment
assignments_hat = comp_probs.argmax(axis=1)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Classification accuracy: {(assignments_hat == sim['assignments']).mean():.1%}")
```

### Dirichlet Process Mixture for Online Clustering

```python
from particlefilterbox.models import MixtureModel
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig

# DP mixture: number of clusters determined by data
dp_mix = MixtureModel(
    variant="dirichlet-process",
    component_dist="gaussian",
    params={"alpha": 1.0, "mu_0": 0.0, "kappa_0": 0.1, "alpha_0": 2.0, "beta_0": 1.0},
)

# Filter sequentially
config = PFConfig(n_particles=2000, seed=42)
pf = BootstrapPF(model=dp_mix, config=config)
result = pf.filter(y)

# Number of active clusters over time
n_clusters = result.n_active_components  # shape (T,)
print(f"Final number of clusters: {n_clusters[-1]}")
```

### Visualizing Cluster Evolution

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Observations colored by true assignment
colors = ["steelblue", "coral", "forestgreen", "gold", "purple"]
for k in range(mix.n_components):
    mask = sim["assignments"] == k
    axes[0].scatter(np.where(mask)[0], y[mask], c=colors[k],
                    alpha=0.5, s=10, label=f"Component {k+1}")
axes[0].legend()
axes[0].set_ylabel("$y_t$")
axes[0].set_title("Observations (true labels)")

# Posterior component probabilities
for k in range(mix.n_components):
    axes[1].plot(comp_probs[:, k], color=colors[k], alpha=0.7,
                 label=f"$P(c_t={k+1} \\mid y_{{1:t}})$")
axes[1].legend()
axes[1].set_ylabel("Probability")
axes[1].set_title("Filtered Component Probabilities")

# Number of active clusters (for DP mixture)
axes[2].plot(n_clusters, color="darkred", linewidth=1.5)
axes[2].set_ylabel("Active clusters")
axes[2].set_xlabel("Time")
axes[2].set_title("Number of Active Components")

plt.tight_layout()
plt.show()
```

---

## Parameter Estimation with PMMH

```python
from particlefilterbox.models import MixtureModel
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.pmcmc import PMMH
from particlefilterbox.core.config import PFConfig

# True model
true_mix = MixtureModel(
    variant="finite",
    n_components=2,
    component_dist="gaussian",
    params={
        "weights": [0.6, 0.4],
        "means": [-1.0, 2.0],
        "stds": [0.5, 0.8],
    },
)
sim = true_mix.simulate(T=500, seed=42)
y = sim["observations"]

# Estimation
mix_est = MixtureModel(variant="finite", n_components=2, component_dist="gaussian")
pf_config = PFConfig(n_particles=1000, seed=0)

pmmh = PMMH(
    model=mix_est,
    filter_cls=BootstrapPF,
    pf_config=pf_config,
    priors=mix_est.default_prior(),
    n_iterations=10000,
    burn_in=3000,
    seed=42,
)

chain = pmmh.run(y)

print("Posterior estimates:")
for param in ["mu_1", "mu_2", "sigma_1", "sigma_2", "pi_1"]:
    samples = chain[param]
    print(f"  {param}: {samples.mean():.4f} +/- {samples.std():.4f}")
```

!!! warning "Label Switching in Mixtures"
    Like regime-switching models, mixture models suffer from **label switching**: permuting component indices gives an equivalent model. Impose ordering constraints (e.g., $\mu_1 < \mu_2 < \ldots$) or use post-hoc relabeling algorithms (Stephens, 2000).

---

## Example: Anomaly Detection in Sensor Data

Using a DP mixture for online anomaly detection --- observations that create new clusters are flagged as anomalies:

```python
import numpy as np
from particlefilterbox.models import MixtureModel
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig

# --- 1. DP mixture model ---
dp_model = MixtureModel(
    variant="dirichlet-process",
    component_dist="gaussian",
    params={"alpha": 0.5, "mu_0": 0.0, "kappa_0": 0.01, "alpha_0": 3.0, "beta_0": 1.0},
)

# --- 2. Simulate sensor data with anomalies ---
np.random.seed(42)
T = 500
normal_data = np.random.normal(0.0, 1.0, T)
# Inject anomalies at random positions
anomaly_idx = np.random.choice(T, size=15, replace=False)
normal_data[anomaly_idx] += np.random.normal(5.0, 1.0, len(anomaly_idx))
y = normal_data.reshape(-1, 1)

# --- 3. Filter ---
config = PFConfig(n_particles=2000, seed=42)
pf = BootstrapPF(model=dp_model, config=config)
result = pf.filter(y)

# --- 4. Anomaly scoring ---
# Observations assigned to new/singleton clusters are anomalies
anomaly_scores = result.new_component_probability  # P(new cluster | y_{1:t})
threshold = 0.5
detected = anomaly_scores > threshold

print(f"True anomalies: {len(anomaly_idx)}")
print(f"Detected anomalies: {detected.sum()}")
print(f"Active clusters at end: {result.n_active_components[-1]}")
```

---

## Filter Recommendations

| Scenario | Recommended Filter | Particles | Notes |
|:---------|:-------------------|:----------|:------|
| Finite mixture, $K \leq 5$ | [Bootstrap PF](../filters/bootstrap.md) | 1000--2000 | Need enough particles per component |
| DP mixture, online clustering | [Bootstrap PF](../filters/bootstrap.md) | 2000--5000 | More particles for exploring new clusters |
| Static mixture (no evolution) | [SMC Sampler](../smc/smc-sampler.md) | 1000--2000 | Better for batch inference |
| Parameter estimation | [PMMH](../pmcmc/pmmh.md) + Bootstrap PF | 500--1000 | With label-switching constraints |

!!! tip "Particle Count Rule of Thumb"
    For a $K$-component mixture, use at least $500 \times K$ particles to ensure adequate representation of all components. For DP mixtures, use at least 2000 particles to allow exploration of new clusters.

---

## See Also

- [Bootstrap PF](../filters/bootstrap.md) --- Standard filter for sequential mixture inference
- [SMC Sampler](../smc/smc-sampler.md) --- Batch inference for static mixture models
- [PMMH](../pmcmc/pmmh.md) --- Bayesian estimation of mixture parameters
- [Regime-Switching](regime.md) --- Related model with discrete latent state (Markov switching)
- [Count Data](count.md) --- Can be combined with mixtures (zero-inflated models)
