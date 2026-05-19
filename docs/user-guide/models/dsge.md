---
title: DSGE Models
description: "Dynamic Stochastic General Equilibrium models with particle filtering, kalmanbox integration, and Bayesian estimation"
---

# DSGE Models

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `DSGE` |
    | **Import** | `from particlefilterbox.models import DSGE` |
    | **Approximation** | First-order (linear) and second-order (nonlinear) |
    | **State** | Multivariate continuous (deviations from steady state) |
    | **Observation** | Multivariate continuous (macro aggregates) |
    | **Recommended filter** | [Rao-Blackwellized PF](../filters/rbpf.md) (1st order) / [Bootstrap PF](../filters/bootstrap.md) (2nd order) |
    | **References** | Fernandez-Villaverde & Rubio-Ramirez (2007); An & Schorfheide (2007); Herbst & Schorfheide (2015) |

---

## Overview

The `DSGE` class provides a state-space representation for **linearized and second-order approximated DSGE models**. Rather than solving the DSGE from structural equations, it takes the **solution matrices** as input --- the output of a first- or second-order perturbation solver.

Key features:

- **First-order (linear) approximation**: Gaussian state-space model amenable to Rao-Blackwellization
- **Second-order (nonlinear) approximation**: Captures risk premia and precautionary savings
- **Zero Lower Bound (ZLB)**: Occasionally binding constraint on interest rates
- **Impulse Response Functions**: Computed via particle filter simulation
- **kalmanbox integration**: Linear components handled exactly via Kalman filter

---

## Mathematical Framework

### Linearized DSGE (First Order)

A first-order perturbation solution yields the linear state-space form:

$$
\begin{aligned}
x_t &= A \, x_{t-1} + B \, \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, I_{k_\varepsilon}) \\[6pt]
y_t &= C \, x_t + u_t, \qquad u_t \sim \mathcal{N}(0, H H')
\end{aligned}
$$

where:

| Matrix | Shape | Description |
|:-------|:------|:------------|
| $A$ | $(k_x \times k_x)$ | State transition (policy function derivatives) |
| $B$ | $(k_x \times k_\varepsilon)$ | Shock impact matrix |
| $C$ | $(k_y \times k_x)$ | Observation (measurement) matrix |
| $H$ | $(k_y \times k_y)$ | Measurement error std (optional) |

!!! tip "kalmanbox Integration"
    For first-order (linear) DSGE models, the state-space is fully Gaussian. Use the **[Rao-Blackwellized PF](../filters/rbpf.md)** which delegates the linear substate to a Kalman filter (via [kalmanbox](https://github.com/guhaase/kalmanbox)) while handling nonlinear components with particles. This dramatically reduces the number of particles needed.

### Second-Order Approximation

The second-order solution adds a quadratic correction:

$$
x_t = A \, x_{t-1} + B \, \varepsilon_t + \frac{1}{2} \sigma^2 \sum_{j,k} Q_{ijk} \, x_{t-1,j} \, x_{t-1,k}
$$

where $Q$ is the tensor of second-order derivatives from the perturbation solution, and $\sigma$ is the perturbation parameter.

The quadratic term captures:

- **Risk premia**: Agents' compensation for uncertainty
- **Precautionary savings**: Prudent behavior under uncertainty
- **Asymmetric responses**: Booms and recessions differ in magnitude

### Zero Lower Bound (ZLB)

When `zlb=True`, the model enforces a non-negativity constraint on the interest rate observation:

$$
y_t^{(r)} = \max\!\big(0, \; C_r \, x_t + u_t^{(r)}\big)
$$

This introduces a kink that makes the model nonlinear even with a first-order approximation, requiring particle methods.

---

## API

### Constructor

```python
from particlefilterbox.models import DSGE
import numpy as np

# Default: simple 3-equation New Keynesian model
dsge = DSGE()

# From explicit matrices
A = np.array([
    [0.8,  0.1, 0.0],   # output gap
    [-0.2, 0.9, 0.3],   # inflation
    [0.0, -0.1, 0.7],   # interest rate
])
B = np.eye(3) * 0.1     # shock impacts
C = np.eye(3)            # observe all states

dsge = DSGE(A=A, B=B, C=C)
```

### From Matrices (Class Method)

```python
dsge = DSGE.from_matrices(
    A=A, B=B, C=C,
    H=np.eye(3) * 0.05,  # measurement error
    order=1,
)
```

### Second-Order Model

```python
Q = np.zeros((3, 3, 3))
Q[0, 0, 0] = -0.05  # nonlinear output gap dynamics
Q[1, 1, 0] = 0.02   # nonlinear inflation-output interaction

dsge_2nd = DSGE(
    A=A, B=B, C=C,
    order=2,
    sigma_scale=1.0,
    quadratic_terms=Q,
)
```

### ZLB Model

```python
dsge_zlb = DSGE(
    A=A, B=B, C=C,
    zlb=True,
    zlb_index=2,  # interest rate is the 3rd observable
)
```

### Model Properties

| Property | Type | Description |
|:---------|:-----|:------------|
| `k_states` | `int` | Number of state variables (from $A$) |
| `k_shocks` | `int` | Number of structural shocks (from $B$) |
| `k_obs` | `int` | Number of observables (from $C$) |
| `order` | `int` | Approximation order (1 or 2) |
| `param_names` | `list[str]` | Estimable parameter names |

---

## Example: 3-Equation New Keynesian Model

The textbook NK model consists of:

1. **IS curve** (output gap): $\hat{y}_t = E_t[\hat{y}_{t+1}] - \frac{1}{\sigma}(\hat{r}_t - E_t[\hat{\pi}_{t+1}]) + g_t$
2. **Phillips curve** (inflation): $\hat{\pi}_t = \beta E_t[\hat{\pi}_{t+1}] + \kappa \hat{y}_t + u_t$
3. **Taylor rule** (interest rate): $\hat{r}_t = \phi_\pi \hat{\pi}_t + \phi_y \hat{y}_t + v_t$

After solving with a first-order perturbation method, the solution takes the form $x_t = A x_{t-1} + B \varepsilon_t$.

```python
import numpy as np
from particlefilterbox.models import DSGE
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig

# Calibrated transition matrix (from perturbation solution)
A = np.array([
    [0.85,  0.10, -0.05],  # output gap
    [-0.15, 0.90,  0.20],  # inflation
    [0.10, -0.05,  0.75],  # interest rate
])
B = np.diag([0.008, 0.005, 0.003])  # shock std: demand, cost-push, monetary
C = np.eye(3)                        # observe all three
H = np.diag([0.001, 0.001, 0.001])  # small measurement error

# Create model
dsge = DSGE.from_matrices(A=A, B=B, C=C, H=H, order=1)

# Simulate quarterly data (200 quarters = 50 years)
sim = dsge.simulate(T=200, seed=42)
y = sim["observations"]

print(f"States: {dsge.k_states}, Shocks: {dsge.k_shocks}, Obs: {dsge.k_obs}")

# Filter
config = PFConfig(n_particles=500, seed=42)
pf = BootstrapPF(model=dsge, config=config)
result = pf.filter(y)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
```

---

## kalmanbox Integration: Rao-Blackwellized Filtering

For first-order DSGE models, the entire state-space is linear-Gaussian. The **Rao-Blackwellized Particle Filter** (RBPF) exploits this by delegating the linear substate to a Kalman filter, reducing variance.

!!! note "When to use RBPF for DSGE"
    Use RBPF when your DSGE model has:

    - A **first-order linear** core with **nonlinear additions** (e.g., ZLB, regime switching)
    - A mix of linear and nonlinear state variables
    - The goal is to reduce the number of particles needed

    For a purely linear first-order DSGE without constraints, the Kalman filter alone (via kalmanbox) is optimal. Particle methods shine when nonlinearities are present.

### RBPF Setup

```python
import numpy as np
from particlefilterbox.models import DSGE
from particlefilterbox.filters import RaoBlackwellizedPF
from particlefilterbox.core.config import PFConfig

# DSGE with ZLB (nonlinear due to constraint)
A = np.array([
    [0.85,  0.10, -0.05],
    [-0.15, 0.90,  0.20],
    [0.10, -0.05,  0.75],
])
B = np.diag([0.008, 0.005, 0.003])
C = np.eye(3)

dsge_zlb = DSGE(
    A=A, B=B, C=C,
    zlb=True,
    zlb_index=2,  # interest rate
)

# Simulate with ZLB episodes
sim = dsge_zlb.simulate(T=200, seed=42)

# Rao-Blackwellized filtering
# Linear substate handled by Kalman (kalmanbox), ZLB nonlinearity by particles
config = PFConfig(n_particles=500, seed=42)
rbpf = RaoBlackwellizedPF(model=dsge_zlb, config=config)
result = rbpf.filter(sim["observations"])

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Mean ESS: {np.mean(result.ess_history):.0f}")
```

### How RBPF Works for DSGE

```mermaid
graph LR
    A[DSGE State x_t] --> B{Decompose}
    B --> C[Linear substate x_t^L]
    B --> D[Nonlinear substate x_t^N]
    C --> E["Kalman Filter (kalmanbox)"]
    D --> F[Particle Filter]
    E --> G[Exact posterior p(x^L | y, x^N)]
    F --> H[Particle approximation p(x^N | y)]
    G --> I[Combined posterior]
    H --> I
```

The decomposition reduces the effective dimensionality of the particle filter, allowing fewer particles to achieve the same accuracy.

---

## Bayesian Estimation with PMMH

Estimate structural parameters of DSGE models using the particle filter likelihood within a Metropolis-Hastings sampler.

```python
import numpy as np
from particlefilterbox.models import DSGE
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.pmcmc import PMMH
from particlefilterbox.core.config import PFConfig

# True model (data-generating process)
A_true = np.array([
    [0.85,  0.10, -0.05],
    [-0.15, 0.90,  0.20],
    [0.10, -0.05,  0.75],
])
B_true = np.diag([0.008, 0.005, 0.003])
C_true = np.eye(3)

dsge_true = DSGE(A=A_true, B=B_true, C=C_true, order=1)
sim = dsge_true.simulate(T=200, seed=42)
y = sim["observations"]

# Estimation model (parameterize shock variances)
dsge_est = DSGE(
    A=A_true, B=B_true, C=C_true, order=1,
    params={"sigma_scale": 1.0},
)

# PMMH
pf_config = PFConfig(n_particles=300, seed=0)
priors = {
    "sigma_scale": {"distribution": "inverse_gamma", "a": 3.0, "b": 1.0},
}

pmmh = PMMH(
    model=dsge_est,
    filter_cls=BootstrapPF,
    pf_config=pf_config,
    priors=priors,
    n_iterations=5000,
    burn_in=1000,
    seed=42,
)
chain = pmmh.run(y)

print(f"sigma_scale: {chain['sigma_scale'].mean():.4f} +/- {chain['sigma_scale'].std():.4f}")
```

---

## Impulse Response Functions

Compute IRFs via Monte Carlo simulation with the particle filter:

```python
import numpy as np
import matplotlib.pyplot as plt
from particlefilterbox.models import DSGE

A = np.array([
    [0.85,  0.10, -0.05],
    [-0.15, 0.90,  0.20],
    [0.10, -0.05,  0.75],
])
B = np.diag([0.008, 0.005, 0.003])
dsge = DSGE(A=A, B=B, C=np.eye(3), order=1)

# IRF to a demand shock (shock index 0)
irf = dsge.impulse_response(shock=0, periods=40, n_particles=5000, seed=42)

# Plot
labels = ["Output Gap", "Inflation", "Interest Rate"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, (ax, label) in enumerate(zip(axes, labels)):
    ax.plot(irf[:, i], color="steelblue", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Response of {label}")
    ax.set_xlabel("Quarters")

plt.suptitle("IRF to a Demand Shock", fontsize=14)
plt.tight_layout()
plt.show()
```

!!! note "IRFs for Nonlinear Models"
    For second-order models, the IRF is **state-dependent** --- the response differs depending on the initial state. The `impulse_response()` method computes the average IRF across many particle draws, which corresponds to the **generalized impulse response function (GIRF)**.

---

## Second-Order Approximation

When the first-order approximation misses important dynamics:

```python
import numpy as np
from particlefilterbox.models import DSGE
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig

A = np.array([
    [0.85,  0.10, -0.05],
    [-0.15, 0.90,  0.20],
    [0.10, -0.05,  0.75],
])
B = np.diag([0.008, 0.005, 0.003])
C = np.eye(3)

# Quadratic correction tensor
Q = np.zeros((3, 3, 3))
Q[0, 0, 0] = -0.05  # output gap self-interaction
Q[1, 1, 0] = 0.02   # inflation-output interaction
Q[2, 0, 0] = 0.01   # monetary policy response to output^2

dsge_2 = DSGE(
    A=A, B=B, C=C,
    order=2,
    sigma_scale=1.0,
    quadratic_terms=Q,
)

# Second-order models REQUIRE particle methods
sim = dsge_2.simulate(T=200, seed=42)
config = PFConfig(n_particles=1000, seed=42)
pf = BootstrapPF(model=dsge_2, config=config)
result = pf.filter(sim["observations"])

print(f"Log-likelihood (2nd order): {result.log_likelihood:.2f}")
```

---

## When to Use Each Approach

| Scenario | Approximation | Filter | Notes |
|:---------|:-------------|:-------|:------|
| Linear DSGE, no constraints | 1st order | Kalman (kalmanbox) | Exact solution, no particles needed |
| Linear DSGE + ZLB | 1st order + ZLB | [RBPF](../filters/rbpf.md) | Particles for ZLB, Kalman for rest |
| Nonlinear DSGE | 2nd order | [Bootstrap PF](../filters/bootstrap.md) | Full particle methods required |
| DSGE + regime switching | 1st order per regime | [Bootstrap PF](../filters/bootstrap.md) | See [NonlinearRegime](regime.md) |
| Bayesian estimation | Any | [PMMH](../pmcmc/pmmh.md) + any PF | Structural parameter inference |

---

## See Also

- [Rao-Blackwellized PF](../filters/rbpf.md) --- Optimal filter for linear DSGE with nonlinear additions
- [Bootstrap PF](../filters/bootstrap.md) --- General-purpose filter for nonlinear DSGE
- [PMMH](../pmcmc/pmmh.md) --- Bayesian estimation of DSGE parameters
- [SMC Sampler](../smc/smc-sampler.md) --- Alternative to PMMH for DSGE estimation
- [Stochastic Volatility](stochastic-volatility.md) --- Financial model with similar filtering setup
