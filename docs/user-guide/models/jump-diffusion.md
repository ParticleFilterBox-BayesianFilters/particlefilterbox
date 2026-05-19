---
title: Jump-Diffusion
description: "Jump-Diffusion models (Merton, Kou, Bates) with discretization, auxiliary particle filtering, and jump detection"
---

# Jump-Diffusion Models

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `JumpDiffusion` |
    | **Import** | `from particlefilterbox.models import JumpDiffusion` |
    | **Variants** | `merton`, `kou`, `bates` |
    | **State** | Continuous (log-price, optionally variance) |
    | **Observation** | Continuous (log-returns) |
    | **Recommended filter** | [Auxiliary PF](../filters/auxiliary.md) |
    | **References** | Merton (1976); Kou (2002); Bates (1996) |

---

## Overview

Jump-Diffusion models extend geometric Brownian motion by adding **discontinuous jumps** --- rare, large movements that cannot be captured by a diffusion process alone. These jumps are essential for modeling:

- **Market crashes** and flash crashes
- **Commodity price spikes** (oil, natural gas)
- **Earnings surprises** and event-driven moves
- **Fat tails** in return distributions

particlefilterbox provides **3 variants** of increasing complexity, all sharing the same API.

!!! warning "Why Auxiliary PF?"
    Jumps are rare events. The standard Bootstrap PF proposes particles "blind" to the observation, so most particles will not propose a jump when one actually occurred. The [Auxiliary PF](../filters/auxiliary.md) uses a look-ahead step that upweights particles likely to match the observation, making it much more efficient for jump detection.

---

## Mathematical Framework

### Merton Model (1976)

The Merton jump-diffusion in continuous time:

$$
\frac{dS_t}{S_t} = \left(\mu - \lambda k\right) dt + \sigma \, dW_t + J_t \, dN_t
$$

where:

- $W_t$ is a standard Brownian motion
- $N_t \sim \text{Poisson}(\lambda)$ is a counting process
- $J_t \sim \mathcal{N}(\mu_J, \sigma_J^2)$ is the jump size
- $k = E[e^{J_t}] - 1 = \exp(\mu_J + \tfrac{1}{2}\sigma_J^2) - 1$ is the compensator

In log-price form, the Euler-Maruyama discretization over step $\Delta t$ is:

$$
\log S_{t+1} = \log S_t + \left(\mu - \tfrac{1}{2}\sigma^2 - \lambda k\right)\Delta t + \sigma \sqrt{\Delta t} \, z_t + \sum_{j=1}^{N_t} J_j
$$

where $z_t \sim \mathcal{N}(0, 1)$ and $N_t \sim \text{Poisson}(\lambda \Delta t)$.

| Parameter | Symbol | Default | Description |
|:----------|:-------|:--------|:------------|
| Drift | $\mu$ | $0.08$ | Expected return (annualized) |
| Diffusion vol | $\sigma$ | $0.2$ | Continuous volatility |
| Jump intensity | $\lambda$ | $3.0$ | Expected jumps per year |
| Jump mean | $\mu_J$ | $-0.02$ | Average jump size (log) |
| Jump std | $\sigma_J$ | $0.05$ | Jump size variability |

### Kou Model (2002)

The Kou model replaces Gaussian jumps with an **asymmetric double-exponential** distribution, better capturing the empirical observation that downward jumps are larger than upward ones:

$$
f_J(x) = p \cdot \eta_1 e^{-\eta_1 x} \mathbf{1}_{x \geq 0} + (1 - p) \cdot \eta_2 e^{\eta_2 x} \mathbf{1}_{x < 0}
$$

where:

| Parameter | Symbol | Default | Description |
|:----------|:-------|:--------|:------------|
| Up probability | $p$ | $0.4$ | Probability of upward jump |
| Up rate | $\eta_1$ | $10.0$ | Rate of upward exponential ($\eta_1 > 1$) |
| Down rate | $\eta_2$ | $5.0$ | Rate of downward exponential ($\eta_2 > 0$) |

The compensator becomes:

$$
k = \frac{p \eta_1}{\eta_1 - 1} + \frac{(1-p) \eta_2}{\eta_2 + 1} - 1
$$

### Bates Model (1996)

The Bates model combines Merton jumps with **Heston stochastic volatility**:

$$
\begin{aligned}
d\log S_t &= \left(\mu - \frac{v_t}{2} - \lambda k\right) dt + \sqrt{v_t} \, dW_t^{(1)} + J_t \, dN_t \\[6pt]
dv_t &= \kappa (\theta - v_t) \, dt + \sigma_v \sqrt{v_t} \, dW_t^{(2)}
\end{aligned}
$$

with $\text{Corr}(dW_t^{(1)}, dW_t^{(2)}) = \rho$.

The state is 2-dimensional: $(\log S_t, v_t)$.

| Parameter | Symbol | Default | Description |
|:----------|:-------|:--------|:------------|
| Mean reversion | $\kappa$ | $5.0$ | Speed of variance reversion |
| Long-run var | $\theta$ | $0.04$ | Long-run variance level |
| Vol-of-vol | $\sigma_v$ | $0.5$ | Volatility of variance |
| Correlation | $\rho$ | $-0.7$ | Leverage effect |

!!! note "Feller Condition"
    The Bates model enforces $v_t > 0$ numerically via `np.maximum(v, 1e-8)`. The Feller condition $2\kappa\theta > \sigma_v^2$ ensures the variance process does not reach zero in continuous time. With Euler-Maruyama discretization, small violations can occur, hence the numerical floor.

---

## Discretization

All variants use **Euler-Maruyama discretization** with configurable time step and substeps:

```python
from particlefilterbox.models import JumpDiffusion

# Daily frequency (default: dt = 1/252)
jd = JumpDiffusion(variant="merton", dt=1/252)

# Weekly frequency with 5 substeps for accuracy
jd_weekly = JumpDiffusion(variant="merton", dt=5/252, n_substeps=5)

# Monthly frequency
jd_monthly = JumpDiffusion(variant="bates", dt=21/252, n_substeps=10)
```

!!! tip "Substeps and Accuracy"
    For the Bates model, the CIR-like variance process benefits from **multiple substeps** per observation interval. A good rule of thumb: use `n_substeps >= 5` when $\Delta t > 1/252$.

---

## API

### Constructor

```python
from particlefilterbox.models import JumpDiffusion

# Merton (default)
jd = JumpDiffusion(
    variant="merton",
    params={
        "mu": 0.08, "sigma": 0.2,
        "lambda_jump": 3.0, "mu_jump": -0.02, "sigma_jump": 0.05,
    },
    dt=1/252,
)

# Kou with asymmetric jumps
jd_kou = JumpDiffusion(
    variant="kou",
    params={
        "mu": 0.08, "sigma": 0.2, "lambda_jump": 3.0,
        "p_up": 0.4, "eta1": 10.0, "eta2": 5.0,
    },
)

# Bates (jumps + stochastic volatility)
jd_bates = JumpDiffusion(
    variant="bates",
    params={
        "mu": 0.08, "kappa": 5.0, "theta": 0.04,
        "sigma_v": 0.5, "rho": -0.7,
        "lambda_jump": 3.0, "mu_jump": -0.02, "sigma_jump": 0.05,
    },
)
```

### Parameters by Variant

=== "merton"

    | Parameter | Key | Default | Prior |
    |:----------|:----|:--------|:------|
    | $\mu$ | `mu` | $0.08$ | $\mathcal{N}(0.05, 0.1)$ |
    | $\sigma$ | `sigma` | $0.2$ | $\text{InvGamma}(5, 0.2)$ |
    | $\lambda$ | `lambda_jump` | $3.0$ | $\text{Gamma}(2, 1)$ |
    | $\mu_J$ | `mu_jump` | $-0.02$ | $\mathcal{N}(0, 0.1)$ |
    | $\sigma_J$ | `sigma_jump` | $0.05$ | $\text{InvGamma}(5, 0.05)$ |

=== "kou"

    | Parameter | Key | Default | Prior |
    |:----------|:----|:--------|:------|
    | $\mu$ | `mu` | $0.08$ | $\mathcal{N}(0.05, 0.1)$ |
    | $\sigma$ | `sigma` | $0.2$ | $\text{InvGamma}(5, 0.2)$ |
    | $\lambda$ | `lambda_jump` | $3.0$ | $\text{Gamma}(2, 1)$ |
    | $p$ | `p_up` | $0.4$ | $\text{Beta}(2, 3)$ |
    | $\eta_1$ | `eta1` | $10.0$ | $\text{Gamma}(5, 0.5)$ |
    | $\eta_2$ | `eta2` | $5.0$ | $\text{Gamma}(3, 0.5)$ |

=== "bates"

    | Parameter | Key | Default | Prior |
    |:----------|:----|:--------|:------|
    | $\mu$ | `mu` | $0.08$ | $\mathcal{N}(0.05, 0.1)$ |
    | $\kappa$ | `kappa` | $5.0$ | $\text{Gamma}(5, 1)$ |
    | $\theta$ | `theta` | $0.04$ | $\text{InvGamma}(5, 0.04)$ |
    | $\sigma_v$ | `sigma_v` | $0.5$ | $\text{InvGamma}(5, 0.5)$ |
    | $\rho$ | `rho` | $-0.7$ | $\text{Uniform}(-1, 0)$ |
    | $\lambda$ | `lambda_jump` | $3.0$ | $\text{Gamma}(2, 1)$ |
    | $\mu_J$ | `mu_jump` | $-0.02$ | $\mathcal{N}(0, 0.1)$ |
    | $\sigma_J$ | `sigma_jump` | $0.05$ | $\text{InvGamma}(5, 0.05)$ |

### Simulation

```python
jd = JumpDiffusion(variant="merton")
sim = jd.simulate(T=504, seed=42)  # 2 years of daily data

log_returns = sim["observations"]  # shape (504, 1)
log_prices = sim["states"]         # shape (504, 1)
prices = sim["prices"]             # shape (505,) - includes S_0
```

---

## Filtering

### Basic Filtering with Auxiliary PF

```python
import numpy as np
from particlefilterbox.models import JumpDiffusion
from particlefilterbox.filters import AuxiliaryPF
from particlefilterbox.core.config import PFConfig

# Model
jd = JumpDiffusion(
    variant="merton",
    params={
        "mu": 0.05, "sigma": 0.2,
        "lambda_jump": 5.0, "mu_jump": -0.03, "sigma_jump": 0.04,
    },
)

# Simulate
sim = jd.simulate(T=504, seed=42)
y = sim["observations"]

# Filter with Auxiliary PF (recommended for jump models)
config = PFConfig(n_particles=2000, seed=42)
apf = AuxiliaryPF(model=jd, config=config)
result = apf.filter(y)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Mean ESS: {np.mean(result.ess_history):.0f}")
```

### Why Not Bootstrap PF?

The Bootstrap PF struggles with jumps because:

1. At each time step, particles are propagated from the **prior** $p(x_t \mid x_{t-1})$
2. Jumps are rare ($\lambda \Delta t \ll 1$), so most particles will **not** propose a jump
3. When a jump actually occurs, the few particles that happened to jump will receive almost all the weight
4. This causes **severe weight degeneracy** and low ESS

The Auxiliary PF mitigates this by pre-weighting particles based on a first-stage approximation of $p(y_t \mid x_{t-1})$, effectively upweighting particles that are likely to match the observation --- including those near jump-compatible regions.

```python
# Comparison: Bootstrap vs Auxiliary on jump data
from particlefilterbox.filters import BootstrapPF, AuxiliaryPF

config = PFConfig(n_particles=2000, seed=42)

bpf = BootstrapPF(model=jd, config=config)
apf = AuxiliaryPF(model=jd, config=config)

result_bpf = bpf.filter(y)
result_apf = apf.filter(y)

print(f"Bootstrap - Mean ESS: {np.mean(result_bpf.ess_history):.0f}, "
      f"LogLik: {result_bpf.log_likelihood:.2f}")
print(f"Auxiliary - Mean ESS: {np.mean(result_apf.ess_history):.0f}, "
      f"LogLik: {result_apf.log_likelihood:.2f}")
```

---

## Jump Detection

One of the most valuable outputs of particle filtering for jump-diffusion models is the **posterior probability of a jump at each time step**.

### Merton: Jump Posterior

For the Merton model, the state includes the log-price. We can detect jumps by comparing the filtered return distribution to what diffusion alone would predict:

```python
import numpy as np
import matplotlib.pyplot as plt
from particlefilterbox.models import JumpDiffusion
from particlefilterbox.filters import AuxiliaryPF
from particlefilterbox.core.config import PFConfig

# Model with moderate jump activity
jd = JumpDiffusion(
    variant="merton",
    params={
        "mu": 0.05, "sigma": 0.15,
        "lambda_jump": 5.0, "mu_jump": -0.03, "sigma_jump": 0.06,
    },
)

# Simulate
sim = jd.simulate(T=504, seed=42)
y = sim["observations"]

# Filter
config = PFConfig(n_particles=3000, seed=42)
apf = AuxiliaryPF(model=jd, config=config)
result = apf.filter(y)

# Detect jumps: returns that are extreme relative to diffusion std
sigma = jd.params["sigma"]
dt = jd.dt
diffusion_std = sigma * np.sqrt(dt)

# Large moves (|return| > 3 * diffusion_std) are likely jumps
returns = y.squeeze()
threshold = 3.0 * diffusion_std
jump_mask = np.abs(returns) > threshold

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Returns with detected jumps
axes[0].plot(returns, color="steelblue", alpha=0.6, linewidth=0.5)
axes[0].scatter(
    np.where(jump_mask)[0], returns[jump_mask],
    color="red", s=20, zorder=5, label="Detected jumps"
)
axes[0].axhline(threshold, color="red", linestyle="--", alpha=0.3)
axes[0].axhline(-threshold, color="red", linestyle="--", alpha=0.3)
axes[0].set_ylabel("Log-return")
axes[0].legend()

# Simulated prices
axes[1].plot(sim["prices"], color="steelblue")
axes[1].set_ylabel("Price")

# ESS over time
axes[2].plot(result.ess_history, color="steelblue", alpha=0.7)
axes[2].axhline(config.n_particles * 0.5, color="red", linestyle="--", alpha=0.5)
axes[2].set_ylabel("ESS")
axes[2].set_xlabel("Time (days)")

plt.tight_layout()
plt.show()

print(f"Jumps detected: {jump_mask.sum()} / {len(returns)}")
print(f"Expected jumps: {jd.params['lambda_jump'] * dt * len(returns):.1f}")
```

### Bates: Volatility and Jumps

The Bates model provides both **jump detection** and **filtered stochastic volatility**:

```python
import numpy as np
from particlefilterbox.models import JumpDiffusion
from particlefilterbox.filters import AuxiliaryPF
from particlefilterbox.core.config import PFConfig

jd_bates = JumpDiffusion(
    variant="bates",
    params={
        "mu": 0.05, "kappa": 3.0, "theta": 0.04,
        "sigma_v": 0.4, "rho": -0.7,
        "lambda_jump": 3.0, "mu_jump": -0.02, "sigma_jump": 0.05,
    },
)

sim = jd_bates.simulate(T=504, seed=42)

config = PFConfig(n_particles=3000, seed=42)
apf = AuxiliaryPF(model=jd_bates, config=config)
result = apf.filter(sim["observations"])

# Extract filtered variance
states = result.filtered_states  # shape (T, N, 2): [log_S, v]
v_filtered = states[:, :, 1].mean(axis=1)
vol_filtered = np.sqrt(v_filtered) * np.sqrt(252)  # annualized

print(f"Mean filtered vol: {vol_filtered.mean() * 100:.1f}%")
print(f"True long-run vol: {np.sqrt(0.04) * np.sqrt(252) * 100:.1f}%")
```

---

## Parameter Estimation with PMMH

Bayesian estimation of jump-diffusion parameters:

```python
import numpy as np
from particlefilterbox.models import JumpDiffusion
from particlefilterbox.filters import AuxiliaryPF
from particlefilterbox.pmcmc import PMMH
from particlefilterbox.core.config import PFConfig

# True model
jd_true = JumpDiffusion(
    variant="merton",
    params={
        "mu": 0.05, "sigma": 0.2,
        "lambda_jump": 5.0, "mu_jump": -0.03, "sigma_jump": 0.05,
    },
)
sim = jd_true.simulate(T=1008, seed=42)  # 4 years
y = sim["observations"]

# Estimation
jd_est = JumpDiffusion(variant="merton")
pf_config = PFConfig(n_particles=1000, seed=0)
priors = jd_est.default_prior()

pmmh = PMMH(
    model=jd_est,
    filter_cls=AuxiliaryPF,
    pf_config=pf_config,
    priors=priors,
    n_iterations=10000,
    burn_in=3000,
    seed=42,
)
chain = pmmh.run(y)

# Results
print("Posterior estimates (true values in parentheses):")
true_vals = {"mu": 0.05, "sigma": 0.2, "lambda_jump": 5.0, "mu_jump": -0.03, "sigma_jump": 0.05}
for param, true_val in true_vals.items():
    samples = chain[param]
    print(f"  {param}: {samples.mean():.4f} +/- {samples.std():.4f}  (true: {true_val})")
```

---

## Example: Commodity Price Modeling

Commodities exhibit jump behavior from supply disruptions, weather events, and geopolitical shocks:

```python
import numpy as np
from particlefilterbox.models import JumpDiffusion
from particlefilterbox.filters import AuxiliaryPF
from particlefilterbox.core.config import PFConfig

# Model calibrated for oil-like dynamics
# Higher jump intensity, larger negative jumps (supply shocks)
oil_model = JumpDiffusion(
    variant="merton",
    params={
        "mu": 0.03,              # lower drift for commodities
        "sigma": 0.30,           # higher base volatility
        "lambda_jump": 8.0,      # ~8 jumps per year
        "mu_jump": -0.01,        # slightly negative average jump
        "sigma_jump": 0.08,      # large jump variability
    },
    dt=1/252,
)

# Simulate 5 years of daily data
sim = oil_model.simulate(T=1260, seed=42)

# Filter with many particles (jumps need more particles)
config = PFConfig(n_particles=5000, seed=42)
apf = AuxiliaryPF(model=oil_model, config=config)
result = apf.filter(sim["observations"])

# Identify extreme moves
returns = sim["observations"].squeeze()
threshold = 3.0 * oil_model.params["sigma"] * np.sqrt(oil_model.dt)
large_moves = np.abs(returns) > threshold

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Large moves (|r| > 3 sigma_diff): {large_moves.sum()}")
print(f"Largest single-day move: {returns[np.argmax(np.abs(returns))]:.4f}")
print(f"Price range: [{sim['prices'].min():.2f}, {sim['prices'].max():.2f}]")
```

---

## Variant Comparison

| Feature | Merton | Kou | Bates |
|:--------|:-------|:----|:------|
| **Jump distribution** | Gaussian | Double-exponential | Gaussian |
| **Volatility** | Constant | Constant | Stochastic (Heston) |
| **State dimension** | 1 | 1 | 2 |
| **Asymmetric jumps** | Via $\mu_J < 0$ | Intrinsic ($p, \eta_1, \eta_2$) | Via $\mu_J < 0$ |
| **Leverage effect** | No | No | Yes ($\rho < 0$) |
| **Parameters** | 5 | 6 | 8 |
| **Best for** | Simple jump modeling | Asymmetric tails | Full dynamics |
| **Particles needed** | 1000--2000 | 1000--2000 | 2000--5000 |

!!! tip "Choosing a Variant"
    - **Merton**: Start here. Simple, well-understood, sufficient for many applications.
    - **Kou**: When you need to separately control upward vs downward jump magnitudes.
    - **Bates**: When both jumps *and* stochastic volatility are needed. Most realistic but most expensive.

---

## See Also

- [Auxiliary PF](../filters/auxiliary.md) --- Recommended filter for jump models
- [Bootstrap PF](../filters/bootstrap.md) --- Simpler alternative (less efficient for jumps)
- [PMMH](../pmcmc/pmmh.md) --- Bayesian parameter estimation
- [Stochastic Volatility](stochastic-volatility.md) --- SV model (no jumps, simpler)
- [ContinuousTime](continuous-time.md) --- Related continuous-time SDEs (CIR, Vasicek, Heston)
