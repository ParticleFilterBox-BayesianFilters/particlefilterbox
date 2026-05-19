---
title: IBIS
description: "IBIS — Iterated Batch Importance Sampling (Chopin, 2002) for online Bayesian parameter estimation"
---

# IBIS — Iterated Batch Importance Sampling

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `IBIS` |
    | **Import** | `from particlefilterbox.smc import IBIS` |
    | **Target** | Parameter posterior $p(\theta \mid y_{1:t})$ updated sequentially |
    | **Complexity** | $O(N \cdot T \cdot C_{\text{lik}})$ |
    | **Reference** | Chopin (2002) |

## Overview

IBIS (**Iterated Batch Importance Sampling**) is an online SMC method for **Bayesian parameter estimation**. It maintains a weighted particle cloud over the parameter space $\theta$ and updates it as new observations arrive. The key idea is that each new observation $y_t$ changes the target from $p(\theta \mid y_{1:t-1})$ to $p(\theta \mid y_{1:t})$, and the incremental weight is simply the **likelihood of the new observation**:

$$
w_t^{(i)} \propto w_{t-1}^{(i)} \cdot p(y_t \mid y_{1:t-1}, \theta^{(i)})
$$

When weights become too uneven (ESS drops), particles are **resampled** and **rejuvenated** via MCMC moves targeting the current posterior.

**Advantages:**

- Simpler and cheaper than SMC$^2$ when states can be integrated out
- Fully online --- processes observations one at a time
- Provides marginal likelihood as a by-product
- Natural for models with tractable likelihoods

**Disadvantages:**

- Requires the ability to evaluate $p(y_t \mid y_{1:t-1}, \theta)$ --- the **incremental likelihood**
- Not applicable when latent states cannot be analytically marginalized (use [SMC$^2$](smc-squared.md) instead)
- MCMC rejuvenation can be expensive for high-dimensional $\theta$

---

## Algorithm

$$
\boxed{
\begin{aligned}
&\textbf{IBIS} \\[6pt]
&\textbf{Input: } \text{Prior } p(\theta), \text{ incremental likelihood } p(y_t \mid y_{1:t-1}, \theta) \\[4pt]
&\text{1. } \textbf{Initialize: } \text{For } i = 1, \ldots, N: \\
&\qquad \theta^{(i)} \sim p(\theta), \quad w_0^{(i)} = \tfrac{1}{N} \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T \text{ (each new observation } y_t\text{):} \\
&\qquad \text{a. } \textbf{Compute incremental weights:} \\
&\qquad \qquad \alpha_t^{(i)} = p(y_t \mid y_{1:t-1}, \theta^{(i)}) \\[4pt]
&\qquad \text{b. } \textbf{Update weights: } \tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot \alpha_t^{(i)} \\[4pt]
&\qquad \text{c. } \textbf{Normalize: } w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_{j=1}^{N} \tilde{w}_t^{(j)}} \\[4pt]
&\qquad \text{d. } \textbf{Update sufficient statistics: } S_t^{(i)} = \text{update}(S_{t-1}^{(i)}, y_t) \\[4pt]
&\qquad \text{e. } \textbf{If } \widehat{\text{ESS}}_t < \tau \cdot N: \\
&\qquad \qquad \text{(i) Resample: draw indices } \{a^{(i)}\} \text{ from } \{w_t^{(i)}\} \\
&\qquad \qquad \text{(ii) Rejuvenate: for each } i, \text{ apply } M \text{ MCMC steps:} \\
&\qquad \qquad \qquad \theta^{(i)} \sim K_t(\cdot \mid \theta^{(a^{(i)})}) \text{ targeting } p(\theta \mid y_{1:t}) \\
&\qquad \qquad \text{(iii) Reset weights: } w_t^{(i)} = \tfrac{1}{N} \\[4pt]
&\text{3. } \textbf{Output: } \{(\theta^{(i)}, w_T^{(i)})\}, \quad \log \hat{p}(y_{1:T}) = \sum_{t=1}^{T} \log\left(\sum_{i=1}^{N} \tilde{w}_t^{(i)}\right)
\end{aligned}
}
$$

### Sufficient Statistics

A key efficiency feature of IBIS is maintaining **sufficient statistics** $S_t^{(i)}$ for each particle. For models in the exponential family, these statistics allow computing the incremental likelihood $p(y_t \mid y_{1:t-1}, \theta^{(i)})$ in $O(1)$ rather than reprocessing all past data.

For example, in a linear Gaussian model with known $\theta$:

$$
S_t = (m_t, P_t) \quad \text{(Kalman filter mean and covariance)}
$$

The predictive likelihood is then:

$$
p(y_t \mid y_{1:t-1}, \theta) = \mathcal{N}(y_t; H m_{t|t-1}, H P_{t|t-1} H^\top + R)
$$

---

## API Reference

### Constructor

```python
from particlefilterbox.smc import IBIS, IBISConfig

config = IBISConfig(
    n_particles=1000,
    resampling="systematic",
    ess_threshold=0.5,
    n_mcmc_moves=5,
    seed=42,
)

ibis = IBIS(model=my_model, config=config)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `1000` | Number of parameter particles $N$ |
| `resampling` | `str` | `"systematic"` | Resampling scheme |
| `ess_threshold` | `float` | `0.5` | Trigger rejuvenation when $\text{ESS} < \tau N$ |
| `n_mcmc_moves` | `int` | `5` | MCMC steps per rejuvenation |
| `kernel` | `str` | `"random_walk"` | MCMC kernel type |
| `adapt_kernel` | `bool` | `True` | Adapt kernel covariance from particle population |

### Running

=== "Batch mode"

    ```python
    result = ibis.filter(observations)
    ```

=== "Online mode"

    ```python
    cloud = ibis.initialize(rng)

    for t, y_t in enumerate(observations):
        cloud = ibis.step(cloud, y_t, t)
        print(f"t={t}: ESS={cloud.ess:.0f}, log-lik={cloud.log_likelihood:.3f}")
    ```

### Result Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `particles` | `(N, dim_\theta)` | Parameter particles |
| `weights` | `(N,)` | Normalized weights |
| `log_marginal_likelihood` | scalar | $\log \hat{p}(y_{1:T})$ |
| `ess_history` | `(T,)` | ESS at each observation |
| `rejuvenation_times` | list | Times when rejuvenation was triggered |
| `acceptance_rates` | list | MCMC acceptance rates at each rejuvenation |

---

## Examples

### Example 1: Online Estimation of AR(1) Parameters

A simple model where the likelihood is available in closed form:

```python
import numpy as np
from particlefilterbox.smc import IBIS, IBISConfig

# --- Simulate AR(1) data ---
# y_t = phi * y_{t-1} + sigma * eps_t
rng = np.random.default_rng(42)
T = 500
phi_true, sigma_true = 0.8, 0.5

y = np.zeros(T)
y[0] = rng.normal(0, sigma_true / np.sqrt(1 - phi_true**2))
for t in range(1, T):
    y[t] = phi_true * y[t - 1] + sigma_true * rng.normal()

# --- Define model ---
class AR1Model:
    param_names = ["phi", "sigma"]
    param_dim = 2

    def prior_sample(self, n, rng):
        phi = rng.uniform(-0.99, 0.99, size=n)
        sigma = rng.exponential(1.0, size=n)
        return np.column_stack([phi, sigma])

    def log_prior(self, theta):
        phi, sigma = theta[0], theta[1]
        if abs(phi) >= 0.99 or sigma <= 0:
            return -np.inf
        return -sigma  # Exponential(1) prior on sigma

    def log_incremental_likelihood(self, theta, y_t, y_prev, t):
        phi, sigma = theta[0], theta[1]
        if t == 0:
            # Stationary distribution
            var_0 = sigma**2 / (1 - phi**2)
            return -0.5 * np.log(2 * np.pi * var_0) - 0.5 * y_t**2 / var_0
        else:
            residual = y_t - phi * y_prev
            return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * residual**2 / sigma**2

model = AR1Model()

# --- Run IBIS ---
config = IBISConfig(n_particles=1000, ess_threshold=0.5, seed=42)
ibis = IBIS(model=model, config=config)
result = ibis.filter(y)

# --- Results ---
w = result.weights
theta = result.particles

phi_mean = np.average(theta[:, 0], weights=w)
sigma_mean = np.average(theta[:, 1], weights=w)
print(f"phi:   true={phi_true:.3f}, estimate={phi_mean:.3f}")
print(f"sigma: true={sigma_true:.3f}, estimate={sigma_mean:.3f}")
print(f"Log marginal likelihood: {result.log_marginal_likelihood:.2f}")
print(f"Rejuvenation steps: {len(result.rejuvenation_times)}")
```

### Example 2: Bayesian Model Comparison

Using IBIS to compare two competing models via marginal likelihoods:

```python
import numpy as np
from particlefilterbox.smc import IBIS, IBISConfig

# --- Two competing models for the same data ---
# Model 1: AR(1) with Gaussian innovations
# Model 2: AR(1) with Student-t innovations

config = IBISConfig(n_particles=2000, seed=42)

ibis_gaussian = IBIS(model=AR1Gaussian(), config=config)
ibis_student = IBIS(model=AR1StudentT(), config=config)

result_g = ibis_gaussian.filter(y)
result_t = ibis_student.filter(y)

# --- Bayes factor ---
log_bf = result_g.log_marginal_likelihood - result_t.log_marginal_likelihood
print(f"Log Bayes Factor (Gaussian vs Student-t): {log_bf:.2f}")

if log_bf > 0:
    print("Evidence favors Gaussian innovations")
else:
    print("Evidence favors Student-t innovations")
```

!!! tip "Interpreting Bayes factors"
    | $|\log \text{BF}|$ | Evidence |
    |:------------------:|----------|
    | 0--1 | Negligible |
    | 1--3 | Positive |
    | 3--5 | Strong |
    | > 5 | Decisive |

---

## IBIS vs. SMC$^2$

The choice between IBIS and SMC$^2$ depends on whether latent states can be analytically marginalized:

| Scenario | Use IBIS | Use SMC$^2$ |
|----------|:--------:|:-----------:|
| No latent states (e.g., regression) | **Yes** | Overkill |
| Linear Gaussian states | **Yes** (via Kalman filter) | Possible but wasteful |
| Nonlinear states | Only with approximations | **Yes** |
| Very high-dimensional states | Not applicable | **Yes** |

### When IBIS Can Handle Latent States

IBIS can still be used with latent states if the **predictive likelihood** $p(y_t \mid y_{1:t-1}, \theta)$ can be computed:

- **Linear Gaussian dynamics** --- use the Kalman filter as a sufficient statistic
- **Finite-state HMM** --- use the forward algorithm
- **Conjugate models** --- use analytic updates

In these cases, IBIS is dramatically cheaper than SMC$^2$ because it avoids running $N_\theta$ inner particle filters.

!!! note "Requires kalmanbox"
    For linear Gaussian state-space models, particlefilterbox uses [kalmanbox](https://github.com/guhaase/kalmanbox) as the inner Kalman filter. Install with `pip install kalmanbox`.

---

## Tuning Guide

### Number of Particles

| Scenario | Recommended $N$ |
|----------|:--------------:|
| Quick exploration | 500 |
| Standard estimation (2--5 parameters) | 1,000--2,000 |
| High-dimensional $\theta$ (> 10) | 5,000+ |
| Model comparison (need accurate $\hat{Z}$) | 2,000--5,000 |

### Rejuvenation Frequency

The ESS threshold controls how often rejuvenation is triggered:

- **Higher threshold** (0.7--0.8): more frequent rejuvenation, better particle diversity, higher cost
- **Lower threshold** (0.3--0.5): less frequent rejuvenation, cheaper, but risk of weight collapse
- **Default** (0.5): good balance for most problems

### Computational Complexity

| Operation | Cost |
|-----------|------|
| Incremental weights | $O(N \cdot C_{\text{lik}})$ per observation |
| Resampling | $O(N)$ when triggered |
| MCMC rejuvenation | $O(N \cdot M \cdot C_{\text{post}})$ when triggered |
| **Total** | $O(N \cdot T \cdot C_{\text{lik}} + R \cdot N \cdot M \cdot C_{\text{post}})$ |

Where $R$ = number of rejuvenation events, $M$ = MCMC steps per event, $C_{\text{post}}$ = cost of evaluating $\log p(\theta \mid y_{1:t})$.

---

## See Also

- [SMC$^2$](smc-squared.md) --- use this when latent states cannot be marginalized
- [SMC Tempering](tempering.md) --- offline alternative for static parameter estimation
- [Waste-Free SMC](waste-free.md) --- can reduce waste during the rejuvenation step

---

## References

- Chopin, N. (2002). A Sequential Particle Filter Method for Static Models. *Biometrika*, 89(3), 539--552.
- Chopin, N. (2004). Central Limit Theorem for Sequential Monte Carlo Methods and its Application to Bayesian Inference. *Annals of Statistics*, 32(6), 2385--2411.
- Chopin, N. & Papaspiliopoulos, O. (2020). *An Introduction to Sequential Monte Carlo*. Springer, Chapter 11.
