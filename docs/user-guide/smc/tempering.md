---
title: SMC Tempering
description: "SMC Tempering / Annealing — adaptive temperature schedules for multimodal posteriors and marginal likelihood estimation"
---

# SMC Tempering

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `SMCTempering` |
    | **Import** | `from particlefilterbox.smc import SMCTempering` |
    | **Target** | Posterior $p(\theta \mid y) \propto p(\theta) \cdot p(y \mid \theta)$ |
    | **Complexity** | $O(N \cdot T \cdot C_{\text{MCMC}})$ |
    | **Key output** | Marginal likelihood $\hat{p}(y)$ |

## Overview

SMC Tempering constructs a sequence of **tempered distributions** that smoothly interpolate between the prior and the posterior:

$$
\pi_t(\theta) \propto p(\theta) \cdot p(y \mid \theta)^{\beta_t}, \qquad 0 = \beta_0 < \beta_1 < \cdots < \beta_T = 1
$$

At $\beta = 0$, the distribution is the prior $p(\theta)$. At $\beta = 1$, it is the full posterior $p(\theta \mid y)$. By gradually increasing $\beta$, the algorithm "anneals" particles from the prior into the posterior, avoiding the mode-trapping problems that plague MCMC.

**Advantages:**

- Robust exploration of multimodal posteriors
- Provides an unbiased estimate of the marginal likelihood $p(y)$ as a by-product
- Adaptive $\beta$ selection makes tuning easy
- Conceptually simple and easy to implement

**Disadvantages:**

- Offline method --- requires all data upfront
- Computational cost depends on the distance between prior and posterior
- MCMC kernel quality affects final sample quality

---

## Algorithm

$$
\boxed{
\begin{aligned}
&\textbf{SMC Tempering} \\[6pt]
&\textbf{Input: } \text{Prior } p(\theta), \text{ likelihood } p(y \mid \theta), \text{ target ESS ratio } \alpha \\[4pt]
&\text{1. } \textbf{Initialize: } \text{For } i = 1, \ldots, N: \\
&\qquad \theta_0^{(i)} \sim p(\theta), \quad w_0^{(i)} = \tfrac{1}{N}, \quad \beta_0 = 0 \\[4pt]
&\text{2. } \textbf{While } \beta_t < 1: \\
&\qquad \text{a. } \textbf{Select next temperature: } \\
&\qquad \qquad \beta_{t+1} = \arg\min_{\beta > \beta_t} \left| \text{ESS}(\beta) - \alpha \cdot N \right| \\
&\qquad \qquad \text{where } \text{ESS}(\beta) = \frac{\left(\sum_i w_t^{(i)} \cdot p(y \mid \theta_t^{(i)})^{\beta - \beta_t}\right)^2}{\sum_i \left(w_t^{(i)} \cdot p(y \mid \theta_t^{(i)})^{\beta - \beta_t}\right)^2} \\[4pt]
&\qquad \text{b. } \textbf{Reweight: } \tilde{w}_{t+1}^{(i)} = w_t^{(i)} \cdot p(y \mid \theta_t^{(i)})^{\beta_{t+1} - \beta_t} \\[4pt]
&\qquad \text{c. } \textbf{Normalize: } w_{t+1}^{(i)} = \frac{\tilde{w}_{t+1}^{(i)}}{\sum_j \tilde{w}_{t+1}^{(j)}} \\[4pt]
&\qquad \text{d. } \textbf{Resample: } \text{If } \widehat{\text{ESS}}_{t+1} < \tau \cdot N \\[4pt]
&\qquad \text{e. } \textbf{Move: } \theta_{t+1}^{(i)} \sim K_{t+1}(\cdot \mid \theta_t^{(i)}) \text{ targeting } \pi_{t+1} \\[4pt]
&\text{3. } \textbf{Output: } \{(\theta_T^{(i)}, w_T^{(i)})\}, \quad \log \hat{p}(y) = \sum_{t=0}^{T-1} \log\left(\sum_i \tilde{w}_{t+1}^{(i)}\right)
\end{aligned}
}
$$

### Adaptive Temperature Selection

The adaptive schedule finds $\beta_{t+1}$ via **bisection** such that the ESS after reweighting equals a target fraction $\alpha$ of $N$. This ensures:

- Each tempering step makes a "manageable" change to the distribution
- The algorithm automatically uses more steps where the likelihood surface is complex
- No manual schedule tuning is required

$$
\beta_{t+1} = \underset{\beta \in (\beta_t, 1]}{\text{solve}} \quad \text{ESS}_\beta = \alpha \cdot N
$$

If $\text{ESS}(\beta = 1) > \alpha \cdot N$, the algorithm jumps directly to $\beta_{t+1} = 1$.

!!! tip "Choosing the ESS target $\\alpha$"
    A target of $\alpha = 0.5$ (default) works well for most problems. Lower values (e.g., 0.3) produce fewer, larger steps. Higher values (e.g., 0.8) produce many small steps with more robust exploration.

---

## Marginal Likelihood Estimation

One of the most valuable features of SMC Tempering is the **unbiased estimate of the marginal likelihood**:

$$
\hat{p}(y) = \prod_{t=0}^{T-1} \left( \frac{1}{N} \sum_{i=1}^{N} \tilde{w}_{t+1}^{(i)} \right)
$$

or equivalently in log space:

$$
\log \hat{p}(y) = \sum_{t=0}^{T-1} \log \left( \frac{1}{N} \sum_{i=1}^{N} p(y \mid \theta_t^{(i)})^{\beta_{t+1} - \beta_t} \right)
$$

This estimate is:

- **Unbiased** (in the non-log domain) --- consistent for model comparison
- **Low variance** --- because each incremental weight ratio is controlled by the adaptive schedule
- **Available at no extra cost** --- computed as part of the algorithm

### Bayes Factors via Tempering

To compare models $\mathcal{M}_1$ and $\mathcal{M}_2$:

$$
\text{BF}_{12} = \frac{\hat{p}(y \mid \mathcal{M}_1)}{\hat{p}(y \mid \mathcal{M}_2)}
$$

---

## Connection to Other Methods

### Simulated Annealing

SMC Tempering is related to simulated annealing (SA) but is fundamentally different:

| Aspect | Simulated Annealing | SMC Tempering |
|--------|:------------------:|:-------------:|
| Goal | Optimization (find mode) | Sampling (approximate posterior) |
| Particles | Single point | $N$ weighted particles |
| Output | Point estimate | Full posterior approximation |
| Marginal likelihood | No | **Yes** |

### Parallel Tempering (Replica Exchange MCMC)

| Aspect | Parallel Tempering | SMC Tempering |
|--------|:-----------------:|:-------------:|
| Temperatures | Fixed, run in parallel | Sequential, adaptive |
| Communication | Swap proposals between chains | Reweighting + resampling |
| Output | Samples from target chain | Weighted particles + $\hat{Z}$ |
| Parallelism | Across temperatures | Across particles |

---

## API Reference

### Constructor

```python
from particlefilterbox.smc import SMCTempering, SMCTemperingConfig

config = SMCTemperingConfig(
    n_particles=2000,
    resampling="systematic",
    ess_target=0.5,          # target ESS fraction for adaptive schedule
    n_mcmc_steps=5,
    kernel="random_walk",
    seed=42,
)

tempering = SMCTempering(
    log_likelihood=log_lik,      # callable: theta -> log p(y | theta)
    log_prior=log_prior,         # callable: theta -> log p(theta)
    prior_sample=sample_prior,   # callable: (n, rng) -> array (n, dim)
    config=config,
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `2000` | Number of particles $N$ |
| `ess_target` | `float` | `0.5` | Target ESS fraction $\alpha$ for adaptive $\beta$ |
| `n_mcmc_steps` | `int` | `5` | MCMC moves per tempering step |
| `kernel` | `str` | `"random_walk"` | MCMC kernel type |
| `waste_free` | `bool` | `False` | Enable [Waste-Free](waste-free.md) mode |
| `max_steps` | `int` | `200` | Maximum number of tempering steps |

### Running

```python
result = tempering.sample()
```

### Result Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `particles` | `(N, dim)` | Posterior particles |
| `weights` | `(N,)` | Normalized weights |
| `log_marginal_likelihood` | scalar | $\log \hat{p}(y)$ |
| `betas` | `(T+1,)` | Adaptive temperature schedule $\beta_0, \ldots, \beta_T$ |
| `ess_history` | `(T,)` | ESS at each tempering step |
| `acceptance_rates` | `(T,)` | MCMC acceptance rates |
| `n_steps` | `int` | Total number of tempering steps $T$ |

---

## Examples

### Example 1: Multimodal Posterior with Marginal Likelihood

```python
import numpy as np
from particlefilterbox.smc import SMCTempering, SMCTemperingConfig

# --- Bimodal target posterior ---
def log_likelihood(theta):
    """Likelihood with two modes."""
    x, y = theta[0], theta[1]
    mode1 = -0.5 * ((x - 3)**2 + (y - 3)**2) / 0.5**2
    mode2 = -0.5 * ((x + 3)**2 + (y + 3)**2) / 0.5**2
    return np.logaddexp(mode1, mode2)

def log_prior(theta):
    """Wide Gaussian prior."""
    return -0.5 * np.sum(theta**2) / 25.0

def sample_prior(n, rng):
    return rng.normal(0, 5, size=(n, 2))

# --- Run SMC Tempering ---
config = SMCTemperingConfig(
    n_particles=3000,
    ess_target=0.5,
    n_mcmc_steps=5,
    kernel="random_walk",
    seed=42,
)

tempering = SMCTempering(
    log_likelihood=log_likelihood,
    log_prior=log_prior,
    prior_sample=sample_prior,
    config=config,
)

result = tempering.sample()

# --- Results ---
print(f"Log marginal likelihood: {result.log_marginal_likelihood:.3f}")
print(f"Number of tempering steps: {result.n_steps}")
print(f"Temperature schedule: {result.betas[:5]}... -> {result.betas[-3:]}")
print(f"Mean acceptance rate: {result.acceptance_rates.mean():.3f}")

# Verify both modes are captured
particles = result.particles
in_mode1 = np.sum((particles[:, 0] > 0) & (particles[:, 1] > 0))
in_mode2 = np.sum((particles[:, 0] < 0) & (particles[:, 1] < 0))
print(f"Mode 1: {in_mode1} particles, Mode 2: {in_mode2} particles")
```

### Example 2: Bayesian Model Comparison

Using tempering to estimate marginal likelihoods for competing models:

```python
import numpy as np
from particlefilterbox.smc import SMCTempering, SMCTemperingConfig

# --- Simulate data from a quadratic model ---
rng = np.random.default_rng(42)
n_obs = 50
x = np.linspace(-2, 2, n_obs)
y_true = 1.0 + 0.5 * x + 0.3 * x**2
y = y_true + rng.normal(0, 0.3, size=n_obs)

# --- Model 1: Linear ---
def log_lik_linear(theta):
    a, b, sigma = theta[0], theta[1], np.exp(theta[2])
    pred = a + b * x
    return np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y - pred) / sigma)**2)

def log_prior_linear(theta):
    return -0.5 * np.sum(theta[:2]**2) / 10.0 - 0.5 * theta[2]**2 / 4.0

def sample_prior_linear(n, rng):
    return np.column_stack([
        rng.normal(0, np.sqrt(10), size=(n, 2)),
        rng.normal(0, 2, size=n),
    ])

# --- Model 2: Quadratic ---
def log_lik_quad(theta):
    a, b, c, sigma = theta[0], theta[1], theta[2], np.exp(theta[3])
    pred = a + b * x + c * x**2
    return np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y - pred) / sigma)**2)

def log_prior_quad(theta):
    return -0.5 * np.sum(theta[:3]**2) / 10.0 - 0.5 * theta[3]**2 / 4.0

def sample_prior_quad(n, rng):
    return np.column_stack([
        rng.normal(0, np.sqrt(10), size=(n, 3)),
        rng.normal(0, 2, size=n),
    ])

# --- Run tempering for both models ---
config = SMCTemperingConfig(n_particles=3000, seed=42)

temp_linear = SMCTempering(
    log_likelihood=log_lik_linear,
    log_prior=log_prior_linear,
    prior_sample=sample_prior_linear,
    config=config,
)

temp_quad = SMCTempering(
    log_likelihood=log_lik_quad,
    log_prior=log_prior_quad,
    prior_sample=sample_prior_quad,
    config=config,
)

result_linear = temp_linear.sample()
result_quad = temp_quad.sample()

# --- Compare ---
log_bf = result_quad.log_marginal_likelihood - result_linear.log_marginal_likelihood
print(f"Log ML (linear):    {result_linear.log_marginal_likelihood:.2f}")
print(f"Log ML (quadratic): {result_quad.log_marginal_likelihood:.2f}")
print(f"Log Bayes Factor (quad vs linear): {log_bf:.2f}")
```

### Example 3: Stochastic Volatility Parameter Estimation

```python
import numpy as np
from particlefilterbox.smc import SMCTempering, SMCTemperingConfig
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.models import StochasticVolatility

# --- Simulate data ---
rng = np.random.default_rng(42)
sv_true = StochasticVolatility(phi=0.98, sigma=0.16, beta=0.65)
x_true, y_obs = sv_true.simulate(T=300, rng=rng)

# --- Likelihood via particle filter ---
pf_config = PFConfig(n_particles=500, resampling="systematic")

def log_likelihood(theta):
    phi, sigma, beta = theta[0], np.exp(theta[1]), np.exp(theta[2])
    if abs(phi) >= 1.0:
        return -np.inf
    model = StochasticVolatility(phi=phi, sigma=sigma, beta=beta)
    pf = BootstrapPF(model=model, config=pf_config)
    result = pf.filter(y_obs)
    return result.log_likelihood

def log_prior(theta):
    phi = theta[0]
    log_sigma, log_beta = theta[1], theta[2]
    lp = 0.0
    lp += -0.5 * (phi - 0.95)**2 / 0.05**2  # prior on phi
    lp += -0.5 * log_sigma**2 / 1.0**2        # prior on log(sigma)
    lp += -0.5 * log_beta**2 / 1.0**2          # prior on log(beta)
    return lp

def sample_prior(n, rng):
    phi = rng.normal(0.95, 0.05, size=n).clip(-0.999, 0.999)
    log_sigma = rng.normal(0, 1, size=n)
    log_beta = rng.normal(0, 1, size=n)
    return np.column_stack([phi, log_sigma, log_beta])

# --- Run ---
config = SMCTemperingConfig(n_particles=1000, n_mcmc_steps=3, seed=42)

tempering = SMCTempering(
    log_likelihood=log_likelihood,
    log_prior=log_prior,
    prior_sample=sample_prior,
    config=config,
)

result = tempering.sample()

# --- Posterior ---
w = result.weights
theta = result.particles
phi_est = np.average(theta[:, 0], weights=w)
sigma_est = np.average(np.exp(theta[:, 1]), weights=w)
beta_est = np.average(np.exp(theta[:, 2]), weights=w)

print(f"phi:   true=0.980, estimate={phi_est:.3f}")
print(f"sigma: true=0.160, estimate={sigma_est:.3f}")
print(f"beta:  true=0.650, estimate={beta_est:.3f}")
print(f"Log marginal likelihood: {result.log_marginal_likelihood:.2f}")
```

!!! warning "Particle filter likelihood"
    When the likelihood is estimated via a particle filter (as in Example 3), the SMC tempering algorithm uses a **noisy** likelihood estimate. This is valid but increases variance. Use enough inner particles ($N_x \geq 500$) for a reasonably stable likelihood estimate.

---

## Tuning Guide

### ESS Target ($\alpha$)

| $\alpha$ | Behavior | Steps $T$ |
|:---------:|----------|:---------:|
| 0.3 | Aggressive: large $\Delta\beta$, fewer steps | Low |
| 0.5 | **Default**: balanced | Medium |
| 0.8 | Conservative: small $\Delta\beta$, many steps | High |
| 0.95 | Very conservative: near-continuous annealing | Very high |

### Temperature Schedule Diagnostics

The adaptive schedule provides useful diagnostics:

- **Many small steps near $\beta = 0$**: the prior and likelihood are very different in some region
- **Large jumps near $\beta = 1$**: the posterior is well-approximated by a tempered version at moderate $\beta$
- **Total steps $T$**: a rough measure of the "distance" between prior and posterior

### Computational Complexity

| Operation | Cost |
|-----------|------|
| Adaptive $\beta$ selection | $O(N)$ per step (bisection with weight computation) |
| Reweighting | $O(N)$ per step |
| MCMC moves | $O(N \cdot M \cdot C_K)$ per step |
| **Total** | $O(N \cdot T \cdot M \cdot C_K)$ |

The number of steps $T$ is data-dependent (determined by the adaptive schedule).

---

## See Also

- [SMC Sampler](smc-sampler.md) --- tempering is a special case of the general SMC sampler framework
- [Waste-Free SMC](waste-free.md) --- combine with `waste_free=True` for improved estimates
- [IBIS](ibis.md) --- an online alternative that processes data sequentially rather than tempering

---

## References

- Neal, R.M. (2001). Annealed Importance Sampling. *Statistics and Computing*, 11(2), 125--139.
- Del Moral, P., Doucet, A. & Jasra, A. (2006). Sequential Monte Carlo Samplers. *JRSS-B*, 68(3), 411--436.
- Jasra, A., Stephens, D.A., Doucet, A. & Tsagaris, T. (2011). Inference for Levy-Driven Stochastic Volatility Models via Adaptive Sequential Monte Carlo. *Scandinavian Journal of Statistics*, 38(1), 1--22.
- Chopin, N. & Papaspiliopoulos, O. (2020). *An Introduction to Sequential Monte Carlo*. Springer, Chapter 17.
