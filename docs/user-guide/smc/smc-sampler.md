---
title: SMC Sampler
description: "SMC Sampler (Del Moral, Doucet & Jasra, 2006) — general-purpose sampling via a sequence of bridging distributions"
---

# SMC Sampler

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `SMCSampler` |
    | **Import** | `from particlefilterbox.smc import SMCSampler` |
    | **Target** | Any distribution $\pi(\theta)$ accessible up to a normalizing constant |
    | **Complexity** | $O(N \cdot T \cdot C_{\text{MCMC}})$ per run |
    | **Reference** | Del Moral, Doucet & Jasra (2006) |

## Overview

The SMC Sampler is a **general-purpose algorithm** for sampling from complex target distributions. Instead of attempting to sample directly from the target $\pi^*$, it constructs a sequence of **bridging distributions** that gradually transform a simple initial distribution into the target:

$$
\pi_0 \longrightarrow \pi_1 \longrightarrow \cdots \longrightarrow \pi_T = \pi^*
$$

At each step, particles are **reweighted** according to the ratio of consecutive distributions, **resampled** to eliminate low-weight particles, and **moved** via an MCMC kernel that preserves the current distribution.

**Advantages:**

- Works for *any* target distribution --- no state-space structure required
- Provides a normalizing constant estimate $\hat{Z}$ as a by-product
- Can handle multimodal targets by exploring modes early in the sequence
- Embarrassingly parallelizable across particles

**Disadvantages:**

- Requires choosing a bridging schedule and MCMC kernel
- Computational cost scales with the number of bridging steps
- May struggle in very high dimensions without good MCMC moves

---

## Algorithm

$$
\boxed{
\begin{aligned}
&\textbf{SMC Sampler} \\[6pt]
&\textbf{Input: } \text{Sequence } \pi_0, \ldots, \pi_T = \pi^*, \text{ MCMC kernels } K_1, \ldots, K_T \\[4pt]
&\text{1. } \textbf{Initialize: } \text{For } i = 1, \ldots, N: \\
&\qquad \theta_0^{(i)} \sim \pi_0(\theta), \qquad w_0^{(i)} = \tfrac{1}{N} \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T: \\
&\qquad \text{a. } \textbf{Reweight: } \tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot \frac{\gamma_t(\theta_{t-1}^{(i)})}{\gamma_{t-1}(\theta_{t-1}^{(i)})} \\[4pt]
&\qquad \text{b. } \textbf{Normalize: } w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_{j=1}^{N} \tilde{w}_t^{(j)}} \\[4pt]
&\qquad \text{c. } \textbf{Compute ESS: } \widehat{\text{ESS}}_t = \frac{1}{\sum_{i=1}^{N} (w_t^{(i)})^2} \\[4pt]
&\qquad \text{d. } \textbf{Resample: } \text{If } \widehat{\text{ESS}}_t < \tau \cdot N: \\
&\qquad \qquad \text{resample } \{\theta_{t-1}^{(i)}\}, \quad w_t^{(i)} = \tfrac{1}{N} \\[4pt]
&\qquad \text{e. } \textbf{Move: } \theta_t^{(i)} \sim K_t(\cdot \mid \theta_{t-1}^{(i)}) \\[4pt]
&\text{3. } \textbf{Output: } \{(\theta_T^{(i)}, w_T^{(i)})\}_{i=1}^{N}, \quad \log \hat{Z} = \sum_{t=1}^{T} \log \left( \sum_{i=1}^{N} \tilde{w}_t^{(i)} \right)
\end{aligned}
}
$$

where $\gamma_t$ denotes the unnormalized version of $\pi_t$, i.e., $\pi_t(\theta) = \gamma_t(\theta) / Z_t$.

### Incremental Weights

The key quantity is the **incremental weight**:

$$
\alpha_t(\theta) = \frac{\gamma_t(\theta)}{\gamma_{t-1}(\theta)}
$$

For a geometric bridge (tempering schedule):

$$
\gamma_t(\theta) = \pi_0(\theta)^{1 - \beta_t} \cdot \pi^*(\theta)^{\beta_t}
$$

the incremental weight simplifies to:

$$
\alpha_t(\theta) = \left( \frac{\pi^*(\theta)}{\pi_0(\theta)} \right)^{\beta_t - \beta_{t-1}}
$$

---

## MCMC Kernels

The move step uses an MCMC kernel $K_t$ that is **invariant** with respect to $\pi_t$. particlefilterbox supports several kernels:

| Kernel | Class | Best for | Tuning |
|--------|-------|----------|--------|
| Random Walk MH | `"random_walk"` | Low-to-moderate dimensions | Step size $\sigma$ |
| Independent MH | `"independent"` | When a good proposal is available | Proposal distribution |
| HMC | `"hmc"` | High dimensions, smooth targets | Step size $\epsilon$, path length $L$ |
| MALA | `"mala"` | Moderate dimensions, smooth targets | Step size $\epsilon$ |

!!! tip "Kernel selection"
    Start with `"random_walk"` for problems with $\text{dim}(\theta) \leq 20$. For higher dimensions, switch to `"hmc"` or `"mala"` if the target gradient is available. The kernel is adapted at each step using the current particle population.

---

## API Reference

### Constructor

```python
from particlefilterbox.smc import SMCSampler, SMCConfig

config = SMCConfig(
    n_particles=2000,
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
)

sampler = SMCSampler(
    target=posterior,         # callable: log_prob(theta) -> float
    prior=prior,              # callable: log_prob(theta) -> float, sample(n) -> array
    n_steps=50,               # number of bridging distributions
    kernel="random_walk",     # MCMC kernel type
    schedule="adaptive",      # "linear", "geometric", or "adaptive"
    config=config,
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `2000` | Number of particles $N$ |
| `n_steps` | `int` | `50` | Number of bridging distributions $T$ |
| `kernel` | `str` | `"random_walk"` | MCMC kernel: `"random_walk"`, `"independent"`, `"hmc"`, `"mala"` |
| `schedule` | `str` | `"adaptive"` | Temperature schedule: `"linear"`, `"geometric"`, `"adaptive"` |
| `n_mcmc_steps` | `int` | `5` | MCMC iterations per move step |
| `target_acceptance` | `float` | `0.234` | Target acceptance rate for kernel adaptation |

### Running the Sampler

```python
result = sampler.sample()
```

### Result Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `particles` | `(N, dim)` | Final weighted particles approximating $\pi^*$ |
| `weights` | `(N,)` | Normalized importance weights |
| `log_marginal_likelihood` | scalar | $\log \hat{Z}$ --- normalizing constant estimate |
| `ess_history` | `(T,)` | ESS at each bridging step |
| `acceptance_rates` | `(T,)` | MCMC acceptance rate per step |
| `schedule` | `(T,)` | Temperature schedule $\beta_0, \ldots, \beta_T$ |

---

## Examples

### Example 1: Sampling a Multimodal Posterior

A challenging target with well-separated modes that would trap a single MCMC chain:

```python
import numpy as np
from particlefilterbox.smc import SMCSampler, SMCConfig

# --- Define a bimodal target ---
def log_target(theta):
    """Mixture of two Gaussians in 2D."""
    mu1 = np.array([-3.0, 0.0])
    mu2 = np.array([3.0, 0.0])
    sigma = 0.5

    ll1 = -0.5 * np.sum((theta - mu1)**2) / sigma**2
    ll2 = -0.5 * np.sum((theta - mu2)**2) / sigma**2

    return np.logaddexp(ll1, ll2) - np.log(2)

# --- Define the prior ---
class GaussianPrior:
    def log_prob(self, theta):
        return -0.5 * np.sum(theta**2) / 25.0  # N(0, 5I)

    def sample(self, n, rng):
        return rng.normal(0.0, 5.0, size=(n, 2))

prior = GaussianPrior()

# --- Configure and run ---
config = SMCConfig(n_particles=2000, resampling="systematic", seed=42)

sampler = SMCSampler(
    target=log_target,
    prior=prior,
    n_steps=30,
    kernel="random_walk",
    schedule="adaptive",
    config=config,
)

result = sampler.sample()

print(f"Log marginal likelihood: {result.log_marginal_likelihood:.3f}")
print(f"Final ESS: {result.ess_history[-1]:.0f} / {config.n_particles}")
print(f"Mean acceptance rate: {result.acceptance_rates.mean():.3f}")

# --- Inspect modes ---
particles = result.particles
left_mode = particles[particles[:, 0] < 0]
right_mode = particles[particles[:, 0] > 0]
print(f"Left mode:  {len(left_mode)} particles, mean = {left_mode.mean(axis=0)}")
print(f"Right mode: {len(right_mode)} particles, mean = {right_mode.mean(axis=0)}")
```

!!! tip "What to expect"
    Both modes should be well-represented in the final particle set. MCMC alone would typically get trapped in one mode, but the SMC sampler explores both by starting from the diffuse prior and gradually sharpening toward the target.

### Example 2: Bayesian Logistic Regression

Using the SMC Sampler for posterior inference in a non-conjugate model:

```python
import numpy as np
from particlefilterbox.smc import SMCSampler, SMCConfig

# --- Simulate data ---
rng = np.random.default_rng(123)
n_obs, n_features = 200, 5
X = rng.normal(0, 1, size=(n_obs, n_features))
beta_true = np.array([1.0, -0.5, 0.3, 0.0, 0.8])
prob = 1.0 / (1.0 + np.exp(-X @ beta_true))
y = rng.binomial(1, prob)

# --- Define target and prior ---
def log_likelihood(beta):
    logits = X @ beta
    return np.sum(y * logits - np.log(1 + np.exp(logits)))

def log_prior(beta):
    return -0.5 * np.sum(beta**2) / 10.0  # N(0, sqrt(10) I)

def log_target(beta):
    return log_likelihood(beta) + log_prior(beta)

class NormalPrior:
    def log_prob(self, beta):
        return log_prior(beta)

    def sample(self, n, rng):
        return rng.normal(0.0, np.sqrt(10.0), size=(n, n_features))

# --- Run SMC Sampler ---
config = SMCConfig(n_particles=3000, seed=42)

sampler = SMCSampler(
    target=log_target,
    prior=NormalPrior(),
    n_steps=40,
    kernel="random_walk",
    schedule="adaptive",
    config=config,
)

result = sampler.sample()

# --- Posterior summaries ---
w = result.weights
posterior_mean = np.average(result.particles, weights=w, axis=0)
posterior_std = np.sqrt(
    np.average((result.particles - posterior_mean)**2, weights=w, axis=0)
)

print("Parameter | True  | Post. Mean | Post. Std")
print("-" * 47)
for j in range(n_features):
    print(f"  beta_{j}  | {beta_true[j]:5.2f} | {posterior_mean[j]:10.3f} | {posterior_std[j]:9.3f}")

print(f"\nLog marginal likelihood: {result.log_marginal_likelihood:.2f}")
```

---

## Tuning Guide

### Temperature Schedule

The schedule $0 = \beta_0 < \beta_1 < \cdots < \beta_T = 1$ controls how quickly the sequence bridges from prior to target.

| Schedule | Description | When to use |
|----------|-------------|-------------|
| `"linear"` | $\beta_t = t / T$ | Simple problems, prior and target are similar |
| `"geometric"` | $\beta_t = (t/T)^2$ | More steps near the prior where particles are diverse |
| `"adaptive"` | Choose $\beta_t$ to maintain target ESS | **Recommended default** --- automatic and robust |

!!! warning "Too few steps"
    If $T$ is too small, the incremental weights become highly variable, leading to ESS collapse. With `schedule="adaptive"`, the algorithm automatically adjusts $T$ to maintain stable ESS.

### Number of MCMC Steps

The `n_mcmc_steps` parameter controls how many MCMC iterations are applied in each move step. More steps improve particle diversity but increase computational cost.

| Setting | `n_mcmc_steps` | Trade-off |
|---------|:--------------:|-----------|
| Minimal | 1 | Fast but particles may not reach equilibrium |
| Default | 5 | Good balance for most problems |
| Thorough | 10--20 | Better mixing, needed for high-dimensional targets |

### Computational Complexity

| Operation | Cost |
|-----------|------|
| Reweighting | $O(N)$ per step |
| Resampling | $O(N)$ per step |
| MCMC moves | $O(N \cdot M \cdot C_K)$ per step |
| **Total** | $O(N \cdot T \cdot M \cdot C_K)$ |

Where $M$ = `n_mcmc_steps` and $C_K$ = cost of one kernel evaluation.

---

## See Also

- [SMC Tempering](tempering.md) --- a specialized SMC sampler that focuses on likelihood tempering with adaptive $\beta$ selection
- [Waste-Free SMC](waste-free.md) --- enhances the SMC Sampler by reusing all MCMC particles
- [IBIS](ibis.md) --- an alternative approach that processes data sequentially rather than using a temperature schedule

---

## References

- Del Moral, P., Doucet, A. & Jasra, A. (2006). Sequential Monte Carlo Samplers. *Journal of the Royal Statistical Society: Series B*, 68(3), 411--436.
- Dai, C., Heng, J., Jacob, P.E. & Whiteley, N. (2022). An Invitation to Sequential Monte Carlo Samplers. *Journal of the American Statistical Association*, 117(539), 1587--1600.
- Chopin, N. & Papaspiliopoulos, O. (2020). *An Introduction to Sequential Monte Carlo*. Springer, Chapter 17.
