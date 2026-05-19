---
title: SIR Filter
description: "Sequential Importance Resampling — the general particle filter with customizable proposal distributions"
---

# SIR Filter

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `SIR` |
    | **Import** | `from particlefilterbox.filters import SIR` |
    | **Proposal** | Custom $q(x_t \mid x_{t-1}, y_t)$ or prior fallback |
    | **Complexity** | $O(N)$ per time step |
    | **Reference** | Doucet, Godsill & Andrieu (2000) |

## Overview

The Sequential Importance Resampling (SIR) filter generalizes the [Bootstrap PF](bootstrap.md) by allowing a **custom proposal distribution** $q(x_t \mid x_{t-1}, y_t)$ that incorporates the current observation. When a good proposal is available, SIR can dramatically improve particle efficiency compared to the Bootstrap filter.

If no custom proposal is provided, SIR **falls back to Bootstrap PF behavior** automatically.

**Advantages:**

- Flexible — any proposal distribution can be used
- Can achieve much higher ESS than Bootstrap when the proposal accounts for $y_t$
- Same $O(N)$ complexity as Bootstrap

**Disadvantages:**

- Requires the user to design and implement a proposal
- Requires evaluating the proposal density $\log q(x_t \mid x_{t-1}, y_t)$ and transition density $\log p(x_t \mid x_{t-1})$
- A poor proposal can be worse than the prior

---

## Algorithm

The SIR filter with a general proposal $q$:

$$
\boxed{
\begin{aligned}
&\textbf{SIR Particle Filter} \\[6pt]
&\text{1. } \textbf{Initialize: } x_0^{(i)} \sim p(x_0), \quad w_0^{(i)} = \tfrac{1}{N} \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T: \\
&\qquad \text{a. } \textbf{Propose: } x_t^{(i)} \sim q(x_t \mid x_{t-1}^{(i)}, y_t) \\
&\qquad \text{b. } \textbf{Weight: } \tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot \frac{p(y_t \mid x_t^{(i)}) \; p(x_t^{(i)} \mid x_{t-1}^{(i)})}{q(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t)} \\
&\qquad \text{c. } \textbf{Normalize: } w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_{j=1}^{N} \tilde{w}_t^{(j)}} \\
&\qquad \text{d. } \textbf{Resample: } \text{If } \text{ESS} < \tau \cdot N, \text{ resample}
\end{aligned}
}
$$

### Weight decomposition

The incremental log-weight for particle $i$ at time $t$ is:

$$
\log \tilde{w}_t^{(i)} = \underbrace{\log p(y_t \mid x_t^{(i)})}_{\text{observation likelihood}} + \underbrace{\log p(x_t^{(i)} \mid x_{t-1}^{(i)})}_{\text{transition density}} - \underbrace{\log q(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t)}_{\text{proposal density}}
$$

!!! note "Bootstrap as a special case"
    When $q(x_t \mid x_{t-1}, y_t) = p(x_t \mid x_{t-1})$, the transition and proposal terms cancel, and the weight reduces to $\log p(y_t \mid x_t^{(i)})$ — exactly the Bootstrap PF.

---

## API Reference

### Constructor

```python
from particlefilterbox.filters import SIR
from particlefilterbox.core.config import PFConfig

config = PFConfig(n_particles=1000, resampling="systematic", seed=42)
sir = SIR(model=my_model, config=config)
```

The `SIR` class automatically detects whether the model provides a custom proposal. It checks for two methods on the model:

| Model method | Signature | Purpose |
|-------------|-----------|---------|
| `proposal_sample` | `(particles, y_t, t, rng) → ndarray` | Sample from $q(x_t \mid x_{t-1}, y_t)$ |
| `log_proposal_density` | `(x_curr, x_prev, y_t, t) → ndarray` | Evaluate $\log q(x_t \mid x_{t-1}, y_t)$ |

If **both** are present, SIR uses the custom proposal. Otherwise, it falls back to the prior.

```python
# Check which mode is active
print(sir.uses_custom_proposal)  # True or False
```

### Additional Model Requirements

When using a custom proposal, the model must also provide:

| Model method | Signature | Purpose |
|-------------|-----------|---------|
| `log_transition_density` | `(x_curr, x_prev, t) → ndarray` | Evaluate $\log p(x_t \mid x_{t-1})$ |

!!! warning "Transition density required"
    Unlike the Bootstrap PF which only needs to *sample* from the transition, SIR with a custom proposal also needs to *evaluate* the transition density $p(x_t \mid x_{t-1})$. Make sure your model implements `log_transition_density`.

### Batch Filtering

```python
result = sir.filter(observations)
```

Returns the same `ParticleFilterResults` as the Bootstrap PF. See [Bootstrap PF — Batch Filtering](bootstrap.md#batch-filtering) for the full attribute list.

---

## Examples

### Example 1: Gaussian Proposal Centered on the Observation

In many models, a proposal that "looks at" the observation can substantially improve efficiency. Here we use a Gaussian proposal centered between the prior prediction and the observation.

```python
import numpy as np
from particlefilterbox.filters import SIR
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class LinearGaussianWithProposal(ParticleFilterModel):
    """
    x_t = 0.9 * x_{t-1} + eta_t,   eta_t ~ N(0, 1)
    y_t = x_t + eps_t,              eps_t ~ N(0, 0.25)

    Custom proposal: blend prior prediction and observation.
    q(x_t | x_{t-1}, y_t) = N(mu_q, sigma_q^2)
    where
        mu_q = (sigma_obs^2 * mu_prior + sigma_prior^2 * y_t) /
               (sigma_obs^2 + sigma_prior^2)
        sigma_q^2 = (sigma_obs^2 * sigma_prior^2) /
                     (sigma_obs^2 + sigma_prior^2)
    """
    k_states = 1
    k_obs = 1

    def __init__(self):
        self.phi = 0.9
        self.sigma_eta = 1.0    # transition noise std
        self.sigma_eps = 0.5    # observation noise std

        # Proposal parameters (optimal for linear Gaussian)
        var_prior = self.sigma_eta**2
        var_obs = self.sigma_eps**2
        self.var_q = (var_prior * var_obs) / (var_prior + var_obs)
        self.sigma_q = np.sqrt(self.var_q)
        self.w_obs = var_prior / (var_prior + var_obs)
        self.w_prior = var_obs / (var_prior + var_obs)

    def initial_distribution(self, n_particles, rng):
        return rng.normal(0.0, 1.0, size=(n_particles, 1))

    def transition(self, particles, t, rng):
        return self.phi * particles + rng.normal(0.0, self.sigma_eta, size=particles.shape)

    def log_transition_density(self, x_curr, x_prev, t):
        mu = self.phi * x_prev[:, 0]
        return -0.5 * ((x_curr[:, 0] - mu) / self.sigma_eta)**2

    def log_observation_likelihood(self, particles, y_t, t):
        residual = y_t[0] - particles[:, 0]
        return -0.5 * (residual / self.sigma_eps)**2

    def proposal_sample(self, particles, y_t, t, rng):
        mu_prior = self.phi * particles[:, 0]
        mu_q = self.w_prior * mu_prior + self.w_obs * y_t[0]
        samples = rng.normal(mu_q, self.sigma_q)
        return samples[:, np.newaxis]

    def log_proposal_density(self, x_curr, x_prev, y_t, t):
        mu_prior = self.phi * x_prev[:, 0]
        mu_q = self.w_prior * mu_prior + self.w_obs * y_t[0]
        return -0.5 * ((x_curr[:, 0] - mu_q) / self.sigma_q)**2

# --- Run SIR with custom proposal ---
model = LinearGaussianWithProposal()
config = PFConfig(n_particles=500, resampling="systematic", seed=42)
sir = SIR(model=model, config=config)

print(f"Using custom proposal: {sir.uses_custom_proposal}")  # True

# Simulate data
rng = np.random.default_rng(123)
T = 200
x_true = np.zeros(T)
y_obs = np.zeros(T)

x_true[0] = rng.normal(0.0, 1.0)
y_obs[0] = x_true[0] + rng.normal(0.0, 0.5)
for t in range(1, T):
    x_true[t] = 0.9 * x_true[t - 1] + rng.normal(0.0, 1.0)
    y_obs[t] = x_true[t] + rng.normal(0.0, 0.5)

result_sir = sir.filter(y_obs)
print(f"SIR log-likelihood: {result_sir.log_likelihood:.2f}")
print(f"SIR mean ESS: {result_sir.ess_history.mean():.0f} / {config.n_particles}")
```

---

## Comparison with Bootstrap PF

The key question: **when is a custom proposal worth the effort?**

### Side-by-Side Benchmark

```python
from particlefilterbox.filters import BootstrapPF

# Same data, same number of particles
config = PFConfig(n_particles=500, resampling="systematic", seed=42)

bpf = BootstrapPF(model=model, config=config)
sir = SIR(model=model, config=config)

result_bpf = bpf.filter(y_obs)
result_sir = sir.filter(y_obs)

print(f"{'Metric':<25} {'Bootstrap':>12} {'SIR':>12}")
print("-" * 50)
print(f"{'Log-likelihood':<25} {result_bpf.log_likelihood:>12.2f} {result_sir.log_likelihood:>12.2f}")
print(f"{'Mean ESS':<25} {result_bpf.ess_history.mean():>12.0f} {result_sir.ess_history.mean():>12.0f}")
print(f"{'Min ESS':<25} {result_bpf.ess_history.min():>12.0f} {result_sir.ess_history.min():>12.0f}")
print(f"{'Resampling rate':<25} {result_bpf.resampled.mean():>12.1%} {result_sir.resampled.mean():>12.1%}")
```

!!! tip "Expected results"
    With a well-chosen proposal (like the optimal linear Gaussian proposal above), SIR typically achieves:

    - **2–5× higher mean ESS** with the same number of particles
    - **Lower resampling rate** → less path degeneracy
    - **Lower variance** in log-likelihood estimates

### When to Use SIR over Bootstrap

| Scenario | Recommendation |
|----------|---------------|
| No knowledge of a good proposal | Use **Bootstrap** |
| Observation is weakly informative | Use **Bootstrap** (prior proposal is adequate) |
| Observation strongly constrains the state | Use **SIR** with an observation-informed proposal |
| Linear-Gaussian sub-structure available | Use **SIR** with the optimal proposal |
| Cost of evaluating $\log q$ and $\log p$ is high | Use **Bootstrap** (simpler weight update) |

---

## Designing Good Proposals

The quality of the SIR filter depends entirely on the proposal distribution. A good proposal should:

1. **Cover the target**: $q$ should have heavier tails than $p(x_t \mid x_{t-1}, y_t)$
2. **Incorporate $y_t$**: shift particles toward regions consistent with the observation
3. **Be cheap to sample and evaluate**: the cost per particle includes sampling from $q$ + evaluating $\log q$, $\log p(x \mid x')$, and $\log p(y \mid x)$

### Common Proposal Strategies

=== "Prior (Bootstrap)"

    $$q(x_t \mid x_{t-1}, y_t) = p(x_t \mid x_{t-1})$$

    - No observation information
    - No density evaluation needed
    - Baseline performance

=== "Optimal (Linear Gaussian)"

    $$q^*(x_t \mid x_{t-1}, y_t) = p(x_t \mid x_{t-1}, y_t)$$

    - Minimizes weight variance
    - Available in closed form only for linear-Gaussian models
    - Proposal mean is a precision-weighted average of prior and observation

=== "Extended Kalman Proposal"

    $$q(x_t \mid x_{t-1}, y_t) = \mathcal{N}(\hat{x}_t^{\text{EKF}}, P_t^{\text{EKF}})$$

    - Linearize the model around the prior prediction
    - Run one EKF update step per particle
    - Good for mildly nonlinear models

---

## Tuning Guide

### Number of Particles

With a good proposal, SIR can achieve the same accuracy as Bootstrap with **fewer particles**. A rough guideline:

| Proposal quality | Particles needed (relative to Bootstrap) |
|-----------------|:----------------------------------------:|
| Poor (misspecified) | 2× more |
| Prior (Bootstrap fallback) | Same |
| Moderate (observation-informed) | 0.5× – 0.25× |
| Optimal (linear Gaussian) | 0.1× – 0.2× |

### Resampling Strategy

Same options as Bootstrap PF. The default `"systematic"` is recommended.

### Monitoring Proposal Quality

Track these diagnostics to assess your proposal:

```python
result = sir.filter(y_obs)

# High ESS → good proposal
print(f"Mean ESS: {result.ess_history.mean():.0f}")

# Low resampling rate → good proposal
print(f"Resampling rate: {result.resampled.mean():.1%}")

# Stable log-likelihood → good proposal
print(f"Log-lik std: {np.std(result.log_likelihoods):.3f}")
```

!!! warning "Proposal mismatch"
    If the mean ESS is **lower** than with the Bootstrap PF, your proposal is likely misspecified — it may not cover the tails of the target, or its density evaluation may be incorrect. Double-check the `log_proposal_density` and `log_transition_density` implementations.

---

## References

- Doucet, A., Godsill, S. & Andrieu, C. (2000). On sequential Monte Carlo sampling methods for Bayesian filtering. *Statistics and Computing*, 10(3), 197–208.
- Arulampalam, M.S., Maskell, S., Gordon, N. & Clapp, T. (2002). A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking. *IEEE Transactions on Signal Processing*, 50(2), 174–188.
- Doucet, A. & Johansen, A.M. (2009). A tutorial on particle filtering and smoothing: fifteen years later. In *Handbook of Nonlinear Filtering*, Oxford University Press.
