---
title: Guided Particle Filter
description: "The Guided PF — observation-driven proposal via gradient or Laplace approximation"
---

# Guided Particle Filter

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `GuidedPF` |
    | **Import** | `from particlefilterbox.filters import GuidedPF` |
    | **Key idea** | Shift proposal toward the observation using gradient or Laplace approximation |
    | **Complexity** | $O(N \cdot k)$ per time step |
    | **Reference** | van der Merwe, Doucet, de Freitas & Wan (2001); Pitt & Shephard (1999) |

## Overview

The Guided Particle Filter improves the standard Bootstrap PF by constructing a **proposal distribution that is guided toward the observation**. Instead of proposing particles blindly from the prior $p(x_t \mid x_{t-1})$, the Guided PF shifts each particle's proposal toward regions where $p(y_t \mid x_t)$ is large.

Two main guidance strategies are available:

1. **Gradient guidance**: Use the gradient of the log-likelihood $\nabla_x \log p(y_t \mid x_t)$ to shift the proposal mean toward high-likelihood regions
2. **Laplace approximation**: Find the mode of $p(x_t \mid x_{t-1}, y_t) \propto p(y_t \mid x_t) p(x_t \mid x_{t-1})$ and use a Gaussian centered at the mode as the proposal

Both strategies produce proposals that account for the current observation, yielding **higher ESS** and **lower variance** than the Bootstrap PF, especially when observations are informative.

**Advantages:**

- Automatic — no manual proposal design required
- Works well when observations are highly informative (low observation noise)
- Gradient guidance is cheap — one gradient evaluation per particle
- Laplace approximation gives near-optimal proposals for unimodal posteriors

**Disadvantages:**

- Gradient guidance requires the log-likelihood to be differentiable
- Laplace approximation requires a Hessian evaluation — higher cost
- Both assume a unimodal local posterior — may fail for multimodal targets
- Requires evaluating $\log p(x_t \mid x_{t-1})$ for weight correction

---

## Algorithm

### Gradient-Guided Proposal

The gradient-guided proposal shifts the prior mean toward the observation:

$$
\boxed{
\begin{aligned}
&\textbf{Gradient-Guided Particle Filter} \\[6pt]
&\text{1. } \textbf{Initialize: } x_0^{(i)} \sim p(x_0), \quad w_0^{(i)} = \tfrac{1}{N} \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T: \\
&\qquad \text{a. } \textbf{Prior mean: } \mu_t^{(i)} = \mathbb{E}[x_t \mid x_{t-1}^{(i)}] \\
&\qquad \text{b. } \textbf{Gradient step: } \hat{\mu}_t^{(i)} = \mu_t^{(i)} + \Sigma_t \nabla_x \log p(y_t \mid x)\big|_{x=\mu_t^{(i)}} \\
&\qquad \text{c. } \textbf{Propose: } x_t^{(i)} \sim \mathcal{N}(\hat{\mu}_t^{(i)}, \Sigma_t) \\
&\qquad \text{d. } \textbf{Weight: } \tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot \frac{p(y_t \mid x_t^{(i)}) \; p(x_t^{(i)} \mid x_{t-1}^{(i)})}{q(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t)} \\
&\qquad \text{e. } \textbf{Normalize and resample as usual}
\end{aligned}
}
$$

where $\Sigma_t$ is the transition noise covariance (used as the proposal covariance).

### Laplace-Guided Proposal

The Laplace approximation finds the **mode** of the local posterior and uses a Gaussian centered there:

$$
\boxed{
\begin{aligned}
&\textbf{Laplace-Guided Particle Filter} \\[6pt]
&\text{For each particle } i \text{ at time } t: \\[4pt]
&\qquad \text{a. } \textbf{Find mode: } \hat{x}_t^{(i)} = \arg\max_x \big[\log p(y_t \mid x) + \log p(x \mid x_{t-1}^{(i)})\big] \\
&\qquad \text{b. } \textbf{Hessian: } \Lambda_t^{(i)} = -\nabla_x^2 \big[\log p(y_t \mid x) + \log p(x \mid x_{t-1}^{(i)})\big]\big|_{x=\hat{x}_t^{(i)}} \\
&\qquad \text{c. } \textbf{Propose: } x_t^{(i)} \sim \mathcal{N}(\hat{x}_t^{(i)}, (\Lambda_t^{(i)})^{-1}) \\
&\qquad \text{d. } \textbf{Weight as usual with importance correction}
\end{aligned}
}
$$

### Why Does Guidance Help?

The key insight is that the **optimal proposal** is $q^*(x_t \mid x_{t-1}, y_t) = p(x_t \mid x_{t-1}, y_t)$, which minimizes the conditional variance of the weights. Both gradient and Laplace guidance approximate this optimal proposal:

- **Gradient guidance** takes one Newton-like step toward the mode — cheap but approximate
- **Laplace approximation** finds the exact mode and matches the curvature — more accurate but requires optimization

For strongly informative observations (small $R$), both methods dramatically outperform the Bootstrap PF.

---

## API Reference

### Constructor

```python
from particlefilterbox.filters import GuidedPF
from particlefilterbox.core.config import PFConfig

config = PFConfig(
    n_particles=1000,
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
)

gpf = GuidedPF(model=my_model, config=config, guidance="gradient")
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `1000` | Number of particles $N$ |
| `resampling` | `str` | `"systematic"` | Resampling scheme |
| `ess_threshold` | `float` | `0.5` | Resample when $\text{ESS} < \tau \cdot N$ |
| `seed` | `int \| None` | `None` | Random seed |
| `guidance` | `str` | `"gradient"` | Guidance strategy: `"gradient"` or `"laplace"` |
| `step_size` | `float` | `1.0` | Scaling for the gradient step (gradient mode only) |
| `max_iter` | `int` | `10` | Maximum optimization iterations (Laplace mode only) |

### Model Requirements

| Model method | Signature | Purpose |
|-------------|-----------|---------|
| `initial_distribution` | `(n_particles, rng) → ndarray` | Sample $x_0$ |
| `transition` | `(particles, t, rng) → ndarray` | Sample $x_t \mid x_{t-1}$ |
| `transition_mean` | `(particles, t) → ndarray` | Deterministic transition mean |
| `log_transition_density` | `(x_curr, x_prev, t) → ndarray` | Evaluate $\log p(x_t \mid x_{t-1})$ |
| `log_observation_likelihood` | `(particles, y_t, t) → ndarray` | Evaluate $\log p(y_t \mid x_t)$ |
| `log_likelihood_gradient` | `(x, y_t, t) → ndarray` | $\nabla_x \log p(y_t \mid x)$ (gradient mode) |
| `transition_noise_cov` | `(t) → ndarray` | Transition covariance $\Sigma_t$ |

For Laplace mode, additionally:

| Model method | Signature | Purpose |
|-------------|-----------|---------|
| `log_likelihood_hessian` | `(x, y_t, t) → ndarray` | $\nabla_x^2 \log p(y_t \mid x)$ |

### Batch Filtering

```python
result = gpf.filter(observations)
```

Returns the same `ParticleFilterResults` as Bootstrap PF. See [Bootstrap PF — Batch Filtering](bootstrap.md#batch-filtering).

---

## Examples

### Example 1: Gradient-Guided Filter for Precise Observations

When the observation noise is very low, the Bootstrap PF wastes most particles. The gradient-guided proposal shifts particles toward the observed value.

```python
import numpy as np
from particlefilterbox.filters import GuidedPF, BootstrapPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class PreciseObservation(ParticleFilterModel):
    """
    x_t = 0.9 * x_{t-1} + eta_t,   eta_t ~ N(0, 1)
    y_t = x_t^2 + eps_t,            eps_t ~ N(0, 0.01)

    The observation is very precise (sigma=0.1) and nonlinear.
    """
    k_states = 1
    k_obs = 1

    def initial_distribution(self, n_particles, rng):
        return rng.normal(0.0, 1.0, size=(n_particles, 1))

    def transition(self, particles, t, rng):
        return 0.9 * particles + rng.normal(0.0, 1.0, size=particles.shape)

    def transition_mean(self, particles, t):
        return 0.9 * particles

    def log_transition_density(self, x_curr, x_prev, t):
        mu = 0.9 * x_prev[:, 0]
        return -0.5 * (x_curr[:, 0] - mu)**2

    def log_observation_likelihood(self, particles, y_t, t):
        pred = particles[:, 0]**2
        return -0.5 * ((y_t[0] - pred) / 0.1)**2

    def log_likelihood_gradient(self, x, y_t, t):
        # d/dx log p(y|x) = d/dx [-0.5 * (y - x^2)^2 / 0.01]
        # = (y - x^2) * 2x / 0.01
        grad = (y_t[0] - x[:, 0]**2) * 2 * x[:, 0] / 0.01
        return grad[:, np.newaxis]

    def transition_noise_cov(self, t):
        return np.array([[1.0]])

# --- Simulate ---
rng = np.random.default_rng(42)
T = 200

x_true = np.zeros(T)
y_obs = np.zeros(T)
x_true[0] = rng.normal(0, 1)
y_obs[0] = x_true[0]**2 + rng.normal(0, 0.1)
for t in range(1, T):
    x_true[t] = 0.9 * x_true[t-1] + rng.normal(0, 1)
    y_obs[t] = x_true[t]**2 + rng.normal(0, 0.1)

# --- Compare ---
config = PFConfig(n_particles=1000, resampling="systematic", seed=42)

gpf = GuidedPF(model=PreciseObservation(), config=config, guidance="gradient")
bpf = BootstrapPF(model=PreciseObservation(), config=config)

result_gpf = gpf.filter(y_obs)
result_bpf = bpf.filter(y_obs)

rmse_gpf = np.sqrt(np.mean((result_gpf.filtered_means[:, 0] - x_true)**2))
rmse_bpf = np.sqrt(np.mean((result_bpf.filtered_means[:, 0] - x_true)**2))

print(f"{'Metric':<25} {'Bootstrap':>12} {'Guided':>12}")
print("-" * 50)
print(f"{'RMSE':<25} {rmse_bpf:>12.4f} {rmse_gpf:>12.4f}")
print(f"{'Log-likelihood':<25} {result_bpf.log_likelihood:>12.2f} {result_gpf.log_likelihood:>12.2f}")
print(f"{'Mean ESS':<25} {result_bpf.ess_history.mean():>12.0f} {result_gpf.ess_history.mean():>12.0f}")
```

!!! tip "What to expect"
    With observation noise $\sigma = 0.1$, the Bootstrap PF should show very low ESS (most particles are far from the observed $y_t$). The Guided PF should show **3–10× higher ESS** by shifting particles toward the high-likelihood region.

### Example 2: Laplace-Guided Filter for Stochastic Volatility

```python
import numpy as np
from particlefilterbox.filters import GuidedPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class SVGuided(ParticleFilterModel):
    """
    x_t = 0.98 * x_{t-1} + 0.16 * eta_t    (log-vol)
    y_t = 0.65 * exp(x_t/2) * eps_t          (returns)
    """
    k_states = 1
    k_obs = 1

    def __init__(self, phi=0.98, sigma=0.16, beta=0.65):
        self.phi = phi
        self.sigma = sigma
        self.beta = beta

    def initial_distribution(self, n_particles, rng):
        std = self.sigma / np.sqrt(1 - self.phi**2)
        return rng.normal(0.0, std, size=(n_particles, 1))

    def transition(self, particles, t, rng):
        return self.phi * particles + rng.normal(0.0, self.sigma, size=particles.shape)

    def transition_mean(self, particles, t):
        return self.phi * particles

    def log_transition_density(self, x_curr, x_prev, t):
        mu = self.phi * x_prev[:, 0]
        return -0.5 * ((x_curr[:, 0] - mu) / self.sigma)**2

    def log_observation_likelihood(self, particles, y_t, t):
        vol = self.beta * np.exp(particles[:, 0] / 2)
        return -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (y_t[0] / vol)**2

    def log_likelihood_gradient(self, x, y_t, t):
        # d/dx [-log(vol) - 0.5*(y/vol)^2]  where vol = beta*exp(x/2)
        # = -0.5 + 0.5 * (y / (beta*exp(x/2)))^2
        vol = self.beta * np.exp(x[:, 0] / 2)
        grad = -0.5 + 0.5 * (y_t[0] / vol)**2
        return grad[:, np.newaxis]

    def log_likelihood_hessian(self, x, y_t, t):
        vol = self.beta * np.exp(x[:, 0] / 2)
        hess = -0.5 * (y_t[0] / vol)**2
        return hess[:, np.newaxis, np.newaxis]

    def transition_noise_cov(self, t):
        return np.array([[self.sigma**2]])

# --- Simulate ---
sv = SVGuided()
rng = np.random.default_rng(456)
T = 500

x_true = np.zeros(T)
y_obs = np.zeros(T)
std_0 = sv.sigma / np.sqrt(1 - sv.phi**2)
x_true[0] = rng.normal(0.0, std_0)
y_obs[0] = sv.beta * np.exp(x_true[0] / 2) * rng.normal()
for t in range(1, T):
    x_true[t] = sv.phi * x_true[t-1] + rng.normal(0, sv.sigma)
    y_obs[t] = sv.beta * np.exp(x_true[t] / 2) * rng.normal()

# --- Filter ---
config = PFConfig(n_particles=1000, resampling="systematic", seed=42)
gpf = GuidedPF(model=sv, config=config, guidance="laplace", max_iter=5)
result = gpf.filter(y_obs)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Mean ESS: {result.ess_history.mean():.0f} / {config.n_particles}")
```

---

## Tuning Guide

### Guidance Strategy Selection

| Strategy | Cost per particle | Accuracy | When to use |
|----------|:-----------------:|:--------:|-------------|
| **Gradient** | $O(k)$ | Moderate | Default choice; cheap, works for most models |
| **Laplace** | $O(k^3)$ | High | When you need near-optimal proposals; smooth models |

### Step Size (Gradient Mode)

The `step_size` parameter controls how far the proposal is shifted toward the observation:

| Value | Effect |
|-------|--------|
| `0.0` | No guidance — reduces to Bootstrap PF |
| `0.5` | Conservative guidance — less shift |
| `1.0` | Full Newton-like step (default) |
| `> 1.0` | Aggressive — may overshoot |

!!! warning "Gradient step too large"
    If the step size is too large, particles may be pushed into low-prior-density regions, causing the transition density term in the weight to dominate. Monitor ESS — if it drops below Bootstrap levels, reduce `step_size`.

### When to Use the Guided PF

| Scenario | Recommendation |
|----------|---------------|
| Informative observations, differentiable likelihood | **Use Guided PF** |
| Very informative observations, smooth model | Use Laplace guidance |
| No gradient available | Use [Auxiliary PF](auxiliary.md) or [UPF](upf.md) |
| Linear sub-structure available | Use [RBPF](rbpf.md) — exact marginalization is better |
| Multimodal posterior | Gradient/Laplace may miss modes — use [Regularized PF](regularized.md) |

### Computational Complexity

| Operation | Gradient mode | Laplace mode |
|-----------|:------------:|:------------:|
| Transition mean | $O(N)$ | $O(N)$ |
| Guidance computation | $O(N \cdot k)$ | $O(N \cdot k^3)$ |
| Proposal sampling | $O(N \cdot k)$ | $O(N \cdot k)$ |
| Weight computation | $O(N)$ | $O(N)$ |
| **Total per step** | **$O(N \cdot k)$** | **$O(N \cdot k^3)$** |

---

## References

- van der Merwe, R., Doucet, A., de Freitas, N. & Wan, E. (2001). The Unscented Particle Filter. In *Advances in Neural Information Processing Systems 13 (NIPS)*.
- Pitt, M.K. & Shephard, N. (1999). Filtering via Simulation: Auxiliary Particle Filters. *Journal of the American Statistical Association*, 94(446), 590–599.
- Doucet, A., Godsill, S. & Andrieu, C. (2000). On sequential Monte Carlo sampling methods for Bayesian filtering. *Statistics and Computing*, 10(3), 197–208.
- Doucet, A. & Johansen, A.M. (2009). A tutorial on particle filtering and smoothing: fifteen years later. In *Handbook of Nonlinear Filtering*, Oxford University Press.
