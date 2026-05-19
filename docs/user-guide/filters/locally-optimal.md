---
title: Locally Optimal Particle Filter
description: "The Locally Optimal PF — minimum conditional weight variance via the exact posterior proposal"
---

# Locally Optimal Particle Filter

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `LocallyOptimalPF` |
    | **Import** | `from particlefilterbox.filters import LocallyOptimalPF` |
    | **Key idea** | Use the exact conditional posterior $p(x_t \mid x_{t-1}, y_t)$ as the proposal |
    | **Complexity** | $O(N)$ per time step (for models where the optimal proposal is available) |
    | **Reference** | Doucet, Godsill & Andrieu (2000) |

## Overview

The Locally Optimal Particle Filter uses the **theoretically optimal proposal distribution**:

$$
q^*(x_t \mid x_{t-1}^{(i)}, y_t) = p(x_t \mid x_{t-1}^{(i)}, y_t)
$$

This proposal **minimizes the conditional variance** of the importance weights given $x_{t-1}^{(i)}$ and $y_t$. It is the best possible proposal in the class of proposals that depend on $(x_{t-1}, y_t)$.

The catch: the optimal proposal is only available in **closed form for specific model classes** — primarily models with linear-Gaussian observation equations or discrete observation spaces. For other models, the Locally Optimal PF provides a framework for computing the optimal proposal numerically.

**Advantages:**

- **Minimum weight variance** — the best possible ESS for a given number of particles
- Weights depend only on the previous particle, not the proposed state — more stable
- Exact for linear-Gaussian observation models (any transition)
- Can be approximated numerically for other models

**Disadvantages:**

- Only available in closed form for restricted model classes
- Requires evaluating the predictive likelihood $p(y_t \mid x_{t-1})$
- Numerical computation can be expensive for high-dimensional observation spaces
- For models with nonlinear observations, consider [Guided PF](guided.md) or [UPF](upf.md) as approximations

---

## Algorithm

$$
\boxed{
\begin{aligned}
&\textbf{Locally Optimal Particle Filter} \\[6pt]
&\text{1. } \textbf{Initialize: } x_0^{(i)} \sim p(x_0), \quad w_0^{(i)} = \tfrac{1}{N} \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T: \\
&\qquad \text{a. } \textbf{Propose: } x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)}, y_t) \\
&\qquad \text{b. } \textbf{Weight: } \tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot p(y_t \mid x_{t-1}^{(i)}) \\
&\qquad \text{c. } \textbf{Normalize: } w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_j \tilde{w}_t^{(j)}} \\
&\qquad \text{d. } \textbf{Resample: } \text{If } \text{ESS} < \tau \cdot N, \text{ resample}
\end{aligned}
}
$$

### Weight Simplification

The general importance weight for the SIR filter is:

$$
\tilde{w}_t^{(i)} \propto \frac{p(y_t \mid x_t^{(i)}) \, p(x_t^{(i)} \mid x_{t-1}^{(i)})}{q(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t)}
$$

When $q = q^* = p(x_t \mid x_{t-1}, y_t)$, we can write:

$$
p(x_t \mid x_{t-1}, y_t) = \frac{p(y_t \mid x_t) \, p(x_t \mid x_{t-1})}{p(y_t \mid x_{t-1})}
$$

Substituting into the weight expression:

$$
\tilde{w}_t^{(i)} \propto \frac{p(y_t \mid x_t^{(i)}) \, p(x_t^{(i)} \mid x_{t-1}^{(i)})}{p(y_t \mid x_t^{(i)}) \, p(x_t^{(i)} \mid x_{t-1}^{(i)}) / p(y_t \mid x_{t-1}^{(i)})} = p(y_t \mid x_{t-1}^{(i)})
$$

The weight **does not depend on the proposed $x_t^{(i)}$** — only on the predictive likelihood at the previous particle. This eliminates all sampling noise from the weights, which is why the variance is minimized.

### When Is the Optimal Proposal Available?

The optimal proposal $p(x_t \mid x_{t-1}, y_t)$ is available in closed form when:

=== "Linear-Gaussian Observations"

    If $y_t = H x_t + \varepsilon_t$ with $\varepsilon_t \sim \mathcal{N}(0, R)$ and $x_t \mid x_{t-1} \sim \mathcal{N}(\mu_t, \Sigma_t)$:

    $$
    p(x_t \mid x_{t-1}, y_t) = \mathcal{N}(m_t, V_t)
    $$

    where:

    $$
    \begin{aligned}
    V_t &= (\Sigma_t^{-1} + H^\top R^{-1} H)^{-1} \\
    m_t &= V_t (\Sigma_t^{-1} \mu_t + H^\top R^{-1} y_t)
    \end{aligned}
    $$

    and the predictive likelihood is:

    $$
    p(y_t \mid x_{t-1}) = \mathcal{N}(y_t; H \mu_t, H \Sigma_t H^\top + R)
    $$

=== "Discrete Observations"

    If $y_t$ takes values in a finite set and $x_t$ is continuous:

    $$
    p(x_t \mid x_{t-1}, y_t) = \frac{p(y_t \mid x_t) \, p(x_t \mid x_{t-1})}{p(y_t \mid x_{t-1})}
    $$

    The normalizing constant $p(y_t \mid x_{t-1})$ can be computed by integration (often analytically for exponential family transitions).

=== "Numerical Approximation"

    For general models, the optimal proposal can be approximated via:

    - **Quadrature**: discretize $x_t$ and compute $p(x_t \mid x_{t-1}, y_t)$ on a grid
    - **MCMC**: run a few MCMC steps targeting $p(x_t \mid x_{t-1}, y_t)$
    - **Laplace approximation**: see [Guided PF](guided.md) with Laplace guidance

---

## API Reference

### Constructor

```python
from particlefilterbox.filters import LocallyOptimalPF
from particlefilterbox.core.config import PFConfig

config = PFConfig(
    n_particles=500,
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
)

lopf = LocallyOptimalPF(model=my_model, config=config)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `1000` | Number of particles $N$ |
| `resampling` | `str` | `"systematic"` | Resampling scheme |
| `ess_threshold` | `float` | `0.5` | Resample when $\text{ESS} < \tau \cdot N$ |
| `seed` | `int \| None` | `None` | Random seed |
| `method` | `str` | `"analytic"` | Optimal proposal method: `"analytic"`, `"quadrature"` |

### Model Requirements

For the analytic method (linear-Gaussian observations):

| Model method | Signature | Purpose |
|-------------|-----------|---------|
| `initial_distribution` | `(n_particles, rng) → ndarray` | Sample $x_0$ |
| `transition` | `(particles, t, rng) → ndarray` | Sample $x_t \mid x_{t-1}$ |
| `transition_mean` | `(particles, t) → ndarray` | Transition mean $\mu_t$ |
| `transition_cov` | `(t) → ndarray` | Transition covariance $\Sigma_t$ |
| `observation_matrix` | `(t) → ndarray` | Observation matrix $H$ |
| `observation_noise_cov` | `(t) → ndarray` | Observation noise covariance $R$ |
| `log_observation_likelihood` | `(particles, y_t, t) → ndarray` | Evaluate $\log p(y_t \mid x_t)$ |

### Batch Filtering

```python
result = lopf.filter(observations)
```

Returns the same `ParticleFilterResults` as Bootstrap PF. See [Bootstrap PF — Batch Filtering](bootstrap.md#batch-filtering).

---

## Examples

### Example 1: Nonlinear Transition with Linear Observation

The canonical use case: the transition is nonlinear (so we need particles), but the observation is linear Gaussian (so the optimal proposal is available).

```python
import numpy as np
from particlefilterbox.filters import LocallyOptimalPF, BootstrapPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class NonlinearTransLinearObs(ParticleFilterModel):
    """
    x_t = 0.5 * x_{t-1} + 25 * x_{t-1} / (1 + x_{t-1}^2)
          + 8 * cos(1.2 * t) + eta_t,   eta_t ~ N(0, 10)
    y_t = x_t / 20 + eps_t,             eps_t ~ N(0, 1)

    This is the classic benchmark from Gordon et al. (1993).
    The optimal proposal is available because y_t is linear in x_t.
    """
    k_states = 1
    k_obs = 1

    def initial_distribution(self, n_particles, rng):
        return rng.normal(0.0, np.sqrt(5.0), size=(n_particles, 1))

    def _mean_fn(self, x, t):
        return 0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * np.cos(1.2 * t)

    def transition(self, particles, t, rng):
        mean = self._mean_fn(particles, t)
        return mean + rng.normal(0.0, np.sqrt(10.0), size=particles.shape)

    def transition_mean(self, particles, t):
        return self._mean_fn(particles, t)

    def transition_cov(self, t):
        return np.array([[10.0]])

    def observation_matrix(self, t):
        return np.array([[1.0 / 20.0]])

    def observation_noise_cov(self, t):
        return np.array([[1.0]])

    def log_observation_likelihood(self, particles, y_t, t):
        pred = particles[:, 0] / 20.0
        return -0.5 * (y_t[0] - pred)**2

# --- Simulate ---
rng = np.random.default_rng(42)
T = 200
model = NonlinearTransLinearObs()

x_true = np.zeros(T)
y_obs = np.zeros(T)
x_true[0] = rng.normal(0, np.sqrt(5))
y_obs[0] = x_true[0] / 20 + rng.normal(0, 1)
for t in range(1, T):
    x_true[t] = model._mean_fn(x_true[t-1:t], t)[0] + rng.normal(0, np.sqrt(10))
    y_obs[t] = x_true[t] / 20 + rng.normal(0, 1)

# --- Compare ---
config = PFConfig(n_particles=500, resampling="systematic", seed=42)

lopf = LocallyOptimalPF(model=model, config=config)
bpf = BootstrapPF(model=model, config=config)

result_lopf = lopf.filter(y_obs)
result_bpf = bpf.filter(y_obs)

rmse_lopf = np.sqrt(np.mean((result_lopf.filtered_means[:, 0] - x_true)**2))
rmse_bpf = np.sqrt(np.mean((result_bpf.filtered_means[:, 0] - x_true)**2))

print(f"{'Metric':<25} {'Bootstrap':>12} {'Locally Opt':>12}")
print("-" * 50)
print(f"{'RMSE':<25} {rmse_bpf:>12.4f} {rmse_lopf:>12.4f}")
print(f"{'Log-likelihood':<25} {result_bpf.log_likelihood:>12.2f} {result_lopf.log_likelihood:>12.2f}")
print(f"{'Mean ESS':<25} {result_bpf.ess_history.mean():>12.0f} {result_lopf.ess_history.mean():>12.0f}")
print(f"{'Min ESS':<25} {result_bpf.ess_history.min():>12.0f} {result_lopf.ess_history.min():>12.0f}")
```

!!! tip "What to expect"
    The Locally Optimal PF should achieve **near-perfect ESS** (close to $N$) because the weights only depend on the predictive likelihood, not the proposed state. The Bootstrap PF will show significantly lower ESS, especially when the observation is informative.

### Example 2: Time-Varying Parameter Model

```python
import numpy as np
from particlefilterbox.filters import LocallyOptimalPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class TVPModel(ParticleFilterModel):
    """
    Time-varying parameter regression:
    beta_t = beta_{t-1} + eta_t,   eta_t ~ N(0, Q)
    y_t = X_t * beta_t + eps_t,    eps_t ~ N(0, R)

    The observation is linear in beta_t, so the optimal proposal
    is available in closed form.
    """
    k_states = 2
    k_obs = 1

    def __init__(self, Q_diag=(0.01, 0.01), R=0.5):
        self.Q = np.diag(Q_diag)
        self.R_val = R
        self.X_data = None  # set before filtering

    def initial_distribution(self, n_particles, rng):
        return rng.normal(0.0, 1.0, size=(n_particles, 2))

    def transition(self, particles, t, rng):
        return particles + rng.multivariate_normal(
            np.zeros(2), self.Q, size=particles.shape[0]
        )

    def transition_mean(self, particles, t):
        return particles.copy()

    def transition_cov(self, t):
        return self.Q

    def observation_matrix(self, t):
        return self.X_data[t:t+1]  # (1, 2) matrix

    def observation_noise_cov(self, t):
        return np.array([[self.R_val]])

    def log_observation_likelihood(self, particles, y_t, t):
        pred = particles @ self.X_data[t]
        return -0.5 * ((y_t[0] - pred) / np.sqrt(self.R_val))**2

# --- Simulate ---
rng = np.random.default_rng(123)
T = 300

X = np.column_stack([np.ones(T), rng.normal(0, 1, T)])
beta_true = np.zeros((T, 2))
beta_true[0] = [1.0, 0.5]
y_obs = np.zeros(T)
y_obs[0] = X[0] @ beta_true[0] + rng.normal(0, np.sqrt(0.5))

for t in range(1, T):
    beta_true[t] = beta_true[t-1] + rng.multivariate_normal(np.zeros(2), 0.01 * np.eye(2))
    y_obs[t] = X[t] @ beta_true[t] + rng.normal(0, np.sqrt(0.5))

# --- Filter ---
model = TVPModel(Q_diag=(0.01, 0.01), R=0.5)
model.X_data = X

config = PFConfig(n_particles=500, resampling="systematic", seed=42)
lopf = LocallyOptimalPF(model=model, config=config)
result = lopf.filter(y_obs)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Mean ESS: {result.ess_history.mean():.0f} / {config.n_particles}")
print(f"Final beta estimate: [{result.filtered_means[-1, 0]:.3f}, {result.filtered_means[-1, 1]:.3f}]")
print(f"True final beta:     [{beta_true[-1, 0]:.3f}, {beta_true[-1, 1]:.3f}]")
```

!!! tip "What to expect"
    The Locally Optimal PF should track the time-varying parameters with near-perfect ESS. For linear-Gaussian observation models, the optimal proposal eliminates all sampling noise from the weights.

---

## Tuning Guide

### Number of Particles

Because the optimal proposal minimizes weight variance, the Locally Optimal PF needs **fewer particles** than any other filter for the same accuracy:

| Scenario | Particles (relative to Bootstrap) |
|----------|:--------------------------------:|
| Linear-Gaussian observations | 0.05× – 0.2× |
| Mildly nonlinear observations (with approximation) | 0.2× – 0.5× |

### When to Use the Locally Optimal PF

| Scenario | Recommendation |
|----------|---------------|
| Linear-Gaussian observation equation | **Use Locally Optimal PF** — exact optimal proposal available |
| Nonlinear transition, linear observation | **Use Locally Optimal PF** — best possible efficiency |
| Nonlinear observation, smooth model | Use [Guided PF](guided.md) or [UPF](upf.md) — approximate the optimal proposal |
| Mixed linear/nonlinear state | Use [RBPF](rbpf.md) — different but complementary approach |
| High-dimensional observations | Optimal proposal cost may be high — consider [Auxiliary PF](auxiliary.md) |

### Computational Complexity

For the analytic (linear-Gaussian observation) case:

| Operation | Cost |
|-----------|------|
| Transition mean | $O(N \cdot k)$ |
| Optimal proposal parameters ($V_t, m_t$) | $O(k^2 \cdot k_y + k^3)$ — computed once, shared |
| Proposal sampling | $O(N \cdot k)$ |
| Predictive likelihood | $O(N \cdot k_y)$ |
| **Total per step** | **$O(N \cdot k + k^3)$** |

!!! note "Shared computation"
    The proposal covariance $V_t$ depends only on $\Sigma_t$, $H$, and $R$ — not on the individual particles. It is computed **once** per time step and shared across all particles. Only the proposal mean $m_t^{(i)}$ is particle-specific.

---

## References

- Doucet, A., Godsill, S. & Andrieu, C. (2000). On sequential Monte Carlo sampling methods for Bayesian filtering. *Statistics and Computing*, 10(3), 197–208.
- Zaritskii, V.S., Svetnik, V.B. & Shimelevich, L.I. (1975). Monte Carlo technique in problems of optimal information processing. *Automation and Remote Control*, 36, 2015–2022.
- Doucet, A. & Johansen, A.M. (2009). A tutorial on particle filtering and smoothing: fifteen years later. In *Handbook of Nonlinear Filtering*, Oxford University Press.
- Chopin, N. & Papaspiliopoulos, O. (2020). *An Introduction to Sequential Monte Carlo*. Springer.
