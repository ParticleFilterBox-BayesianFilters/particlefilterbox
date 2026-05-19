---
title: Bootstrap Particle Filter
description: "The Bootstrap PF (Gordon et al., 1993) — the simplest and most general particle filter"
---

# Bootstrap Particle Filter

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `BootstrapPF` |
    | **Import** | `from particlefilterbox.filters import BootstrapPF` |
    | **Proposal** | Prior: $q(x_t \mid x_{t-1}, y_t) = p(x_t \mid x_{t-1})$ |
    | **Complexity** | $O(N)$ per time step |
    | **Reference** | Gordon, Salmond & Smith (1993) |

## Overview

The Bootstrap Particle Filter is the original and simplest particle filter. Introduced by Gordon, Salmond, and Smith (1993), it uses the **state transition density as the proposal distribution**. This makes it universally applicable — it requires only the ability to *simulate* from the transition and *evaluate* the observation likelihood.

**Advantages:**

- Simple to implement — no derivatives, no Jacobians
- Works with any nonlinear, non-Gaussian state-space model
- No model-specific tuning beyond `n_particles`

**Disadvantages:**

- Inefficient when the observation is highly informative (particles are proposed "blind" to $y_t$)
- Can suffer from weight degeneracy in high dimensions
- May require many particles to achieve acceptable ESS

---

## Algorithm

The Bootstrap PF repeats the following steps for each observation $y_t$, $t = 1, \ldots, T$:

$$
\boxed{
\begin{aligned}
&\textbf{Bootstrap Particle Filter} \\[6pt]
&\text{1. } \textbf{Initialize: } \text{For } i = 1, \ldots, N: \quad x_0^{(i)} \sim p(x_0), \quad w_0^{(i)} = \tfrac{1}{N} \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T: \\
&\qquad \text{a. } \textbf{Propagate: } x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)}) \\
&\qquad \text{b. } \textbf{Weight: } \tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot p(y_t \mid x_t^{(i)}) \\
&\qquad \text{c. } \textbf{Normalize: } w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_{j=1}^{N} \tilde{w}_t^{(j)}} \\
&\qquad \text{d. } \textbf{Resample: } \text{If } \text{ESS} < \tau \cdot N, \text{ resample indices } \{a_t^{(i)}\} \\
&\qquad \phantom{\text{d. }} \text{and set } x_t^{(i)} \leftarrow x_t^{(a_t^{(i)})}, \quad w_t^{(i)} = \tfrac{1}{N}
\end{aligned}
}
$$

### Why does the weight simplify?

Since the proposal equals the prior, the general importance weight:

$$
w_t^{(i)} \propto \frac{p(y_t \mid x_t^{(i)}) \, p(x_t^{(i)} \mid x_{t-1}^{(i)})}{q(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t)}
$$

simplifies to:

$$
w_t^{(i)} \propto p(y_t \mid x_t^{(i)})
$$

because $q = p(x_t \mid x_{t-1})$ cancels the transition term in the numerator.

---

## API Reference

### Constructor

```python
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig

config = PFConfig(
    n_particles=1000,
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
)

bpf = BootstrapPF(model=my_model, config=config)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `1000` | Number of particles $N$ |
| `resampling` | `str` | `"systematic"` | Resampling scheme: `"multinomial"`, `"systematic"`, `"stratified"`, `"residual"` |
| `ess_threshold` | `float` | `0.5` | Resample when $\text{ESS} < \tau \cdot N$ |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `store_particles` | `bool` | `False` | Store full particle history (higher memory) |

### Batch Filtering

```python
result = bpf.filter(observations)
```

| Result attribute | Shape | Description |
|------------------|-------|-------------|
| `filtered_means` | `(T, k)` | Weighted mean at each time step |
| `filtered_covs` | `(T, k, k)` | Weighted covariance at each step |
| `log_likelihood` | scalar | Total log-marginal likelihood $\log \hat{p}(y_{1:T})$ |
| `log_likelihoods` | `(T,)` | Incremental log-likelihoods |
| `ess_history` | `(T,)` | Effective sample size at each step |
| `resampled` | `(T,)` | Boolean mask of resampling events |
| `final_cloud` | `ParticleCloud` | Final particle cloud |

### Online Filtering

```python
rng = np.random.default_rng(42)
cloud = bpf.initialize(rng)

for t, y_t in enumerate(observations):
    cloud, ll_t = bpf.filter_step(cloud, y_t, t)
```

---

## Examples

### Example 1: Linear Gaussian Model (Comparison with Kalman Filter)

This example applies the Bootstrap PF to a simple linear Gaussian model where the Kalman filter provides the exact solution. This lets us verify the particle filter's accuracy.

!!! note "Requires kalmanbox"
    This example uses [kalmanbox](https://github.com/guhaase/kalmanbox) for the Kalman filter benchmark. Install with `pip install kalmanbox`.

```python
import numpy as np
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

# --- Define a linear Gaussian state-space model ---
# x_t = 0.9 * x_{t-1} + eta_t,   eta_t ~ N(0, 1)
# y_t = x_t + eps_t,              eps_t ~ N(0, 0.5^2)

class LinearGaussian(ParticleFilterModel):
    k_states = 1
    k_obs = 1

    def initial_distribution(self, n_particles, rng):
        return rng.normal(0.0, 1.0, size=(n_particles, 1))

    def transition(self, particles, t, rng):
        return 0.9 * particles + rng.normal(0.0, 1.0, size=particles.shape)

    def log_observation_likelihood(self, particles, y_t, t):
        # y_t | x_t ~ N(x_t, 0.25)
        residual = y_t - particles[:, 0]
        return -0.5 * residual**2 / 0.25

model = LinearGaussian()

# --- Simulate data ---
rng = np.random.default_rng(123)
T = 200
x_true = np.zeros(T)
y_obs = np.zeros(T)

x_true[0] = rng.normal(0.0, 1.0)
y_obs[0] = x_true[0] + rng.normal(0.0, 0.5)
for t in range(1, T):
    x_true[t] = 0.9 * x_true[t - 1] + rng.normal(0.0, 1.0)
    y_obs[t] = x_true[t] + rng.normal(0.0, 0.5)

# --- Bootstrap PF ---
config = PFConfig(n_particles=1000, resampling="systematic", seed=42)
bpf = BootstrapPF(model=model, config=config)
result = bpf.filter(y_obs)

print(f"PF  log-likelihood: {result.log_likelihood:.2f}")
print(f"Mean ESS: {result.ess_history.mean():.0f} / {config.n_particles}")

# --- Kalman filter benchmark ---
from kalmanbox import KalmanFilter

kf = KalmanFilter(
    F=np.array([[0.9]]),
    H=np.array([[1.0]]),
    Q=np.array([[1.0]]),
    R=np.array([[0.25]]),
    x0=np.array([0.0]),
    P0=np.array([[1.0]]),
)
kf_result = kf.filter(y_obs.reshape(-1, 1))

# --- Compare ---
rmse_pf = np.sqrt(np.mean((result.filtered_means[:, 0] - x_true) ** 2))
rmse_kf = np.sqrt(np.mean((kf_result.filtered_means[:, 0] - x_true) ** 2))
print(f"RMSE  PF: {rmse_pf:.4f}")
print(f"RMSE  KF: {rmse_kf:.4f}")
```

!!! tip "What to expect"
    With 1000 particles, the Bootstrap PF RMSE should be close to the Kalman filter. Increasing `n_particles` will narrow the gap, but with diminishing returns past ~5000 for this simple model.

### Example 2: Nonlinear Stochastic Volatility Model

A classic benchmark for particle filters — the stochastic volatility model where the Kalman filter cannot be applied.

```python
import numpy as np
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class StochasticVolatility(ParticleFilterModel):
    """
    x_t = phi * x_{t-1} + sigma * eta_t      (log-volatility)
    y_t = beta * exp(x_t / 2) * eps_t         (returns)
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
        noise = rng.normal(0.0, self.sigma, size=particles.shape)
        return self.phi * particles + noise

    def log_observation_likelihood(self, particles, y_t, t):
        vol = self.beta * np.exp(particles[:, 0] / 2)
        return -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (y_t[0] / vol) ** 2

# --- Simulate ---
sv = StochasticVolatility(phi=0.98, sigma=0.16, beta=0.65)
rng = np.random.default_rng(456)
T = 500

x_true = np.zeros(T)
y_obs = np.zeros(T)

std_0 = sv.sigma / np.sqrt(1 - sv.phi**2)
x_true[0] = rng.normal(0.0, std_0)
y_obs[0] = sv.beta * np.exp(x_true[0] / 2) * rng.normal()
for t in range(1, T):
    x_true[t] = sv.phi * x_true[t - 1] + rng.normal(0.0, sv.sigma)
    y_obs[t] = sv.beta * np.exp(x_true[t] / 2) * rng.normal()

# --- Filter ---
config = PFConfig(n_particles=2000, resampling="systematic", seed=42)
bpf = BootstrapPF(model=sv, config=config)
result = bpf.filter(y_obs)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Mean ESS: {result.ess_history.mean():.0f} / {config.n_particles}")

# Compare filtered volatility with truth
filtered_vol = np.exp(result.filtered_means[:, 0] / 2)
true_vol = np.exp(x_true / 2)
rmse = np.sqrt(np.mean((filtered_vol - true_vol) ** 2))
print(f"Volatility RMSE: {rmse:.4f}")
```

---

## Tuning Guide

### Number of Particles

The most important parameter. More particles = better approximation but higher cost.

| Scenario | Recommended $N$ |
|----------|:--------------:|
| Quick exploration | 500 – 1,000 |
| Production filtering | 2,000 – 10,000 |
| Likelihood estimation (PMCMC) | 500 – 2,000 |
| High-dimensional states ($k > 5$) | 10,000+ |

!!! warning "Curse of dimensionality"
    The number of particles needed grows **exponentially** with state dimension. For $k > 10$, consider [Rao-Blackwellized PF](rbpf.md) or [Ensemble PF](ensemble.md) instead.

### Resampling Method

| Method | Variance | Determinism | Recommendation |
|--------|----------|-------------|----------------|
| `"systematic"` | Low | Quasi-deterministic | **Default choice** |
| `"stratified"` | Low | Stratified random | Good alternative |
| `"multinomial"` | Higher | Fully random | Theoretical baseline |
| `"residual"` | Low | Hybrid | Good for large $N$ |

### ESS Threshold

The `ess_threshold` parameter (default 0.5) controls when resampling is triggered:

- **Higher** (e.g., 0.8): more frequent resampling → less weight degeneracy but more path degeneracy
- **Lower** (e.g., 0.3): less frequent resampling → preserves particle diversity but risks weight collapse
- **1.0**: resample at every step (standard Bootstrap PF)

!!! tip "Rule of thumb"
    Start with `ess_threshold=0.5`. If the mean ESS is consistently low (below $0.3N$), consider switching to a filter with a better proposal, such as the [Auxiliary PF](auxiliary.md) or [SIR](sir.md) with a custom proposal.

### Computational Complexity

| Operation | Cost |
|-----------|------|
| Propagation | $O(N)$ — one transition sample per particle |
| Weighting | $O(N)$ — one likelihood evaluation per particle |
| Resampling | $O(N)$ — systematic resampling is linear |
| **Total per step** | **$O(N)$** |
| **Full filter** | **$O(N \cdot T)$** |

---

## When to Move Beyond Bootstrap

Consider upgrading to a more sophisticated filter when:

1. **Low ESS**: Mean ESS consistently below $0.3N$ → try [Auxiliary PF](auxiliary.md)
2. **Informative observations**: The observation strongly constrains the state → try [SIR](sir.md) with a tailored proposal
3. **Mixed linear/nonlinear**: Part of the state evolves linearly → try [Rao-Blackwellized PF](rbpf.md)
4. **High dimensions**: State dimension $k > 5$ → try [Ensemble PF](ensemble.md)

---

## References

- Gordon, N.J., Salmond, D.J. & Smith, A.F.M. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F*, 140(2), 107–113.
- Doucet, A., de Freitas, N. & Gordon, N. (2001). *Sequential Monte Carlo Methods in Practice*. Springer.
- Chopin, N. & Papaspiliopoulos, O. (2020). *An Introduction to Sequential Monte Carlo*. Springer.
