---
title: "FFBSi — Forward-Filtering Backward-Simulation"
description: "Particle smoother that generates full smoothed trajectories via backward simulation"
---

# Forward-Filtering Backward-Simulation

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `FFBSi` |
    | **Import** | `from particlefilterbox.smoothers import FFBSi` |
    | **Type** | Offline (requires full filter result) |
    | **Complexity** | $O(NM)$ per time step ($M$ trajectories, $N$ particles) |
    | **Output** | $M$ complete smoothed trajectories |
    | **Reference** | Godsill, Doucet & West (2004) |

## Overview

The Forward-Filtering Backward-Simulation algorithm — **FFBSi** — generates **complete smoothed trajectories** $x_{0:T}^{(m)} \sim p(x_{0:T} \mid y_{1:T})$ by running a forward particle filter followed by a backward simulation pass. Unlike [FFBSm](ffbsm.md) which only reweights existing particles, FFBSi constructs new paths by tracing backward through the filter history.

**Advantages:**

- Produces full joint trajectories $x_{0:T}$, not just marginals
- Essential for path-dependent functionals $\mathbb{E}[f(x_{0:T}) \mid y_{1:T}]$
- Each trajectory is an independent sample from the smoothing distribution
- Complexity $O(NM)$ can be better than FFBSm's $O(N^2)$ when $M \ll N$

**Disadvantages:**

- Requires storing the full particle history from the forward pass
- Cost scales linearly with the number of trajectories $M$
- Backward simulation can suffer from path degeneracy for very long series

---

## Algorithm

### Forward Pass

Run any particle filter to obtain, at each time $t = 0, \ldots, T$:

- Particles: $\{x_t^{(i)}\}_{i=1}^{N}$
- Filtering weights: $\{w_t^{(i)}\}_{i=1}^{N}$

### Backward Simulation

For each trajectory $m = 1, \ldots, M$:

$$
\boxed{
\begin{aligned}
&\textbf{FFBSi — Backward Simulation} \\[6pt]
&\text{For } m = 1, \ldots, M: \\[4pt]
&\qquad \text{1. } \textbf{Initialize: } \text{Draw } \tilde{x}_T^{(m)} \text{ from } \{x_T^{(i)}\} \text{ with weights } \{w_T^{(i)}\} \\[4pt]
&\qquad \text{2. } \textbf{For } t = T-1, \ldots, 0: \\
&\qquad\qquad \text{a. Compute backward weights:} \\
&\qquad\qquad\qquad \tilde{w}_{t|t+1}^{(i)} = w_t^{(i)} \cdot p(\tilde{x}_{t+1}^{(m)} \mid x_t^{(i)}), \quad i = 1, \ldots, N \\[4pt]
&\qquad\qquad \text{b. Normalize:} \\
&\qquad\qquad\qquad w_{t|t+1}^{(i)} = \frac{\tilde{w}_{t|t+1}^{(i)}}{\sum_{j=1}^{N} \tilde{w}_{t|t+1}^{(j)}} \\[4pt]
&\qquad\qquad \text{c. Draw: } \tilde{x}_t^{(m)} \sim \text{Categorical}\left(\{x_t^{(i)}\}, \{w_{t|t+1}^{(i)}\}\right)
\end{aligned}
}
$$

### Why Does This Work?

The backward weight $w_{t|t+1}^{(i)} \propto w_t^{(i)} \cdot p(\tilde{x}_{t+1}^{(m)} \mid x_t^{(i)})$ combines two pieces of information:

1. **$w_t^{(i)}$**: How likely particle $i$ is under the filtering distribution $p(x_t \mid y_{1:t})$
2. **$p(\tilde{x}_{t+1}^{(m)} \mid x_t^{(i)})$**: How well particle $i$ connects to the already-chosen next state

This product ensures that the drawn trajectory is consistent both with the observations (via filtering weights) and with the model dynamics (via the transition density).

!!! note "Intuition"
    Think of it as building a path from the end backward. At each step, you pick the most plausible ancestor — the one that the filter liked *and* that leads naturally to where you already are at $t+1$.

---

## Complexity Analysis

| Component | Cost | Notes |
|-----------|------|-------|
| Forward filter | $O(N \cdot T)$ | Any standard particle filter |
| Backward simulation | $O(N \cdot M \cdot T)$ | For each trajectory, evaluate $N$ backward weights at each step |
| **Total** | **$O(N(M + 1) \cdot T)$** | |

### When Is FFBSi Cheaper Than FFBSm?

- **FFBSm** costs $O(N^2 \cdot T)$
- **FFBSi** costs $O(NM \cdot T)$

FFBSi is cheaper when $M < N$, which is typical: you often need $M = 50\text{–}500$ trajectories but run $N = 1000\text{–}10000$ particles.

| $N$ | $M$ | FFBSm cost | FFBSi cost | Winner |
|-----|-----|:----------:|:----------:|:------:|
| 1000 | 100 | $10^6$ | $10^5$ | FFBSi |
| 1000 | 1000 | $10^6$ | $10^6$ | Tie |
| 5000 | 100 | $2.5 \times 10^7$ | $5 \times 10^5$ | FFBSi |
| 5000 | 5000 | $2.5 \times 10^7$ | $2.5 \times 10^7$ | Tie |

---

## API Reference

### Constructor

```python
from particlefilterbox.smoothers import FFBSi

smoother = FFBSi(
    filter_result,           # Output from any particle filter
    n_trajectories=100,      # Number of backward trajectories M
    seed=None,               # Random seed for backward simulation
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_result` | `FilterResult` | *required* | Output from a forward filter pass |
| `n_trajectories` | `int` | `100` | Number of smoothed trajectories $M$ |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

### Smoothing

```python
result = smoother.smooth()
```

| Result attribute | Shape | Description |
|------------------|-------|-------------|
| `paths` | `(M, T, k)` | Smoothed trajectories |
| `smoothed_means` | `(T, k)` | Mean across trajectories at each time step |
| `smoothed_covs` | `(T, k, k)` | Covariance across trajectories |
| `log_likelihood` | scalar | Log-marginal likelihood (from forward filter) |

---

## Examples

### Example 1: Latent State Trajectory Estimation

Generate smoothed trajectories for a nonlinear state-space model and compare with the filtered estimate.

```python
import numpy as np
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.smoothers import FFBSi
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class NonlinearModel(ParticleFilterModel):
    """
    x_t = 0.5 * x_{t-1} + 25 * x_{t-1} / (1 + x_{t-1}^2) + 8 * cos(1.2*t) + eta_t
    y_t = x_t^2 / 20 + eps_t
    Classic nonlinear benchmark (Kitagawa, 1996).
    """
    k_states = 1
    k_obs = 1

    def initial_distribution(self, n_particles, rng):
        return rng.normal(0.0, np.sqrt(5.0), size=(n_particles, 1))

    def transition(self, particles, t, rng):
        x = particles[:, 0]
        mean = 0.5 * x + 25.0 * x / (1.0 + x**2) + 8.0 * np.cos(1.2 * t)
        return mean[:, None] + rng.normal(0.0, np.sqrt(10.0), size=particles.shape)

    def log_observation_likelihood(self, particles, y_t, t):
        predicted = particles[:, 0] ** 2 / 20.0
        return -0.5 * (y_t[0] - predicted) ** 2

    def transition_log_density(self, x_next, x_curr, t):
        mean = 0.5 * x_curr + 25.0 * x_curr / (1.0 + x_curr**2) + 8.0 * np.cos(1.2 * t)
        return -0.5 * (x_next - mean) ** 2 / 10.0

# --- Simulate ---
rng = np.random.default_rng(123)
T = 100
x_true = np.zeros(T)
y_obs = np.zeros(T)

x_true[0] = rng.normal(0.0, np.sqrt(5.0))
y_obs[0] = x_true[0] ** 2 / 20.0 + rng.normal()
for t in range(1, T):
    x_true[t] = (
        0.5 * x_true[t - 1]
        + 25.0 * x_true[t - 1] / (1.0 + x_true[t - 1] ** 2)
        + 8.0 * np.cos(1.2 * t)
        + rng.normal(0.0, np.sqrt(10.0))
    )
    y_obs[t] = x_true[t] ** 2 / 20.0 + rng.normal()

# --- Forward filter ---
config = PFConfig(n_particles=5000, resampling="systematic", seed=42)
pf = BootstrapPF(model=NonlinearModel(), config=config)
filter_result = pf.filter(y_obs)

# --- Backward simulation ---
smoother = FFBSi(filter_result, n_trajectories=200, seed=42)
result = smoother.smooth()

# --- Compare ---
print(f"Trajectories shape: {result.paths.shape}")  # (200, 100, 1)

rmse_filter = np.sqrt(np.mean((filter_result.filtered_means[:, 0] - x_true) ** 2))
rmse_smooth = np.sqrt(np.mean((result.smoothed_means[:, 0] - x_true) ** 2))
print(f"Filtered RMSE: {rmse_filter:.4f}")
print(f"Smoothed RMSE: {rmse_smooth:.4f}")
```

### Example 2: Path-Dependent Functional

Compute a path-dependent expectation: the total absolute change in the latent state.

```python
# Using the smoother result from Example 1
trajectories = result.paths[:, :, 0]  # (M, T)

# Path-dependent functional: total absolute change
total_change = np.sum(np.abs(np.diff(trajectories, axis=1)), axis=1)  # (M,)

print(f"E[total |dx|] = {total_change.mean():.2f} ± {total_change.std():.2f}")

# This cannot be computed with FFBSm, which only gives marginals!
```

### Example 3: Smoothing for EM Parameter Estimation

FFBSi is commonly used in the E-step of an EM algorithm to compute sufficient statistics.

```python
# E-step: generate smoothed trajectories
smoother = FFBSi(filter_result, n_trajectories=100, seed=42)
result = smoother.smooth()

# Sufficient statistics for AR(1) model: E[x_t * x_{t-1}] and E[x_t^2]
paths = result.paths[:, :, 0]  # (M, T)

# Cross-moment: E[x_t * x_{t-1} | y_{1:T}]
cross_moment = np.mean(paths[:, 1:] * paths[:, :-1])

# Second moment: E[x_t^2 | y_{1:T}]
second_moment = np.mean(paths[:, 1:] ** 2)

# M-step: update phi = E[x_t * x_{t-1}] / E[x_{t-1}^2]
phi_hat = cross_moment / np.mean(paths[:, :-1] ** 2)
print(f"Estimated phi: {phi_hat:.4f}")
```

---

## Comparison with FFBSm

| Aspect | [FFBSm](ffbsm.md) | FFBSi (this page) |
|--------|:------------------:|:------------------:|
| **Output** | Smoothing weights | Full trajectories |
| **New particles?** | No (reweights) | Yes (backward draws) |
| **Complexity** | $O(N^2 \cdot T)$ | $O(NM \cdot T)$ |
| **Marginal estimates** | Direct | Via trajectory averaging |
| **Path functionals** | :material-close: Cannot compute | :material-check: Full support |
| **EM sufficient stats** | Limited | :material-check: Full support |
| **Memory** | Filter history | Filter history + trajectories |

!!! tip "When to choose FFBSi"
    - You need full trajectories $x_{0:T}^{(m)}$ for downstream analysis
    - You need path-dependent expectations like $\mathbb{E}[\sum_t f(x_t, x_{t+1}) \mid y_{1:T}]$
    - You are running an EM algorithm and need sufficient statistics across time steps
    - $M \ll N$, making FFBSi cheaper than FFBSm

---

## References

- Godsill, S.J., Doucet, A. & West, M. (2004). Monte Carlo Smoothing for Nonlinear Time Series. *Journal of the American Statistical Association*, 99(465), 156–168.
- Lindsten, F. & Schön, T.B. (2013). Backward Simulation Methods for Monte Carlo Statistical Inference. *Foundations and Trends in Machine Learning*, 6(1), 1–143.
- Douc, R., Garivier, A., Moulines, E. & Olsson, J. (2011). Sequential Monte Carlo Smoothing for General State Space Hidden Markov Models. *Annals of Applied Probability*, 21(6), 2226–2252.
