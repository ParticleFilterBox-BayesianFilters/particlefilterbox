---
title: "FFBSm — Forward-Filtering Backward-Smoothing (Marginal)"
description: "Marginal particle smoother that reweights filter particles using backward smoothing weights"
---

# Forward-Filtering Backward-Smoothing (Marginal)

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `FFBSm` |
    | **Import** | `from particlefilterbox.smoothers import FFBSm` |
    | **Type** | Offline (requires full filter result) |
    | **Complexity** | $O(N^2)$ per time step (reducible to $O(N \log N)$) |
    | **Output** | Smoothing weights for existing particles |
    | **Reference** | Doucet, Godsill & Andrieu (2000) |

## Overview

The Forward-Filtering Backward-Smoothing (Marginal) algorithm — **FFBSm** — is the most straightforward particle smoother. It takes the output of any particle filter and computes **smoothing weights** through a backward recursion. No new particles are generated; instead, the existing filter particles are reweighted to approximate the smoothing distribution $p(x_t \mid y_{1:T})$.

**Advantages:**

- Reuses particles from the forward filter — no new sampling required
- Provides smoothed marginals at every time step
- Compatible with any particle filter (Bootstrap, APF, RBPF, etc.)

**Disadvantages:**

- $O(N^2)$ cost per time step in the backward pass
- Does not produce full smoothed trajectories (only marginals)
- Smoothing weights can degenerate for long time series

---

## Algorithm

The FFBSm algorithm operates in two passes:

### Forward Pass

Run any particle filter to obtain, at each time $t = 1, \ldots, T$:

- Particles: $\{x_t^{(i)}\}_{i=1}^{N}$
- Filtering weights: $\{w_t^{(i)}\}_{i=1}^{N}$

### Backward Pass

Starting from $t = T$ and working backward to $t = 1$:

$$
\boxed{
\begin{aligned}
&\textbf{FFBSm — Backward Smoothing} \\[6pt]
&\text{1. } \textbf{Initialize: } w_{T|T}^{(i)} = w_T^{(i)}, \quad i = 1, \ldots, N \\[4pt]
&\text{2. } \textbf{For } t = T-1, \ldots, 1: \\
&\qquad \text{For } i = 1, \ldots, N: \\[4pt]
&\qquad\qquad w_{t|T}^{(i)} = w_t^{(i)} \sum_{j=1}^{N} \frac{w_{t+1|T}^{(j)} \, p(x_{t+1}^{(j)} \mid x_t^{(i)})}{\sum_{k=1}^{N} w_t^{(k)} \, p(x_{t+1}^{(j)} \mid x_t^{(k)})}
\end{aligned}
}
$$

### Smoothing Weight Derivation

The key insight is Bayes' rule applied to the joint smoothing distribution. The smoothing weight for particle $i$ at time $t$ is:

$$
w_{t|T}^{(i)} \propto w_t^{(i)} \sum_{j=1}^{N} \frac{w_{t+1|T}^{(j)} \, p(x_{t+1}^{(j)} \mid x_t^{(i)})}{\sum_{k=1}^{N} w_t^{(k)} \, p(x_{t+1}^{(j)} \mid x_t^{(k)})}
$$

This expression couples every particle at time $t$ with every particle at time $t+1$, producing the $O(N^2)$ cost.

!!! note "Intuition"
    Each filter particle receives additional weight based on how well it "explains" the smoothed particles at the next time step. Particles that are consistent with the future trajectory get upweighted; those that lead to unlikely futures get downweighted.

---

## Complexity Analysis

| Component | Cost | Notes |
|-----------|------|-------|
| Forward filter | $O(N \cdot T)$ | Any standard particle filter |
| Backward pass (naive) | $O(N^2 \cdot T)$ | Double loop over particles at $t$ and $t+1$ |
| **Total (naive)** | **$O(N^2 \cdot T)$** | Dominates for large $N$ |
| Backward pass (reduced) | $O(N \log N \cdot T)$ | With rejection sampling or kd-tree |

### Complexity Reduction Techniques

The $O(N^2)$ cost can be prohibitive for large particle counts. Two main strategies reduce this:

#### Rejection Sampling

Instead of computing the full sum over all particles at $t$, use rejection sampling to draw backward indices:

$$
\boxed{
\begin{aligned}
&\textbf{Rejection-Based FFBSm} \\[6pt]
&\text{For each } j = 1, \ldots, N \text{ at time } t+1: \\
&\qquad \text{1. Draw } i \sim \text{Categorical}(w_t^{(1)}, \ldots, w_t^{(N)}) \\
&\qquad \text{2. Accept with probability } \frac{p(x_{t+1}^{(j)} \mid x_t^{(i)})}{\sup_x p(x_{t+1}^{(j)} \mid x)} \\
&\qquad \text{3. If rejected, return to step 1}
\end{aligned}
}
$$

Expected cost: $O(N)$ when the transition density is not too peaked. Falls back to $O(N^2)$ in the worst case.

```python
smoother = FFBSm(filter_result, method="rejection")
smoothed = smoother.smooth()
```

#### KD-Tree Approximation

Build a kd-tree over the particles at time $t$ and only evaluate the transition density for the $K$ nearest neighbors:

```python
smoother = FFBSm(filter_result, method="kd_tree", n_neighbors=50)
smoothed = smoother.smooth()
```

Cost: $O(N \log N \cdot T)$ for tree construction + $O(NK \cdot T)$ for lookups.

!!! tip "When to use reduction"
    For $N < 2000$, the naive $O(N^2)$ is usually fast enough. Enable reduction for $N > 5000$ or when profiling shows the backward pass is the bottleneck.

---

## API Reference

### Constructor

```python
from particlefilterbox.smoothers import FFBSm

smoother = FFBSm(
    filter_result,          # Output from any particle filter
    method="naive",         # "naive", "rejection", or "kd_tree"
    n_neighbors=50,         # For kd_tree method
    max_rejection=100,      # Max rejection attempts before fallback
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_result` | `FilterResult` | *required* | Output from a forward filter pass |
| `method` | `str` | `"naive"` | Backward computation method: `"naive"`, `"rejection"`, `"kd_tree"` |
| `n_neighbors` | `int` | `50` | Number of neighbors for kd-tree method |
| `max_rejection` | `int` | `100` | Maximum rejection sampling attempts |

### Smoothing

```python
smoothed = smoother.smooth()
```

| Result attribute | Shape | Description |
|------------------|-------|-------------|
| `smoothed_means` | `(T, k)` | Smoothed state means |
| `smoothed_covs` | `(T, k, k)` | Smoothed state covariances |
| `smoothing_weights` | `(T, N)` | Smoothing weights at each time step |
| `log_likelihood` | scalar | Log-marginal likelihood (from forward filter) |

---

## Examples

### Example 1: Smoothing a Stochastic Volatility Model

This example shows how smoothing recovers the latent log-volatility more accurately than filtering alone.

```python
import numpy as np
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.smoothers import FFBSm
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
        return self.phi * particles + rng.normal(
            0.0, self.sigma, size=particles.shape
        )

    def log_observation_likelihood(self, particles, y_t, t):
        vol = self.beta * np.exp(particles[:, 0] / 2)
        return -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (y_t[0] / vol) ** 2

    def transition_log_density(self, x_next, x_curr, t):
        """Required by FFBSm for backward pass."""
        residual = x_next - self.phi * x_curr
        return -0.5 * (residual / self.sigma) ** 2 - np.log(self.sigma)

# --- Simulate data ---
sv = StochasticVolatility(phi=0.98, sigma=0.16, beta=0.65)
rng = np.random.default_rng(42)
T = 300

x_true = np.zeros(T)
y_obs = np.zeros(T)
std_0 = sv.sigma / np.sqrt(1 - sv.phi**2)
x_true[0] = rng.normal(0.0, std_0)
y_obs[0] = sv.beta * np.exp(x_true[0] / 2) * rng.normal()
for t in range(1, T):
    x_true[t] = sv.phi * x_true[t - 1] + rng.normal(0.0, sv.sigma)
    y_obs[t] = sv.beta * np.exp(x_true[t] / 2) * rng.normal()

# --- Forward filter ---
config = PFConfig(n_particles=2000, resampling="systematic", seed=42)
pf = BootstrapPF(model=sv, config=config)
filter_result = pf.filter(y_obs)

# --- Backward smoothing ---
smoother = FFBSm(filter_result)
smoothed = smoother.smooth()

# --- Compare RMSE ---
rmse_filter = np.sqrt(np.mean((filter_result.filtered_means[:, 0] - x_true) ** 2))
rmse_smooth = np.sqrt(np.mean((smoothed.smoothed_means[:, 0] - x_true) ** 2))

print(f"Filtered RMSE: {rmse_filter:.4f}")
print(f"Smoothed RMSE: {rmse_smooth:.4f}")
print(f"Improvement:   {(1 - rmse_smooth / rmse_filter) * 100:.1f}%")
```

!!! tip "What to expect"
    Smoothing typically reduces RMSE by 10–30% compared to filtering, with the largest gains at the beginning and end of the series where the filter has the least information.

### Example 2: Using Complexity Reduction

For large particle counts, use rejection sampling to speed up the backward pass:

```python
# With 10,000 particles, naive O(N^2) is slow
config = PFConfig(n_particles=10000, resampling="systematic", seed=42)
pf = BootstrapPF(model=sv, config=config)
filter_result = pf.filter(y_obs)

# Rejection-based backward pass — typically O(N) per step
smoother = FFBSm(filter_result, method="rejection")
smoothed = smoother.smooth()

# Or kd-tree approximation
smoother_kd = FFBSm(filter_result, method="kd_tree", n_neighbors=100)
smoothed_kd = smoother_kd.smooth()
```

---

## Comparison with FFBSi

| Aspect | FFBSm (this page) | [FFBSi](ffbsi.md) |
|--------|:------------------:|:------------------:|
| **Output** | Smoothing weights | Full trajectories |
| **New particles?** | No (reweights) | Yes (backward simulation) |
| **Complexity** | $O(N^2)$ per step | $O(NM)$ per step |
| **Best for** | Marginal expectations | Path-dependent functionals |
| **Weight degeneracy** | Can accumulate | Fresh trajectories avoid it |

!!! tip "Rule of thumb"
    Use **FFBSm** when you only need $\mathbb{E}[f(x_t) \mid y_{1:T}]$ for each $t$. Use **FFBSi** when you need the full joint distribution $p(x_{0:T} \mid y_{1:T})$ or path-dependent quantities like $\mathbb{E}[\sum_t g(x_t, x_{t+1}) \mid y_{1:T}]$.

---

## References

- Doucet, A., Godsill, S.J. & Andrieu, C. (2000). On Sequential Monte Carlo Sampling Methods for Bayesian Filtering. *Statistics and Computing*, 10(3), 197–208.
- Kitagawa, G. (1996). Monte Carlo Filter and Smoother for Non-Gaussian Nonlinear State Space Models. *Journal of Computational and Graphical Statistics*, 5(1), 1–25.
- Lindsten, F. & Schön, T.B. (2013). Backward Simulation Methods for Monte Carlo Statistical Inference. *Foundations and Trends in Machine Learning*, 6(1), 1–143.
