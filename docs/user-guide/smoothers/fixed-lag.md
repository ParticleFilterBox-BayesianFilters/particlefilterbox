---
title: "Fixed-Lag Smoother"
description: "Online particle smoother with a fixed delay for real-time smoothed state estimation"
---

# Fixed-Lag Smoother

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `FixedLagSmoother` |
    | **Import** | `from particlefilterbox.smoothers import FixedLagSmoother` |
    | **Type** | Online (processes observations sequentially) |
    | **Complexity** | $O(N \cdot L)$ per time step |
    | **Output** | Smoothed state estimates with lag $L$ |
    | **Reference** | Kitagawa & Sato (2001) |

## Overview

The Fixed-Lag Smoother provides **online smoothing** — it outputs a smoothed estimate of $x_{t-L}$ each time a new observation $y_t$ arrives. Instead of waiting for the entire series (as FFBSm, FFBSi, and Two-Filter require), it applies smoothing over a **sliding window** of $L$ time steps.

$$
p(x_{t-L} \mid y_{1:t}) \approx p(x_{t-L} \mid y_{1:T}) \quad \text{when } L \text{ is large enough}
$$

**Advantages:**

- **Online**: produces smoothed estimates in real-time with a fixed delay of $L$ steps
- Does not require storing the full observation history
- Memory bounded: $O(N \cdot L)$ regardless of series length $T$
- Can be combined with any particle filter as the underlying engine

**Disadvantages:**

- Approximation quality depends on the choice of $L$
- Larger $L$ means more computation and memory per step
- Cannot match the quality of full offline smoothers for small $L$
- Path degeneracy can accumulate over the lag window

---

## Algorithm

$$
\boxed{
\begin{aligned}
&\textbf{Fixed-Lag Smoother} \\[6pt]
&\textbf{Input: } \text{Model, } N \text{ particles, lag } L, \text{ observations } y_1, y_2, \ldots \\[4pt]
&\text{1. } \textbf{Initialize: } x_0^{(i)} \sim p(x_0), \; w_0^{(i)} = \tfrac{1}{N}, \quad i = 1, \ldots, N \\
&\quad\;\; \text{Store ancestor indices: } a_0^{(i)} = i \\[4pt]
&\text{2. } \textbf{For } t = 1, 2, \ldots : \\
&\qquad \text{a. } \textbf{Filter step:} \\
&\qquad\qquad x_t^{(i)} \sim q(x_t \mid x_{t-1}^{(i)}, y_t) \\
&\qquad\qquad \tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot \frac{p(y_t \mid x_t^{(i)}) \, p(x_t^{(i)} \mid x_{t-1}^{(i)})}{q(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t)} \\
&\qquad\qquad w_t^{(i)} = \tilde{w}_t^{(i)} \,/\, \textstyle\sum_j \tilde{w}_t^{(j)} \\[4pt]
&\qquad \text{b. } \textbf{Resample } \text{(if ESS} < \tau N\text{):} \\
&\qquad\qquad \text{Draw } a_t^{(i)} \sim \text{Categorical}(w_t^{(1)}, \ldots, w_t^{(N)}) \\
&\qquad\qquad \text{Inherit history: update ancestor paths} \\[4pt]
&\qquad \text{c. } \textbf{Output smoothed estimate } (t \geq L): \\
&\qquad\qquad \hat{x}_{t-L|t} = \sum_{i=1}^{N} w_t^{(i)} \, x_{t-L}^{(B_L^{(i)})}
\end{aligned}
}
$$

where $B_L^{(i)}$ denotes the ancestor of particle $i$ traced back $L$ steps through the resampling history.

### Key Idea: Ancestor Tracing

The smoother maintains a **genealogy** of particle indices. At time $t$, particle $i$'s ancestor at time $t - L$ is found by tracing back through the resampling indices:

$$
B_L^{(i)} = a_{t-L+1}^{(a_{t-L+2}^{(\cdots a_t^{(i)} \cdots)})}
$$

The smoothed estimate at time $t - L$ is then a weighted average using the *current* weights $w_t^{(i)}$ but the *ancestor* particles $x_{t-L}^{(B_L^{(i)})}$.

!!! note "Intuition"
    The Fixed-Lag Smoother uses the "wisdom of hindsight" from $L$ future observations to improve the estimate of $x_{t-L}$. Particles that survive $L$ rounds of resampling (i.e., have many descendants at time $t$) were more consistent with the data and receive higher effective smoothing weight.

---

## Choosing the Lag $L$

The lag $L$ controls the **bias-variance trade-off** of the smoother:

| Lag $L$ | Bias | Variance | Memory | Computation |
|---------|:----:|:--------:|:------:|:-----------:|
| Small (e.g., 5) | Higher | Lower | Low | Fast |
| Medium (e.g., 20) | Moderate | Moderate | Moderate | Moderate |
| Large (e.g., 50+) | Low | Higher (degeneracy) | High | Slow |

### Bias

The bias comes from using $p(x_{t-L} \mid y_{1:t})$ as an approximation to the full smoothing distribution $p(x_{t-L} \mid y_{1:T})$. For models with **rapidly mixing dynamics** (fast-decaying temporal correlations), even small $L$ gives good approximations:

$$
\|p(x_{t-L} \mid y_{1:t}) - p(x_{t-L} \mid y_{1:T})\| \leq C \cdot \rho^L
$$

where $\rho < 1$ is the mixing rate of the state process.

### Path Degeneracy

For large $L$, all particles may share the same ancestor at time $t - L$ (path degeneracy), making the smoothed estimate effectively a single point. This limits the useful range of $L$.

!!! tip "Rules of thumb"
    - Start with $L = 2 \times$ the effective memory of the system (e.g., for an AR(1) with $\phi = 0.9$, the memory is $\sim 1/(1-0.9) = 10$, so try $L = 20$)
    - Monitor the **number of unique ancestors** at lag $L$: if it drops below $0.1N$, reduce $L$ or increase $N$
    - For strongly persistent processes ($\phi > 0.95$), use an offline smoother instead

---

## API Reference

### Constructor

```python
from particlefilterbox.smoothers import FixedLagSmoother

smoother = FixedLagSmoother(
    model,                  # State-space model
    n_particles=1000,       # Number of particles
    lag=10,                 # Smoothing lag L
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `n_particles` | `int` | `1000` | Number of particles $N$ |
| `lag` | `int` | `10` | Smoothing lag $L$ |
| `resampling` | `str` | `"systematic"` | Resampling scheme |
| `ess_threshold` | `float` | `0.5` | ESS threshold for resampling |
| `seed` | `int \| None` | `None` | Random seed |

### Online (Step-by-Step) Usage

```python
smoother = FixedLagSmoother(model, n_particles=1000, lag=10)

for t, y_t in enumerate(observations):
    result = smoother.step(y_t)

    if result.smoothed_state is not None:  # Available after L steps
        print(f"t={t}: smoothed x[{t - smoother.lag}] = {result.smoothed_state:.3f}")
```

| Result attribute | Description |
|------------------|-------------|
| `smoothed_state` | Smoothed mean of $x_{t-L}$ (or `None` if $t < L$) |
| `smoothed_cov` | Smoothed covariance of $x_{t-L}$ |
| `filtered_state` | Current filtered mean of $x_t$ |
| `ess` | Current effective sample size |
| `n_unique_ancestors` | Number of unique ancestors at lag $L$ |
| `log_likelihood_increment` | Log-likelihood contribution $\log \hat{p}(y_t \mid y_{1:t-1})$ |

### Batch Usage

```python
result = smoother.smooth(observations)
```

| Result attribute | Shape | Description |
|------------------|-------|-------------|
| `smoothed_means` | `(T - L, k)` | Smoothed means from $t = 0$ to $t = T - L - 1$ |
| `smoothed_covs` | `(T - L, k, k)` | Smoothed covariances |
| `filtered_means` | `(T, k)` | Filtered means at all time steps |
| `ess_history` | `(T,)` | ESS at each step |
| `unique_ancestors` | `(T,)` | Unique ancestor count at lag $L$ |
| `log_likelihood` | scalar | Total log-marginal likelihood |

---

## Examples

### Example 1: Real-Time Tracking with Smoothing

Track a target moving in 1D with noisy position observations, using fixed-lag smoothing for improved estimates.

```python
import numpy as np
from particlefilterbox.smoothers import FixedLagSmoother
from particlefilterbox.core.model import ParticleFilterModel

class TrackingModel(ParticleFilterModel):
    """
    x_t = [position_t, velocity_t]
    position_t = position_{t-1} + velocity_{t-1} * dt + eta_pos
    velocity_t = velocity_{t-1} + eta_vel
    y_t = position_t + eps_t
    """
    k_states = 2
    k_obs = 1

    def __init__(self, dt=1.0, sigma_pos=0.1, sigma_vel=0.5, sigma_obs=1.0):
        self.dt = dt
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel
        self.sigma_obs = sigma_obs

    def initial_distribution(self, n_particles, rng):
        particles = np.zeros((n_particles, 2))
        particles[:, 0] = rng.normal(0.0, 1.0, size=n_particles)  # position
        particles[:, 1] = rng.normal(0.0, 0.5, size=n_particles)  # velocity
        return particles

    def transition(self, particles, t, rng):
        new = np.zeros_like(particles)
        new[:, 0] = (
            particles[:, 0]
            + particles[:, 1] * self.dt
            + rng.normal(0.0, self.sigma_pos, size=len(particles))
        )
        new[:, 1] = particles[:, 1] + rng.normal(
            0.0, self.sigma_vel, size=len(particles)
        )
        return new

    def log_observation_likelihood(self, particles, y_t, t):
        residual = y_t[0] - particles[:, 0]
        return -0.5 * (residual / self.sigma_obs) ** 2

# --- Simulate ---
model = TrackingModel(dt=0.1, sigma_pos=0.05, sigma_vel=0.2, sigma_obs=0.5)
rng = np.random.default_rng(42)
T = 500

x_true = np.zeros((T, 2))
y_obs = np.zeros(T)
x_true[0] = [0.0, 1.0]
y_obs[0] = x_true[0, 0] + rng.normal(0.0, 0.5)

for t in range(1, T):
    x_true[t, 0] = (
        x_true[t - 1, 0]
        + x_true[t - 1, 1] * 0.1
        + rng.normal(0.0, 0.05)
    )
    x_true[t, 1] = x_true[t - 1, 1] + rng.normal(0.0, 0.2)
    y_obs[t] = x_true[t, 0] + rng.normal(0.0, 0.5)

# --- Online smoothing ---
smoother = FixedLagSmoother(model=model, n_particles=2000, lag=15, seed=42)

smoothed_positions = []
filtered_positions = []

for t, y_t in enumerate(y_obs):
    result = smoother.step(np.array([y_t]))
    filtered_positions.append(result.filtered_state[0])

    if result.smoothed_state is not None:
        smoothed_positions.append(result.smoothed_state[0])

# --- Compare ---
L = smoother.lag
rmse_filter = np.sqrt(np.mean(
    (np.array(filtered_positions[:T - L]) - x_true[:T - L, 0]) ** 2
))
rmse_smooth = np.sqrt(np.mean(
    (np.array(smoothed_positions) - x_true[:T - L, 0]) ** 2
))

print(f"Filtered RMSE: {rmse_filter:.4f}")
print(f"Smoothed RMSE: {rmse_smooth:.4f}")
print(f"Lag L = {L}")
```

### Example 2: Monitoring Lag Diagnostics

Monitor the number of unique ancestors to detect path degeneracy:

```python
smoother = FixedLagSmoother(model=model, n_particles=1000, lag=20, seed=42)

for t, y_t in enumerate(y_obs):
    result = smoother.step(np.array([y_t]))

    if t >= smoother.lag and t % 50 == 0:
        ratio = result.n_unique_ancestors / smoother.n_particles
        status = "OK" if ratio > 0.1 else "DEGENERACY"
        print(
            f"t={t}: unique ancestors = {result.n_unique_ancestors}"
            f" / {smoother.n_particles} ({ratio:.1%}) [{status}]"
        )
```

!!! warning "Path degeneracy"
    If `n_unique_ancestors` drops below $0.1N$ consistently, the lag $L$ is too large for the given $N$. Either reduce $L$ or increase `n_particles`.

### Example 3: Comparing Different Lag Values

```python
lags = [5, 10, 20, 50]
results = {}

for L in lags:
    smoother = FixedLagSmoother(model=model, n_particles=2000, lag=L, seed=42)
    batch_result = smoother.smooth(y_obs)
    rmse = np.sqrt(np.mean(
        (batch_result.smoothed_means[:, 0] - x_true[:T - L, 0]) ** 2
    ))
    results[L] = rmse
    print(f"Lag {L:3d}: RMSE = {rmse:.4f}")

# Typically: RMSE decreases with L up to a point, then degeneracy hurts
```

---

## Comparison with Offline Smoothers

| Aspect | [FFBSm](ffbsm.md) / [FFBSi](ffbsi.md) | Fixed-Lag (this page) |
|--------|:---------------------------------------:|:---------------------:|
| **Mode** | Offline (batch) | Online (streaming) |
| **Requires all data?** | Yes | No |
| **Latency** | $T$ steps | $L$ steps |
| **Memory** | $O(N \cdot T)$ | $O(N \cdot L)$ |
| **Smoothing quality** | Optimal | Approximate (bias $\propto \rho^L$) |
| **Use case** | Retrospective analysis | Real-time applications |

!!! tip "When to choose Fixed-Lag"
    - Data arrives in real-time and you cannot wait for the full series
    - The series is very long or potentially infinite (streaming data)
    - A small delay of $L$ steps is acceptable
    - Memory is constrained and you cannot store $O(N \cdot T)$ particles

---

## References

- Kitagawa, G. & Sato, S. (2001). Monte Carlo Smoothing and Self-Organising State-Space Model. In *Sequential Monte Carlo Methods in Practice*, Springer, 177–195.
- Olsson, J., Cappé, O., Douc, R. & Moulines, E. (2008). Sequential Monte Carlo Smoothing with Application to Parameter Estimation in Nonlinear State Space Models. *Bernoulli*, 14(1), 155–179.
- Doucet, A. & Johansen, A.M. (2009). A Tutorial on Particle Filtering and Smoothing: Fifteen Years Later. *Handbook of Nonlinear Filtering*, 12, 656–704.
