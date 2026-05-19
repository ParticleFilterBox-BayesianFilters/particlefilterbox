---
title: Ensemble Particle Filter
description: "The Ensemble PF — combining particle methods with ensemble Kalman updates for high-dimensional states"
---

# Ensemble Particle Filter

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `EnsemblePF` |
    | **Import** | `from particlefilterbox.filters import EnsemblePF` |
    | **Key idea** | Combine particle propagation with ensemble Kalman-style updates |
    | **Complexity** | $O(N \cdot k)$ per time step ($k$ = state dimension) |
    | **Reference** | Frei & Künsch (2013) |

## Overview

The Ensemble Particle Filter (EnPF) bridges the gap between **particle filters** and the **Ensemble Kalman Filter (EnKF)**. Standard particle filters suffer from the curse of dimensionality — the number of particles needed grows exponentially with state dimension. The EnKF handles high dimensions well but assumes Gaussian updates, which can be inaccurate for nonlinear, non-Gaussian models.

The EnPF combines the best of both worlds: it uses an **ensemble Kalman update** to shift particles toward the observation, then applies **importance weighting** to correct for the Gaussian approximation. This yields an algorithm that scales to high-dimensional states while retaining the asymptotic correctness of particle methods.

**Advantages:**

- Scales to high-dimensional states ($k > 50$) where standard PFs fail
- Ensemble update provides implicit localization of information
- Robust to moderate non-Gaussianity
- Widely used in geophysics, meteorology, and oceanography

**Disadvantages:**

- Ensemble update assumes approximately Gaussian posterior — poor for strongly multimodal targets
- Requires tuning of inflation and localization parameters
- Correction weights can be degenerate if the Gaussian approximation is poor

---

## Algorithm

$$
\boxed{
\begin{aligned}
&\textbf{Ensemble Particle Filter} \\[6pt]
&\text{1. } \textbf{Initialize: } x_0^{(i)} \sim p(x_0), \quad w_0^{(i)} = \tfrac{1}{N}, \quad i = 1, \ldots, N \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T: \\
&\qquad \text{a. } \textbf{Propagate: } \hat{x}_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)}) \\
&\qquad \text{b. } \textbf{Ensemble mean: } \bar{x}_t = \frac{1}{N} \sum_{i=1}^{N} \hat{x}_t^{(i)} \\
&\qquad \text{c. } \textbf{Anomalies: } A_t = \frac{1}{\sqrt{N-1}} [\hat{x}_t^{(1)} - \bar{x}_t, \ldots, \hat{x}_t^{(N)} - \bar{x}_t] \\
&\qquad \text{d. } \textbf{Ensemble covariance: } P_t^f = A_t A_t^\top \\
&\qquad \text{e. } \textbf{Kalman gain: } K_t = P_t^f H^\top (H P_t^f H^\top + R)^{-1} \\
&\qquad \text{f. } \textbf{Ensemble update: } x_t^{(i)} = \hat{x}_t^{(i)} + K_t (y_t + \varepsilon_t^{(i)} - H \hat{x}_t^{(i)}), \quad \varepsilon_t^{(i)} \sim \mathcal{N}(0, R) \\
&\qquad \text{g. } \textbf{Weight: } \tilde{w}_t^{(i)} = \frac{p(y_t \mid x_t^{(i)})}{g_{\text{EnKF}}(x_t^{(i)} \mid \hat{x}_t^{(i)}, y_t)} \\
&\qquad \text{h. } \textbf{Normalize: } w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_j \tilde{w}_t^{(j)}} \\
&\qquad \text{i. } \textbf{Resample: } \text{If } \text{ESS} < \tau \cdot N, \text{ resample}
\end{aligned}
}
$$

### Ensemble Update as Proposal

The ensemble Kalman update in step (f) acts as an **implicit proposal distribution**. It shifts each particle toward the observation using the ensemble-estimated Kalman gain, producing an analysis ensemble that is closer to the posterior than the raw forecast.

The perturbed observations $y_t + \varepsilon_t^{(i)}$ ensure that the analysis ensemble has the correct spread (Burgers et al., 1998).

### Correction Weights

The weights in step (g) correct for the fact that the ensemble Kalman update is only exact for linear Gaussian models. The correction ensures that the weighted ensemble converges to the true posterior as $N \to \infty$.

### Localization

For very high-dimensional states, the ensemble covariance $P_t^f$ estimated from a small ensemble is noisy and has spurious long-range correlations. **Localization** damps these correlations:

$$
P_t^{\text{loc}} = \rho \circ P_t^f
$$

where $\rho$ is a tapering matrix (e.g., Gaspari-Cohn function) and $\circ$ denotes element-wise multiplication.

### Inflation

Ensemble methods tend to **underestimate** the posterior spread due to sampling errors. **Inflation** counteracts this by multiplying the anomalies by a factor $\delta > 1$:

$$
A_t^{\text{inf}} = \delta \cdot A_t
$$

Typical inflation factors range from $\delta = 1.01$ to $\delta = 1.10$.

---

## API Reference

### Constructor

```python
from particlefilterbox.filters import EnsemblePF
from particlefilterbox.core.config import PFConfig

config = PFConfig(
    n_particles=100,  # ensemble size
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
)

enpf = EnsemblePF(model=my_model, config=config)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `100` | Ensemble size $N$ |
| `resampling` | `str` | `"systematic"` | Resampling scheme |
| `ess_threshold` | `float` | `0.5` | Resample when $\text{ESS} < \tau \cdot N$ |
| `seed` | `int \| None` | `None` | Random seed |
| `inflation` | `float` | `1.0` | Multiplicative inflation factor $\delta$ |
| `localization` | `str \| None` | `None` | Localization method: `"gaspari_cohn"`, `"boxcar"`, or `None` |
| `localization_radius` | `float` | `None` | Cut-off radius for localization (in state-space units) |

### Model Requirements

The EnsemblePF model must provide:

| Model method | Signature | Purpose |
|-------------|-----------|---------|
| `initial_distribution` | `(n_particles, rng) → ndarray` | Sample $x_0$ |
| `transition` | `(particles, t, rng) → ndarray` | Sample $x_t \mid x_{t-1}$ |
| `log_observation_likelihood` | `(particles, y_t, t) → ndarray` | Evaluate $\log p(y_t \mid x_t)$ |
| `observation_matrix` | `(t) → ndarray` | Linear observation matrix $H$ (or linearized) |
| `observation_noise_cov` | `(t) → ndarray` | Observation noise covariance $R$ |

### Batch Filtering

```python
result = enpf.filter(observations)
```

Returns the same `ParticleFilterResults` as Bootstrap PF. See [Bootstrap PF — Batch Filtering](bootstrap.md#batch-filtering).

---

## Examples

### Example 1: High-Dimensional Linear State

A 20-dimensional state where standard particle filters would require millions of particles but the EnPF works with a modest ensemble.

```python
import numpy as np
from particlefilterbox.filters import EnsemblePF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class HighDimLinear(ParticleFilterModel):
    """
    x_t = A * x_{t-1} + eta_t,   eta_t ~ N(0, Q)
    y_t = H * x_t + eps_t,       eps_t ~ N(0, R)

    A = 0.95 * I_k  (stable AR(1) in each dimension)
    H = I_k[:m, :]  (observe first m dimensions)
    """
    def __init__(self, k=20, m=10, sigma_q=0.1, sigma_r=0.5):
        self.k_states = k
        self.k_obs = m
        self.A = 0.95 * np.eye(k)
        self.H = np.eye(m, k)
        self.Q = sigma_q**2 * np.eye(k)
        self.R = sigma_r**2 * np.eye(m)
        self.sigma_q = sigma_q
        self.sigma_r = sigma_r

    def initial_distribution(self, n_particles, rng):
        return rng.normal(0.0, 1.0, size=(n_particles, self.k_states))

    def transition(self, particles, t, rng):
        return particles @ self.A.T + rng.normal(
            0.0, self.sigma_q, size=particles.shape
        )

    def log_observation_likelihood(self, particles, y_t, t):
        pred = particles @ self.H.T
        diff = y_t - pred
        return -0.5 * np.sum(diff**2 / self.sigma_r**2, axis=1)

    def observation_matrix(self, t):
        return self.H

    def observation_noise_cov(self, t):
        return self.R

# --- Simulate ---
k, m = 20, 10
model = HighDimLinear(k=k, m=m)
rng = np.random.default_rng(42)
T = 200

x_true = np.zeros((T, k))
y_obs = np.zeros((T, m))

x_true[0] = rng.normal(0, 1, size=k)
y_obs[0] = model.H @ x_true[0] + rng.normal(0, model.sigma_r, size=m)
for t in range(1, T):
    x_true[t] = model.A @ x_true[t-1] + rng.normal(0, model.sigma_q, size=k)
    y_obs[t] = model.H @ x_true[t] + rng.normal(0, model.sigma_r, size=m)

# --- EnPF ---
config = PFConfig(n_particles=100, resampling="systematic", seed=42)
enpf = EnsemblePF(model=model, config=config, inflation=1.02)
result = enpf.filter(y_obs)

rmse = np.sqrt(np.mean((result.filtered_means - x_true)**2))
print(f"State dimension: {k}")
print(f"Ensemble size: {config.n_particles}")
print(f"Overall RMSE: {rmse:.4f}")
print(f"Mean ESS: {result.ess_history.mean():.0f} / {config.n_particles}")
```

!!! tip "What to expect"
    With only 100 ensemble members for a 20-dimensional state, the EnPF should produce reasonable RMSE. A standard Bootstrap PF would need $O(10^6)$ particles for comparable accuracy in 20 dimensions.

### Example 2: Lorenz 96 — Geophysical Data Assimilation

The Lorenz 96 model is a standard benchmark in data assimilation for weather and climate:

```python
import numpy as np
from particlefilterbox.filters import EnsemblePF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class Lorenz96(ParticleFilterModel):
    """
    dx_j/dt = (x_{j+1} - x_{j-2}) * x_{j-1} - x_j + F
    Discretized with RK4, observed every dt_obs steps.
    """
    def __init__(self, k=40, F=8.0, dt=0.05, sigma_obs=1.0):
        self.k_states = k
        self.k_obs = k
        self.F = F
        self.dt = dt
        self.sigma_obs = sigma_obs

    def _rhs(self, x):
        k = self.k_states
        dxdt = np.zeros_like(x)
        for j in range(k):
            dxdt[..., j] = (
                (x[..., (j+1) % k] - x[..., (j-2) % k]) * x[..., (j-1) % k]
                - x[..., j] + self.F
            )
        return dxdt

    def _rk4_step(self, x):
        dt = self.dt
        k1 = self._rhs(x)
        k2 = self._rhs(x + 0.5 * dt * k1)
        k3 = self._rhs(x + 0.5 * dt * k2)
        k4 = self._rhs(x + dt * k3)
        return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def initial_distribution(self, n_particles, rng):
        x = self.F * np.ones((n_particles, self.k_states))
        x += rng.normal(0, 0.1, size=x.shape)
        return x

    def transition(self, particles, t, rng):
        out = np.array([self._rk4_step(p) for p in particles])
        out += rng.normal(0, 0.1, size=out.shape)
        return out

    def log_observation_likelihood(self, particles, y_t, t):
        diff = y_t - particles
        return -0.5 * np.sum(diff**2 / self.sigma_obs**2, axis=1)

    def observation_matrix(self, t):
        return np.eye(self.k_states)

    def observation_noise_cov(self, t):
        return self.sigma_obs**2 * np.eye(self.k_states)

# --- Simulate ---
rng = np.random.default_rng(123)
model = Lorenz96(k=40, F=8.0)
T = 100

# Spin up
x = model.F * np.ones(40) + rng.normal(0, 0.1, size=40)
for _ in range(1000):
    x = model._rk4_step(x)

x_true = np.zeros((T, 40))
y_obs = np.zeros((T, 40))
x_true[0] = x
y_obs[0] = x + rng.normal(0, model.sigma_obs, size=40)
for t in range(1, T):
    x_true[t] = model._rk4_step(x_true[t-1]) + rng.normal(0, 0.1, size=40)
    y_obs[t] = x_true[t] + rng.normal(0, model.sigma_obs, size=40)

# --- EnPF with localization ---
config = PFConfig(n_particles=50, resampling="systematic", seed=42)
enpf = EnsemblePF(
    model=model,
    config=config,
    inflation=1.05,
    localization="gaspari_cohn",
    localization_radius=5.0,
)
result = enpf.filter(y_obs)

rmse = np.sqrt(np.mean((result.filtered_means - x_true)**2))
print(f"Lorenz-96 (k=40) — EnPF with 50 members")
print(f"RMSE: {rmse:.4f}")
print(f"Mean ESS: {result.ess_history.mean():.0f} / {config.n_particles}")
```

!!! tip "What to expect"
    With localization and inflation, 50 ensemble members should produce stable tracking of the 40-dimensional chaotic Lorenz-96 system. Without localization, the filter would diverge due to spurious correlations.

---

## Tuning Guide

### Inflation Factor

| Value | Effect | Use case |
|-------|--------|----------|
| $\delta = 1.0$ | No inflation | Only if ensemble size $\gg$ state dimension |
| $\delta = 1.01 - 1.05$ | Mild inflation | Moderate ensemble size, low model error |
| $\delta = 1.05 - 1.10$ | Strong inflation | Small ensemble, chaotic dynamics |
| $\delta > 1.15$ | Excessive | Usually unstable — reduce |

!!! warning "Inflation too high"
    Excessive inflation causes the ensemble to spread too widely, making the analysis update unreliable. If the RMSE increases with inflation, reduce $\delta$.

### Localization

| Method | Description | Best for |
|--------|-------------|----------|
| `"gaspari_cohn"` | Smooth tapering (5th-order polynomial) | Default for spatially structured states |
| `"boxcar"` | Hard cutoff at radius | Simpler, but can introduce discontinuities |
| `None` | No localization | Small state dimension ($k < 10$) |

The **localization radius** should be set based on the physical correlation scale of the system. For the Lorenz-96 model, typical values are 3–8 grid points.

### Ensemble Size

| State dimension | Recommended ensemble size |
|----------------|:------------------------:|
| $k \leq 10$ | 50 – 200 |
| $k = 10 - 100$ | 50 – 100 (with localization) |
| $k = 100 - 1000$ | 30 – 100 (with localization) |
| $k > 1000$ | 20 – 50 (with strong localization) |

!!! note "Ensemble size vs particle count"
    Unlike standard particle filters where you need $N \gg k$, the EnPF with localization can work with $N \ll k$. The ensemble Kalman update leverages the spatial structure of the state to extract information efficiently.

### Computational Complexity

| Operation | Cost |
|-----------|------|
| Propagation | $O(N \cdot c_{\text{model}})$ |
| Ensemble covariance | $O(N \cdot k^2)$ or $O(N \cdot k)$ with localization |
| Kalman gain | $O(k \cdot k_y^2)$ |
| Ensemble update | $O(N \cdot k)$ |
| **Total per step** | **$O(N \cdot k^2)$** or **$O(N \cdot k)$** with localization |

---

## References

- Frei, M. & Künsch, H.R. (2013). Bridging the ensemble Kalman and particle filters. *Biometrika*, 100(4), 781–800.
- Evensen, G. (2009). *Data Assimilation: The Ensemble Kalman Filter*. Springer, 2nd edition.
- Burgers, G., van Leeuwen, P.J. & Evensen, G. (1998). Analysis scheme in the ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.
- van Leeuwen, P.J. (2009). Particle filtering in geophysical systems. *Monthly Weather Review*, 137(12), 4089–4114.
