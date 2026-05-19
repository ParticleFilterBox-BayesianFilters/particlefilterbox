---
title: Unscented Particle Filter
description: "The UPF — locally-adapted proposal via Unscented Kalman Filter from kalmanbox"
---

# Unscented Particle Filter

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `UnscentedPF` |
    | **Import** | `from particlefilterbox.filters import UnscentedPF` |
    | **Key idea** | Use UKF as a proposal distribution — locally adapted per particle |
    | **Complexity** | $O(N \cdot k^2)$ per time step ($k$ = state dimension) |
    | **Depends on** | [kalmanbox](https://github.com/guhaase/kalmanbox) — uses `UnscentedKalmanFilter` for sigma-point proposals |
    | **Reference** | van der Merwe, Doucet, de Freitas & Wan (2001) |

## Overview

The Unscented Particle Filter (UPF) improves the standard particle filter by using the **Unscented Kalman Filter** as a proposal distribution. Instead of proposing particles blindly from the prior (Bootstrap) or designing a model-specific proposal (SIR), the UPF runs a local UKF update **for each particle** to generate proposals that account for the current observation.

This produces a proposal that is **locally adapted** — each particle is proposed near the region of high posterior probability — without requiring derivatives or Jacobians (unlike EKF-based proposals).

!!! note "Integration with kalmanbox"
    The UPF delegates the sigma-point computation and UKF update to [kalmanbox](https://github.com/guhaase/kalmanbox). Each particle runs a local `UnscentedKalmanFilter` step to produce a Gaussian proposal centered on the UKF posterior.

    ```
    pip install kalmanbox
    ```

**Advantages:**

- Proposal incorporates the current observation — higher ESS than Bootstrap
- No derivatives or Jacobians needed (unlike EKF-based proposals)
- Handles moderate nonlinearity well via sigma-point approximation
- Automatic — no manual proposal design required

**Disadvantages:**

- Higher per-particle cost: each particle requires a UKF predict + update ($O(k^2)$)
- Assumes locally Gaussian posterior — may struggle with strongly multimodal targets
- Requires evaluating both the transition density and the proposal density for weights

---

## Algorithm

At each time step, the UPF runs a local UKF for every particle to generate the proposal:

$$
\boxed{
\begin{aligned}
&\textbf{Unscented Particle Filter} \\[6pt]
&\text{1. } \textbf{Initialize: } \text{For } i = 1, \ldots, N: \\
&\qquad x_0^{(i)} \sim p(x_0), \quad w_0^{(i)} = \tfrac{1}{N} \\
&\qquad \hat{x}_0^{(i)} = x_0^{(i)}, \quad P_0^{(i)} = P_0 \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T: \\
&\qquad \text{a. } \textbf{UKF predict: } (\hat{x}_{t|t-1}^{(i)}, P_{t|t-1}^{(i)}) = \text{UKF.predict}(\hat{x}_{t-1}^{(i)}, P_{t-1}^{(i)}) \\
&\qquad \text{b. } \textbf{UKF update: } (\hat{x}_{t|t}^{(i)}, P_{t|t}^{(i)}) = \text{UKF.update}(y_t, \hat{x}_{t|t-1}^{(i)}, P_{t|t-1}^{(i)}) \\
&\qquad \text{c. } \textbf{Propose: } x_t^{(i)} \sim \mathcal{N}(\hat{x}_{t|t}^{(i)}, P_{t|t}^{(i)}) \\
&\qquad \text{d. } \textbf{Weight: } \tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot \frac{p(y_t \mid x_t^{(i)}) \; p(x_t^{(i)} \mid x_{t-1}^{(i)})}{q_{\text{UKF}}(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t)} \\
&\qquad \text{e. } \textbf{Normalize: } w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_j \tilde{w}_t^{(j)}} \\
&\qquad \text{f. } \textbf{Resample: } \text{If } \text{ESS} < \tau \cdot N, \text{ resample} \\
&\qquad \text{g. } \textbf{Update UKF state: } \hat{x}_t^{(i)} = x_t^{(i)}, \quad P_t^{(i)} = P_{t|t}^{(i)}
\end{aligned}
}
$$

### Proposal Distribution

The UKF proposal for particle $i$ is a Gaussian:

$$
q_{\text{UKF}}(x_t \mid x_{t-1}^{(i)}, y_t) = \mathcal{N}(x_t; \hat{x}_{t|t}^{(i)}, P_{t|t}^{(i)})
$$

where $\hat{x}_{t|t}^{(i)}$ and $P_{t|t}^{(i)}$ are the posterior mean and covariance from the UKF update centered on particle $i$'s state.

### Why UKF as a Proposal?

The UKF uses **sigma points** to propagate the mean and covariance through nonlinear functions without linearization:

$$
\mathcal{X}^{(j)} = \hat{x} + \left(\sqrt{(k + \lambda) P}\right)_j, \quad j = 0, \ldots, 2k
$$

These sigma points are passed through the transition and observation functions to produce a Gaussian approximation of $p(x_t \mid x_{t-1}^{(i)}, y_t)$. Because this approximation accounts for the observation $y_t$, the resulting proposal is **much more efficient** than the blind prior proposal.

!!! note "UKF vs EKF proposals"
    An EKF-based proposal requires Jacobians of the transition and observation functions. The UKF-based proposal uses sigma points instead, making it **derivative-free**. For mildly nonlinear models, both give similar results; for strongly nonlinear models, the UKF is typically more accurate.

---

## API Reference

### Constructor

```python
from particlefilterbox.filters import UnscentedPF
from particlefilterbox.core.config import PFConfig
from kalmanbox import UnscentedKalmanFilter

config = PFConfig(
    n_particles=500,
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
)

upf = UnscentedPF(model=my_model, config=config, ukf=UnscentedKalmanFilter)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `1000` | Number of particles $N$ |
| `resampling` | `str` | `"systematic"` | Resampling scheme |
| `ess_threshold` | `float` | `0.5` | Resample when $\text{ESS} < \tau \cdot N$ |
| `seed` | `int \| None` | `None` | Random seed |
| `ukf` | `type` | `UnscentedKalmanFilter` | UKF class from kalmanbox |
| `alpha` | `float` | `1e-3` | UKF scaling parameter — controls sigma-point spread |
| `beta` | `float` | `2.0` | UKF parameter — optimal for Gaussian priors |
| `kappa` | `float` | `0.0` | UKF secondary scaling parameter |

### Model Requirements

The UPF model must provide:

| Model method | Signature | Purpose |
|-------------|-----------|---------|
| `initial_distribution` | `(n_particles, rng) → ndarray` | Sample $x_0$ |
| `transition` | `(particles, t, rng) → ndarray` | Sample $x_t \mid x_{t-1}$ |
| `log_transition_density` | `(x_curr, x_prev, t) → ndarray` | Evaluate $\log p(x_t \mid x_{t-1})$ |
| `log_observation_likelihood` | `(particles, y_t, t) → ndarray` | Evaluate $\log p(y_t \mid x_t)$ |
| `transition_function` | `(x, t) → ndarray` | Deterministic transition (no noise) |
| `observation_function` | `(x, t) → ndarray` | Deterministic observation (no noise) |
| `Q` | `(t) → ndarray` | Transition noise covariance |
| `R` | `(t) → ndarray` | Observation noise covariance |

!!! warning "Transition density required"
    Like the SIR filter, the UPF needs to **evaluate** the transition density (not just sample from it). The weight computation requires $\log p(x_t \mid x_{t-1})$ to correct for the difference between the UKF proposal and the prior.

### Batch Filtering

```python
result = upf.filter(observations)
```

Returns the same `ParticleFilterResults` as Bootstrap PF. See [Bootstrap PF — Batch Filtering](bootstrap.md#batch-filtering).

### Online Filtering

```python
rng = np.random.default_rng(42)
cloud = upf.initialize(rng)

for t, y_t in enumerate(observations):
    cloud, ll_t = upf.filter_step(cloud, y_t, t)
```

---

## Examples

### Example 1: Nonlinear Bearings-Only Tracking

A classic tracking problem where the state includes position and velocity, and only bearing (angle) measurements are available. The observation function is highly nonlinear.

```python
import numpy as np
from particlefilterbox.filters import UnscentedPF, BootstrapPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from kalmanbox import UnscentedKalmanFilter

class BearingsOnly(ParticleFilterModel):
    """
    State: [x, y, vx, vy]  (2D position and velocity)
    Observation: bearing = atan2(y - sy, x - sx)

    Transition: constant velocity + noise
    x_t = x_{t-1} + vx_{t-1} * dt + noise
    """
    k_states = 4
    k_obs = 1

    def __init__(self, dt=1.0, q=0.01, r=0.05, sensor_pos=(0, 0)):
        self.dt = dt
        self.q = q
        self.r = r
        self.sx, self.sy = sensor_pos

    def initial_distribution(self, n_particles, rng):
        x = rng.normal(1.0, 0.5, size=(n_particles, 4))
        x[:, 2:] *= 0.1  # smaller initial velocity spread
        return x

    def transition(self, particles, t, rng):
        dt = self.dt
        new = particles.copy()
        new[:, 0] += particles[:, 2] * dt
        new[:, 1] += particles[:, 3] * dt
        new += rng.normal(0, self.q, size=particles.shape)
        return new

    def transition_function(self, x, t):
        dt = self.dt
        out = x.copy()
        out[..., 0] += x[..., 2] * dt
        out[..., 1] += x[..., 3] * dt
        return out

    def observation_function(self, x, t):
        dx = x[..., 0] - self.sx
        dy = x[..., 1] - self.sy
        return np.arctan2(dy, dx)[..., np.newaxis]

    def log_transition_density(self, x_curr, x_prev, t):
        pred = self.transition_function(x_prev, t)
        diff = x_curr - pred
        return -0.5 * np.sum(diff**2, axis=1) / self.q**2

    def log_observation_likelihood(self, particles, y_t, t):
        bearing = np.arctan2(
            particles[:, 1] - self.sy,
            particles[:, 0] - self.sx,
        )
        residual = y_t[0] - bearing
        # Wrap to [-pi, pi]
        residual = (residual + np.pi) % (2 * np.pi) - np.pi
        return -0.5 * (residual / self.r)**2

    def Q(self, t):
        return self.q**2 * np.eye(4)

    def R(self, t):
        return np.array([[self.r**2]])

# --- Simulate ---
rng = np.random.default_rng(42)
T = 100
model = BearingsOnly(dt=1.0, q=0.01, r=0.05)

x_true = np.zeros((T, 4))
y_obs = np.zeros((T, 1))

x_true[0] = [1.0, 0.5, 0.05, 0.03]
for t in range(T):
    if t > 0:
        x_true[t, 0] = x_true[t-1, 0] + x_true[t-1, 2] + rng.normal(0, model.q)
        x_true[t, 1] = x_true[t-1, 1] + x_true[t-1, 3] + rng.normal(0, model.q)
        x_true[t, 2] = x_true[t-1, 2] + rng.normal(0, model.q)
        x_true[t, 3] = x_true[t-1, 3] + rng.normal(0, model.q)
    bearing = np.arctan2(x_true[t, 1], x_true[t, 0])
    y_obs[t, 0] = bearing + rng.normal(0, model.r)

# --- Compare UPF vs Bootstrap ---
config = PFConfig(n_particles=500, resampling="systematic", seed=42)

upf = UnscentedPF(model=model, config=config, ukf=UnscentedKalmanFilter)
bpf = BootstrapPF(model=model, config=config)

result_upf = upf.filter(y_obs)
result_bpf = bpf.filter(y_obs)

rmse_upf = np.sqrt(np.mean((result_upf.filtered_means[:, :2] - x_true[:, :2])**2))
rmse_bpf = np.sqrt(np.mean((result_bpf.filtered_means[:, :2] - x_true[:, :2])**2))

print(f"{'Metric':<25} {'Bootstrap':>12} {'UPF':>12}")
print("-" * 50)
print(f"{'Position RMSE':<25} {rmse_bpf:>12.4f} {rmse_upf:>12.4f}")
print(f"{'Log-likelihood':<25} {result_bpf.log_likelihood:>12.2f} {result_upf.log_likelihood:>12.2f}")
print(f"{'Mean ESS':<25} {result_bpf.ess_history.mean():>12.0f} {result_upf.ess_history.mean():>12.0f}")
```

!!! tip "What to expect"
    The UPF should show **significantly lower RMSE** and **higher ESS** than the Bootstrap PF, especially for the bearings-only problem where the observation is highly nonlinear and informative. The UKF proposal moves particles toward the bearing-consistent region before weighting.

### Example 2: Stochastic Volatility with UPF

```python
import numpy as np
from particlefilterbox.filters import UnscentedPF, BootstrapPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from kalmanbox import UnscentedKalmanFilter

class SVModelUPF(ParticleFilterModel):
    """
    x_t = phi * x_{t-1} + sigma * eta_t    (log-volatility)
    y_t = beta * exp(x_t / 2) * eps_t      (returns)
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

    def transition_function(self, x, t):
        return self.phi * x

    def observation_function(self, x, t):
        return np.zeros_like(x[..., :1])  # E[y|x] = 0

    def log_transition_density(self, x_curr, x_prev, t):
        mu = self.phi * x_prev[:, 0]
        return -0.5 * ((x_curr[:, 0] - mu) / self.sigma)**2

    def log_observation_likelihood(self, particles, y_t, t):
        vol = self.beta * np.exp(particles[:, 0] / 2)
        return -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (y_t[0] / vol)**2

    def Q(self, t):
        return np.array([[self.sigma**2]])

    def R(self, t):
        return np.array([[self.beta**2]])

# --- Simulate ---
sv = SVModelUPF()
rng = np.random.default_rng(456)
T = 500

x_true = np.zeros(T)
y_obs = np.zeros(T)

std_0 = sv.sigma / np.sqrt(1 - sv.phi**2)
x_true[0] = rng.normal(0.0, std_0)
y_obs[0] = sv.beta * np.exp(x_true[0] / 2) * rng.normal()
for t in range(1, T):
    x_true[t] = sv.phi * x_true[t-1] + rng.normal(0.0, sv.sigma)
    y_obs[t] = sv.beta * np.exp(x_true[t] / 2) * rng.normal()

# --- Compare ---
config = PFConfig(n_particles=500, resampling="systematic", seed=42)

upf = UnscentedPF(model=sv, config=config, ukf=UnscentedKalmanFilter)
bpf = BootstrapPF(model=sv, config=config)

result_upf = upf.filter(y_obs)
result_bpf = bpf.filter(y_obs)

print(f"{'Metric':<25} {'Bootstrap':>12} {'UPF':>12}")
print("-" * 50)
print(f"{'Log-likelihood':<25} {result_bpf.log_likelihood:>12.2f} {result_upf.log_likelihood:>12.2f}")
print(f"{'Mean ESS':<25} {result_bpf.ess_history.mean():>12.0f} {result_upf.ess_history.mean():>12.0f}")
```

---

## Tuning Guide

### UKF Scaling Parameters

The UKF parameters $\alpha$, $\beta$, $\kappa$ control how sigma points are placed:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | $10^{-3}$ | Spread of sigma points around the mean. Smaller = tighter |
| `beta` | $2.0$ | Optimal for Gaussian distributions |
| `kappa` | $0.0$ | Secondary scaling. Set to $3 - k$ for some heuristics |

!!! tip "Rule of thumb"
    The defaults work well for most problems. Increase `alpha` (e.g., to 0.1 or 1.0) if the model is highly nonlinear and sigma points need to explore further from the mean.

### Number of Particles

The UKF proposal is observation-informed, so the UPF typically needs **fewer particles** than the Bootstrap PF:

| Scenario | UPF particles (relative to Bootstrap) |
|----------|:-------------------------------------:|
| Mildly nonlinear | 0.2× – 0.5× |
| Moderately nonlinear | 0.3× – 0.5× |
| Highly nonlinear (non-Gaussian) | 0.5× – 0.8× |
| Multimodal posterior | Use [Regularized PF](regularized.md) instead |

### When to Use UPF

| Scenario | Recommendation |
|----------|---------------|
| Nonlinear model, no analytical proposal available | **Use UPF** — automatic observation-informed proposal |
| Model with available Jacobians | Consider EKF-based SIR — lower per-particle cost |
| Model with linear sub-structure | Use [RBPF](rbpf.md) — exact marginalization is better than approximation |
| Very high dimension ($k > 20$) | Use [Ensemble PF](ensemble.md) — UKF cost grows with $k^2$ |
| Strongly multimodal posterior | UPF's Gaussian proposal may miss modes |

### Computational Complexity

| Operation | Cost |
|-----------|------|
| UKF predict (per particle) | $O(k^2)$ — $2k+1$ sigma points |
| UKF update (per particle) | $O(k^2 \cdot k_y)$ |
| Proposal sampling | $O(N \cdot k)$ |
| Weight computation | $O(N)$ |
| **Total per step** | **$O(N \cdot k^2)$** |

---

## References

- van der Merwe, R., Doucet, A., de Freitas, N. & Wan, E. (2001). The Unscented Particle Filter. In *Advances in Neural Information Processing Systems 13 (NIPS)*.
- Julier, S.J. & Uhlmann, J.K. (2004). Unscented filtering and nonlinear estimation. *Proceedings of the IEEE*, 92(3), 401–422.
- Doucet, A. & Johansen, A.M. (2009). A tutorial on particle filtering and smoothing: fifteen years later. In *Handbook of Nonlinear Filtering*, Oxford University Press.
