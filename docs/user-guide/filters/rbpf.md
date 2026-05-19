---
title: Rao-Blackwellized Particle Filter
description: "The RBPF — exploit linear sub-structure to reduce variance via analytical marginalization with kalmanbox"
---

# Rao-Blackwellized Particle Filter

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `RaoBlackwellPF` |
    | **Import** | `from particlefilterbox.filters import RaoBlackwellPF` |
    | **Key idea** | Split state into linear + nonlinear; marginalize the linear part analytically |
    | **Complexity** | $O(N \cdot k_l^3)$ per time step ($k_l$ = linear state dimension) |
    | **Depends on** | [kalmanbox](https://github.com/guhaase/kalmanbox) — each particle carries a `KalmanFilter` instance |
    | **Reference** | Doucet, de Freitas, Murphy & Russell (2000) |

## Overview

The Rao-Blackwellized Particle Filter (RBPF) exploits **mixed linear/nonlinear structure** in the state-space model. When part of the state evolves linearly (conditioned on the nonlinear part), RBPF analytically marginalizes the linear component using a Kalman filter, while representing only the nonlinear component with particles.

This decomposition leverages the **Rao-Blackwell theorem**: the variance of the marginal estimator is always less than or equal to the variance of the joint estimator. In practice, RBPF can achieve the same accuracy as a standard particle filter with **orders of magnitude fewer particles**.

!!! note "Integration with kalmanbox"
    RBPF delegates the linear state update to [kalmanbox](https://github.com/guhaase/kalmanbox). Each particle carries its own `KalmanFilter` instance, which tracks the conditional posterior of the linear sub-state given the particle's nonlinear trajectory.

    ```
    pip install kalmanbox
    ```

**Advantages:**

- Dramatic variance reduction when the model has a significant linear component
- Fewer particles needed for the same accuracy (often 10× – 100× fewer)
- Exact treatment of the linear sub-state — no approximation error for that component
- Seamless integration with kalmanbox for the Kalman filtering sub-problem

**Disadvantages:**

- Requires the model to be decomposable into linear and nonlinear components
- Each particle carries a Kalman filter → higher per-particle cost $O(k_l^3)$
- Model class is more complex to implement than standard `ParticleFilterModel`

---

## Algorithm

Consider a state-space model where the full state $x_t = (z_t, \theta_t)$ decomposes as:

$$
\begin{aligned}
\theta_t &= g(\theta_{t-1}) + \eta_t^{\theta} &\quad &\text{(nonlinear component)} \\
z_t &= F(\theta_t) \, z_{t-1} + B(\theta_t) \, u_t + \eta_t^{z} &\quad &\text{(conditionally linear component)} \\
y_t &= H(\theta_t) \, z_t + \varepsilon_t &\quad &\text{(observation)}
\end{aligned}
$$

where $\eta_t^z \sim \mathcal{N}(0, Q(\theta_t))$ and $\varepsilon_t \sim \mathcal{N}(0, R(\theta_t))$. Conditioned on $\theta_t$, the sub-state $z_t$ is **linear Gaussian** and can be tracked exactly by a Kalman filter.

$$
\boxed{
\begin{aligned}
&\textbf{Rao-Blackwellized Particle Filter} \\[6pt]
&\text{1. } \textbf{Initialize: } \text{For } i = 1, \ldots, N: \\
&\qquad \theta_0^{(i)} \sim p(\theta_0), \quad w_0^{(i)} = \tfrac{1}{N} \\
&\qquad \text{KF}^{(i)} \leftarrow \text{KalmanFilter}(\hat{z}_0, P_0) \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T: \\
&\qquad \text{a. } \textbf{Propagate nonlinear: } \theta_t^{(i)} \sim p(\theta_t \mid \theta_{t-1}^{(i)}) \\
&\qquad \text{b. } \textbf{Kalman predict: } \text{KF}^{(i)}.\text{predict}\!\big(F(\theta_t^{(i)}),\, Q(\theta_t^{(i)})\big) \\
&\qquad \text{c. } \textbf{Kalman update: } \text{KF}^{(i)}.\text{update}\!\big(y_t,\, H(\theta_t^{(i)}),\, R(\theta_t^{(i)})\big) \\
&\qquad \text{d. } \textbf{Weight: } \tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot p(y_t \mid \theta_{1:t}^{(i)}, y_{1:t-1}) \\
&\qquad \text{e. } \textbf{Normalize: } w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_j \tilde{w}_t^{(j)}} \\
&\qquad \text{f. } \textbf{Resample: } \text{If } \text{ESS} < \tau \cdot N, \text{ resample particles and clone KF instances}
\end{aligned}
}
$$

### How Are the Weights Computed?

The incremental weight for particle $i$ is the **predictive likelihood** from the Kalman filter:

$$
p(y_t \mid \theta_{1:t}^{(i)}, y_{1:t-1}) = \mathcal{N}(y_t; \hat{y}_t^{(i)}, S_t^{(i)})
$$

where $\hat{y}_t^{(i)} = H(\theta_t^{(i)}) \hat{z}_{t|t-1}^{(i)}$ is the predicted observation and $S_t^{(i)} = H(\theta_t^{(i)}) P_{t|t-1}^{(i)} H(\theta_t^{(i)})^\top + R(\theta_t^{(i)})$ is the innovation covariance. This is computed automatically by the Kalman filter update step.

### Why Does Rao-Blackwellization Reduce Variance?

By the **Rao-Blackwell theorem**:

$$
\text{Var}\!\big[\mathbb{E}[h(z, \theta) \mid \theta]\big] \leq \text{Var}\!\big[h(z, \theta)\big]
$$

The left side is what the RBPF computes — it marginalizes $z$ analytically via the Kalman filter. The right side is what a standard PF computes — it samples both $z$ and $\theta$. The inequality is strict whenever $h$ depends on $z$, which is almost always the case in practice.

---

## API Reference

### Constructor

```python
from particlefilterbox.filters import RaoBlackwellPF
from particlefilterbox.core.config import PFConfig
from kalmanbox import KalmanFilter

config = PFConfig(
    n_particles=500,
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
)

rbpf = RaoBlackwellPF(model=my_model, config=config, kalman=KalmanFilter)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `1000` | Number of particles $N$ (for the nonlinear component) |
| `resampling` | `str` | `"systematic"` | Resampling scheme |
| `ess_threshold` | `float` | `0.5` | Resample when $\text{ESS} < \tau \cdot N$ |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `kalman` | `type` | `KalmanFilter` | Kalman filter class from kalmanbox |

### Model Requirements

The RBPF requires a model that inherits from `RBParticleFilterModel` and provides the decomposition:

| Model method | Signature | Purpose |
|-------------|-----------|---------|
| `nonlinear_initial` | `(n_particles, rng) → ndarray` | Sample $\theta_0$ |
| `nonlinear_transition` | `(particles, t, rng) → ndarray` | Sample $\theta_t \mid \theta_{t-1}$ |
| `linear_matrices` | `(theta, t) → (F, Q, H, R)` | System matrices conditioned on $\theta_t$ |
| `linear_initial` | `() → (z0, P0)` | Initial mean and covariance for $z_0$ |

### Batch Filtering

```python
result = rbpf.filter(observations)
```

| Result attribute | Shape | Description |
|------------------|-------|-------------|
| `filtered_means` | `(T, k)` | Weighted mean of full state $(z_t, \theta_t)$ |
| `filtered_covs` | `(T, k, k)` | Weighted covariance of full state |
| `log_likelihood` | scalar | Total log-marginal likelihood |
| `log_likelihoods` | `(T,)` | Incremental log-likelihoods |
| `ess_history` | `(T,)` | Effective sample size at each step |
| `resampled` | `(T,)` | Boolean mask of resampling events |
| `final_cloud` | `ParticleCloud` | Final particle cloud (nonlinear component) |

### Online Filtering

```python
rng = np.random.default_rng(42)
cloud = rbpf.initialize(rng)

for t, y_t in enumerate(observations):
    cloud, ll_t = rbpf.filter_step(cloud, y_t, t)
    # Each particle in cloud carries a KalmanFilter state
```

---

## Examples

### Example 1: Mixed Linear/Nonlinear Model

A model where a nonlinear parameter $\theta_t$ drives the dynamics of a linear state $z_t$. This is common in economics (time-varying parameter models) and signal processing.

```python
import numpy as np
from particlefilterbox.filters import RaoBlackwellPF, BootstrapPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import RBParticleFilterModel, ParticleFilterModel
from kalmanbox import KalmanFilter

class MixedModel(RBParticleFilterModel):
    """
    theta_t = 0.95 * theta_{t-1} + 0.1 * eta_t     (nonlinear AR parameter)
    z_t     = phi(theta_t) * z_{t-1} + nu_t          (linear state)
    y_t     = z_t + eps_t                             (observation)

    where phi(theta) = tanh(theta) maps theta to (-1, 1)
    """
    k_nonlinear = 1
    k_linear = 1
    k_obs = 1

    def nonlinear_initial(self, n_particles, rng):
        return rng.normal(0.0, 0.5, size=(n_particles, 1))

    def nonlinear_transition(self, particles, t, rng):
        return 0.95 * particles + rng.normal(0.0, 0.1, size=particles.shape)

    def linear_initial(self):
        z0 = np.array([0.0])
        P0 = np.array([[1.0]])
        return z0, P0

    def linear_matrices(self, theta, t):
        phi = np.tanh(theta[0])
        F = np.array([[phi]])         # transition matrix
        Q = np.array([[0.5]])         # transition noise covariance
        H = np.array([[1.0]])         # observation matrix
        R = np.array([[0.1]])         # observation noise covariance
        return F, Q, H, R

# --- Simulate data ---
rng = np.random.default_rng(123)
T = 300
theta_true = np.zeros(T)
z_true = np.zeros(T)
y_obs = np.zeros(T)

theta_true[0] = rng.normal(0.0, 0.5)
z_true[0] = rng.normal(0.0, 1.0)
y_obs[0] = z_true[0] + rng.normal(0.0, np.sqrt(0.1))

for t in range(1, T):
    theta_true[t] = 0.95 * theta_true[t - 1] + rng.normal(0.0, 0.1)
    phi = np.tanh(theta_true[t])
    z_true[t] = phi * z_true[t - 1] + rng.normal(0.0, np.sqrt(0.5))
    y_obs[t] = z_true[t] + rng.normal(0.0, np.sqrt(0.1))

# --- RBPF with kalmanbox ---
config = PFConfig(n_particles=500, resampling="systematic", seed=42)
rbpf = RaoBlackwellPF(model=MixedModel(), config=config, kalman=KalmanFilter)
result_rbpf = rbpf.filter(y_obs)

print(f"RBPF log-likelihood: {result_rbpf.log_likelihood:.2f}")
print(f"RBPF mean ESS: {result_rbpf.ess_history.mean():.0f} / {config.n_particles}")
```

!!! tip "What to expect"
    With 500 particles, the RBPF achieves high ESS because the linear state $z_t$ is marginalized analytically. Compare with a standard Bootstrap PF on the full 2D state — the RBPF typically achieves **3–10× higher ESS** with the same particle count.

### Example 2: DSGE-Style Model with Linear Core

Many Dynamic Stochastic General Equilibrium (DSGE) models have a linearized core driven by nonlinear shock processes. The RBPF is ideal for this structure.

```python
import numpy as np
from particlefilterbox.filters import RaoBlackwellPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import RBParticleFilterModel
from kalmanbox import KalmanFilter

class DSGELike(RBParticleFilterModel):
    """
    Simplified DSGE-like model:
    - Nonlinear: stochastic volatility sigma_t
    - Linear: output gap (y), inflation (pi) given sigma_t

    sigma_t = rho_s * sigma_{t-1} + eta_t           (log-vol process)
    [y_t, pi_t]' = A [y_{t-1}, pi_{t-1}]' + exp(sigma_t) * nu_t
    obs_t = C [y_t, pi_t]' + eps_t
    """
    k_nonlinear = 1  # sigma
    k_linear = 2     # (y, pi)
    k_obs = 2

    def __init__(self, rho_s=0.95, sigma_eta=0.1):
        self.rho_s = rho_s
        self.sigma_eta = sigma_eta
        self.A = np.array([[0.8, 0.1],
                           [0.2, 0.7]])
        self.C = np.eye(2)
        self.R = 0.01 * np.eye(2)

    def nonlinear_initial(self, n_particles, rng):
        std = self.sigma_eta / np.sqrt(1 - self.rho_s**2)
        return rng.normal(0.0, std, size=(n_particles, 1))

    def nonlinear_transition(self, particles, t, rng):
        return self.rho_s * particles + rng.normal(
            0.0, self.sigma_eta, size=particles.shape
        )

    def linear_initial(self):
        return np.zeros(2), np.eye(2)

    def linear_matrices(self, sigma, t):
        vol = np.exp(sigma[0])
        F = self.A
        Q = (vol**2) * np.eye(2)
        H = self.C
        R = self.R
        return F, Q, H, R

# --- Simulate ---
rng = np.random.default_rng(789)
T = 200
model = DSGELike()

sigma_true = np.zeros(T)
state_true = np.zeros((T, 2))
y_obs = np.zeros((T, 2))

sigma_true[0] = rng.normal(0.0, model.sigma_eta / np.sqrt(1 - model.rho_s**2))
state_true[0] = rng.multivariate_normal(np.zeros(2), np.eye(2))
y_obs[0] = model.C @ state_true[0] + rng.multivariate_normal(np.zeros(2), model.R)

for t in range(1, T):
    sigma_true[t] = model.rho_s * sigma_true[t - 1] + rng.normal(0.0, model.sigma_eta)
    vol = np.exp(sigma_true[t])
    state_true[t] = model.A @ state_true[t - 1] + rng.normal(0.0, vol, size=2)
    y_obs[t] = model.C @ state_true[t] + rng.multivariate_normal(np.zeros(2), model.R)

# --- Filter ---
config = PFConfig(n_particles=500, resampling="systematic", seed=42)
rbpf = RaoBlackwellPF(model=model, config=config, kalman=KalmanFilter)
result = rbpf.filter(y_obs)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Mean ESS: {result.ess_history.mean():.0f} / {config.n_particles}")
```

!!! tip "What to expect"
    The RBPF tracks the 2D linear state analytically via Kalman filters, while particles cover the 1D stochastic volatility process. This 3D state is effectively tracked with only 500 particles on the 1D nonlinear subspace.

---

## Tuning Guide

### Number of Particles

Because the linear component is marginalized, the number of particles only needs to cover the **nonlinear** sub-state:

| Nonlinear state dimension | Recommended $N$ |
|--------------------------|:--------------:|
| 1D (single parameter) | 200 – 500 |
| 2–3D | 500 – 2,000 |
| 4–5D | 2,000 – 5,000 |

!!! warning "Resampling and KF cloning"
    When resampling occurs, each resampled particle's Kalman filter state (mean $\hat{z}$ and covariance $P$) must be **deep-copied**. This is handled automatically by `RaoBlackwellPF`, but it adds memory overhead proportional to $N \cdot k_l^2$.

### When to Use RBPF

| Scenario | Recommendation |
|----------|---------------|
| Model has clear linear sub-structure | **Use RBPF** — significant variance reduction |
| Linear component is high-dimensional | **Use RBPF** — avoids curse of dimensionality for linear states |
| Model is fully nonlinear | Use [Bootstrap PF](bootstrap.md) or [UPF](upf.md) — no benefit from RBPF |
| Nonlinear component is high-dimensional | Consider [Ensemble PF](ensemble.md) — RBPF still needs particles for the nonlinear part |

### Computational Complexity

| Operation | Cost |
|-----------|------|
| Nonlinear propagation | $O(N)$ |
| KF predict (per particle) | $O(k_l^3)$ |
| KF update (per particle) | $O(k_l^2 \cdot k_y)$ |
| Resampling + KF clone | $O(N \cdot k_l^2)$ |
| **Total per step** | **$O(N \cdot k_l^3)$** |

The overhead per particle is dominated by the Kalman filter operations. For $k_l \leq 10$, this is negligible; for larger linear states, the cost is still far less than representing the full state with particles.

---

## References

- Doucet, A., de Freitas, N., Murphy, K. & Russell, S. (2000). Rao-Blackwellised particle filtering for dynamic Bayesian networks. *Proceedings of the 16th Conference on Uncertainty in Artificial Intelligence (UAI)*, 176–183.
- Schön, T., Gustafsson, F. & Nordlund, P.J. (2005). Marginalized particle filters for mixed linear/nonlinear state-space models. *IEEE Transactions on Signal Processing*, 53(7), 2279–2289.
- Chen, R. & Liu, J.S. (2000). Mixture Kalman filters. *Journal of the Royal Statistical Society: Series B*, 62(3), 493–508.
- Chopin, N. & Papaspiliopoulos, O. (2020). *An Introduction to Sequential Monte Carlo*. Springer.
