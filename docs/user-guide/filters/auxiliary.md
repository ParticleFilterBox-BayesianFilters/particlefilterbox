---
title: Auxiliary Particle Filter
description: "The Auxiliary PF (Pitt & Shephard, 1999) — look-ahead filtering via first-stage pre-selection"
---

# Auxiliary Particle Filter

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `AuxiliaryPF` |
    | **Import** | `from particlefilterbox.filters import AuxiliaryPF` |
    | **Key idea** | Pre-select particles using approximate likelihood at predicted means |
    | **Complexity** | $O(N)$ per time step |
    | **Reference** | Pitt & Shephard (1999) |

## Overview

The Auxiliary Particle Filter (APF) improves upon the Bootstrap PF by introducing a **look-ahead** mechanism. Before propagating particles through the transition, the APF evaluates an **approximate likelihood** at the predicted state mean for each particle. Particles whose predictions are more consistent with the current observation receive higher selection probability — effectively guiding particles toward informative regions *before* propagation.

This two-stage strategy is especially effective when observations are **highly informative** (low observation noise), a regime where the Bootstrap PF wastes most particles in low-likelihood regions.

**Advantages:**

- Better particle allocation when observations are informative
- No custom proposal needed — works with any model
- Same $O(N)$ complexity as Bootstrap
- Particularly effective for jump and regime-switching models

**Disadvantages:**

- Requires evaluating the observation likelihood at predicted means (extra cost)
- The first-stage approximation can be poor if the transition is highly nonlinear
- Two resampling steps (first-stage + conditional) can increase path degeneracy

---

## Algorithm

The APF operates in two stages at each time step:

$$
\boxed{
\begin{aligned}
&\textbf{Auxiliary Particle Filter} \\[6pt]
&\text{1. } \textbf{Initialize: } x_0^{(i)} \sim p(x_0), \quad w_0^{(i)} = \tfrac{1}{N} \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T: \\[4pt]
&\qquad \textbf{--- First Stage (Pre-selection) ---} \\
&\qquad \text{a. } \textbf{Predict means: } \mu_t^{(i)} = \mathbb{E}[x_t \mid x_{t-1}^{(i)}] \\
&\qquad \text{b. } \textbf{Approximate likelihood: } \lambda_t^{(i)} = p(y_t \mid \mu_t^{(i)}) \\
&\qquad \text{c. } \textbf{First-stage weights: } \nu_t^{(i)} = w_{t-1}^{(i)} \cdot \lambda_t^{(i)} \\
&\qquad \text{d. } \textbf{Resample: } \text{Draw ancestor indices } k^{(i)} \sim \text{Cat}(\bar{\nu}_t^{(1:N)}) \\[4pt]
&\qquad \textbf{--- Second Stage (Propagation + Correction) ---} \\
&\qquad \text{e. } \textbf{Propagate: } x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(k^{(i)})}) \\
&\qquad \text{f. } \textbf{Correct weights: } w_t^{(i)} \propto \frac{p(y_t \mid x_t^{(i)})}{\lambda_t^{(k^{(i)})}} \\
&\qquad \text{g. } \textbf{Normalize: } w_t^{(i)} = \frac{w_t^{(i)}}{\sum_j w_t^{(j)}}
\end{aligned}
}
$$

### Intuition: Why Does This Help?

In the Bootstrap PF, particles are propagated *blindly* — they don't know the current observation $y_t$. The APF's first stage asks: "if I propagate particle $i$, how likely is it to explain $y_t$?" Particles with high approximate likelihood $\lambda_t^{(i)}$ get duplicated; those with low $\lambda_t^{(i)}$ get discarded — *before* wasting a transition sample.

The second-stage weight correction $p(y_t \mid x_t) / \lambda_t^{(k)}$ accounts for the approximation error: if the true likelihood differs from the predicted one, the correction rebalances.

!!! note "First-stage weight derivation"
    The first-stage weight $\nu_t^{(i)} = w_{t-1}^{(i)} \cdot \lambda_t^{(i)}$ combines the *prior weight* (how important was particle $i$ at $t-1$?) with the *predictive likelihood* (how well does particle $i$'s prediction match $y_t$?). This is equivalent to targeting a joint distribution over the auxiliary index and the state.

---

## API Reference

### Constructor

```python
from particlefilterbox.filters import AuxiliaryPF
from particlefilterbox.core.config import PFConfig

config = PFConfig(
    n_particles=1000,
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
)

apf = AuxiliaryPF(model=my_model, config=config)
```

### First-Stage Mean Computation

The APF computes the transition mean $\mu_t^{(i)} = \mathbb{E}[x_t \mid x_{t-1}^{(i)}]$ for the first-stage weights. The model can provide this via an optional method:

| Model method | Signature | Purpose |
|-------------|-----------|---------|
| `transition_mean` | `(particles, t) → ndarray` | Deterministic predicted mean (no noise) |

If `transition_mean` is not provided, the APF falls back to calling `transition` with a fixed-seed RNG as an approximation.

!!! tip "Implementing `transition_mean`"
    For most models, the transition mean is simply the transition equation **without noise**:

    ```python
    def transition_mean(self, particles, t):
        # x_t = phi * x_{t-1}  (no noise term)
        return self.phi * particles
    ```

    Providing this method avoids the fixed-seed approximation and gives cleaner first-stage weights.

### Batch Filtering

```python
result = apf.filter(observations)
```

Returns the same `ParticleFilterResults` as Bootstrap PF. See [Bootstrap PF — Batch Filtering](bootstrap.md#batch-filtering).

---

## Examples

### Example 1: Jump Model with Informative Observations

A model where the state can jump, and the observation is precise enough to reveal the jump immediately. The APF excels here because its first-stage weights detect the jump *before* propagation.

```python
import numpy as np
from particlefilterbox.filters import BootstrapPF, AuxiliaryPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class JumpModel(ParticleFilterModel):
    """
    x_t = x_{t-1} + jump_t + eta_t
    y_t = x_t + eps_t

    jump_t ~ N(0, 3^2)  with probability p_jump, else 0
    eta_t  ~ N(0, 0.1^2)
    eps_t  ~ N(0, 0.2^2)
    """
    k_states = 1
    k_obs = 1

    def __init__(self, p_jump=0.05):
        self.p_jump = p_jump
        self.sigma_eta = 0.1
        self.sigma_jump = 3.0
        self.sigma_eps = 0.2

    def initial_distribution(self, n_particles, rng):
        return rng.normal(0.0, 1.0, size=(n_particles, 1))

    def transition(self, particles, t, rng):
        n = particles.shape[0]
        jumps = rng.binomial(1, self.p_jump, size=n)
        noise = rng.normal(0.0, self.sigma_eta, size=(n, 1))
        jump_vals = jumps[:, np.newaxis] * rng.normal(0.0, self.sigma_jump, size=(n, 1))
        return particles + noise + jump_vals

    def transition_mean(self, particles, t):
        # Expected value (no noise, no jump)
        return particles.copy()

    def log_observation_likelihood(self, particles, y_t, t):
        residual = y_t[0] - particles[:, 0]
        return -0.5 * (residual / self.sigma_eps)**2

# --- Simulate with jumps ---
rng = np.random.default_rng(789)
T = 300
model = JumpModel(p_jump=0.05)

x_true = np.zeros(T)
y_obs = np.zeros(T)

x_true[0] = 0.0
y_obs[0] = rng.normal(0.0, model.sigma_eps)
for t in range(1, T):
    jump = (rng.random() < model.p_jump) * rng.normal(0.0, model.sigma_jump)
    x_true[t] = x_true[t - 1] + rng.normal(0.0, model.sigma_eta) + jump
    y_obs[t] = x_true[t] + rng.normal(0.0, model.sigma_eps)

# --- Compare Bootstrap vs APF ---
config = PFConfig(n_particles=1000, resampling="systematic", seed=42)

bpf = BootstrapPF(model=model, config=config)
apf = AuxiliaryPF(model=model, config=config)

result_bpf = bpf.filter(y_obs)
result_apf = apf.filter(y_obs)

rmse_bpf = np.sqrt(np.mean((result_bpf.filtered_means[:, 0] - x_true)**2))
rmse_apf = np.sqrt(np.mean((result_apf.filtered_means[:, 0] - x_true)**2))

print(f"{'Metric':<25} {'Bootstrap':>12} {'APF':>12}")
print("-" * 50)
print(f"{'RMSE':<25} {rmse_bpf:>12.4f} {rmse_apf:>12.4f}")
print(f"{'Log-likelihood':<25} {result_bpf.log_likelihood:>12.2f} {result_apf.log_likelihood:>12.2f}")
print(f"{'Mean ESS':<25} {result_bpf.ess_history.mean():>12.0f} {result_apf.ess_history.mean():>12.0f}")
print(f"{'Min ESS':<25} {result_bpf.ess_history.min():>12.0f} {result_apf.ess_history.min():>12.0f}")
```

!!! tip "Expected results"
    The APF should show **lower RMSE** and **higher mean ESS**, especially around the time steps where jumps occur. The first-stage weights allow the APF to rapidly redirect particles toward the post-jump state.

### Example 2: Stochastic Volatility — Bootstrap vs APF

The stochastic volatility model is a classic benchmark. The observation $y_t = \beta \exp(x_t / 2) \, \varepsilon_t$ is informative about the log-volatility $x_t$, especially when $\beta$ is small.

```python
import numpy as np
from particlefilterbox.filters import BootstrapPF, AuxiliaryPF
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel

class SVModel(ParticleFilterModel):
    """Stochastic volatility with transition_mean for APF."""
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

    def log_observation_likelihood(self, particles, y_t, t):
        vol = self.beta * np.exp(particles[:, 0] / 2)
        return -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (y_t[0] / vol)**2

# --- Simulate ---
sv = SVModel(phi=0.98, sigma=0.16, beta=0.65)
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

# --- Compare ---
config = PFConfig(n_particles=2000, resampling="systematic", seed=42)

bpf = BootstrapPF(model=sv, config=config)
apf = AuxiliaryPF(model=sv, config=config)

result_bpf = bpf.filter(y_obs)
result_apf = apf.filter(y_obs)

print(f"{'Metric':<25} {'Bootstrap':>12} {'APF':>12}")
print("-" * 50)
print(f"{'Log-likelihood':<25} {result_bpf.log_likelihood:>12.2f} {result_apf.log_likelihood:>12.2f}")
print(f"{'Mean ESS':<25} {result_bpf.ess_history.mean():>12.0f} {result_apf.ess_history.mean():>12.0f}")
print(f"{'Resampling rate':<25} {result_bpf.resampled.mean():>12.1%} {result_apf.resampled.mean():>12.1%}")
```

---

## Tuning Guide

### First-Stage Strategy

The quality of the APF depends on how well the first-stage approximation $\lambda_t^{(i)} = p(y_t \mid \mu_t^{(i)})$ predicts the true likelihood $p(y_t \mid x_t^{(i)})$.

| Strategy | Description | When to use |
|----------|-------------|-------------|
| **Mean** (default) | $\mu_t^{(i)} = \mathbb{E}[x_t \mid x_{t-1}^{(i)}]$ | General purpose; works well for most models |
| **Mode** | $\mu_t^{(i)} = \arg\max p(x_t \mid x_{t-1}^{(i)})$ | Skewed transition distributions |
| **Sample** | $\mu_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)})$ | Highly non-unimodal transitions |

!!! tip "Implementing `transition_mean`"
    Always implement `transition_mean` on your model when using the APF. Without it, the filter uses a fixed-seed simulation as a proxy, which can introduce unnecessary approximation error.

### Number of Particles

The APF typically needs **fewer particles** than the Bootstrap PF for the same accuracy, because the first stage pre-selects informative particles:

| Scenario | APF particles (relative to Bootstrap) |
|----------|:------------------------------------:|
| Weakly informative observations | ~Same (first stage adds little) |
| Moderately informative observations | 0.5× – 0.7× |
| Highly informative observations | 0.3× – 0.5× |
| Jump / regime-switching models | 0.3× – 0.5× |

### When the APF Helps Most

The first-stage pre-selection is most valuable when:

1. **Observation noise is low** relative to transition noise — the likelihood is peaked
2. **The state can jump** — most prior-proposed particles miss the jump, but the APF pre-selects toward it
3. **Multi-modal transitions** — the APF can allocate particles across modes based on predictive likelihood

### When the APF May Not Help

!!! warning "Diminishing returns"
    If the observation is weakly informative (high observation noise), the approximate likelihoods $\lambda_t^{(i)}$ will be nearly uniform across particles. In this case the first stage reduces to standard resampling, and the APF adds computational overhead without benefit. Use the [Bootstrap PF](bootstrap.md) instead.

### Computational Cost

| Operation | Cost |
|-----------|------|
| Transition mean | $O(N)$ |
| First-stage likelihood | $O(N)$ |
| First-stage resampling | $O(N)$ |
| Propagation | $O(N)$ |
| Second-stage likelihood | $O(N)$ |
| **Total per step** | **$O(N)$** — same order as Bootstrap, ~2× constant |

---

## References

- Pitt, M.K. & Shephard, N. (1999). Filtering via Simulation: Auxiliary Particle Filters. *Journal of the American Statistical Association*, 94(446), 590–599.
- Carpenter, J., Clifford, P. & Fearnhead, P. (1999). Improved particle filter for nonlinear problems. *IEE Proceedings — Radar, Sonar and Navigation*, 146(1), 2–7.
- Johansen, A.M. & Doucet, A. (2008). A note on auxiliary particle filters. *Statistics & Probability Letters*, 78(12), 1498–1504.
