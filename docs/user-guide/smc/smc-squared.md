---
title: "SMC²"
description: "SMC² (Chopin, Jacob & Papaspiliopoulos, 2013) — nested Sequential Monte Carlo for joint state and parameter estimation"
---

# SMC$^2$

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `SMCSquared` |
    | **Import** | `from particlefilterbox.smc import SMCSquared` |
    | **Target** | Joint posterior $p(\theta, x_{0:T} \mid y_{1:T})$ in state-space models |
    | **Complexity** | $O(N_\theta \cdot N_x \cdot T)$ |
    | **Reference** | Chopin, Jacob & Papaspiliopoulos (2013) |

## Overview

SMC$^2$ is a **nested SMC algorithm** designed for **online Bayesian inference** in state-space models. It simultaneously estimates the static parameters $\theta$ and the latent states $x_{0:T}$ by running two levels of SMC:

- **Outer SMC** --- an SMC sampler over the parameter space, with particles $\theta^{(1)}, \ldots, \theta^{(N_\theta)}$
- **Inner SMC** --- a particle filter for each parameter particle, estimating $p(x_{0:t} \mid y_{1:t}, \theta^{(i)})$

This makes SMC$^2$ a fully **online** algorithm: as each new observation $y_t$ arrives, both parameter and state estimates are updated without revisiting past data.

**Advantages:**

- Joint state and parameter estimation in a single pass
- Fully online --- no need to store or reprocess past data
- Provides marginal likelihood estimates for model comparison
- Asymptotically exact (consistent as $N_\theta, N_x \to \infty$)

**Disadvantages:**

- Computationally expensive: $O(N_\theta \cdot N_x \cdot T)$
- Requires running $N_\theta$ particle filters in parallel
- The inner PF must be reset when parameters are rejuvenated

---

## Algorithm

$$
\boxed{
\begin{aligned}
&\textbf{SMC}^2 \\[6pt]
&\textbf{Input: } \text{Prior } p(\theta), \text{ model } p(x_t \mid x_{t-1}, \theta), \, p(y_t \mid x_t, \theta) \\[4pt]
&\text{1. } \textbf{Initialize: } \text{For } i = 1, \ldots, N_\theta: \\
&\qquad \theta^{(i)} \sim p(\theta), \quad w_0^{(i)} = \tfrac{1}{N_\theta} \\
&\qquad \text{Initialize PF}^{(i)} \text{ with } N_x \text{ particles for } \theta^{(i)} \\[4pt]
&\text{2. } \textbf{For } t = 1, \ldots, T \text{ (each new observation } y_t\text{):} \\
&\qquad \text{a. } \textbf{Run inner PFs: } \text{For each } i, \text{ run one step of PF}^{(i)}: \\
&\qquad \qquad \hat{p}(y_t \mid y_{1:t-1}, \theta^{(i)}) = \frac{1}{N_x} \sum_{j=1}^{N_x} \tilde{w}_{t}^{(i,j)} \\[4pt]
&\qquad \text{b. } \textbf{Reweight: } \tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot \hat{p}(y_t \mid y_{1:t-1}, \theta^{(i)}) \\[4pt]
&\qquad \text{c. } \textbf{Normalize: } w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_{j=1}^{N_\theta} \tilde{w}_t^{(j)}} \\[4pt]
&\qquad \text{d. } \textbf{Resample + Rejuvenate: } \text{If } \widehat{\text{ESS}}_t < \tau \cdot N_\theta: \\
&\qquad \qquad \text{(i) Resample parameter particles} \\
&\qquad \qquad \text{(ii) MCMC move: } \theta^{(i)} \to \theta'^{(i)} \text{ with } \pi_t\text{-invariant kernel} \\
&\qquad \qquad \text{(iii) Reset PF}^{(i)} \text{ for new } \theta'^{(i)} \\[4pt]
&\text{3. } \textbf{Output: } \{(\theta^{(i)}, w_T^{(i)}, \text{PF}^{(i)})\}_{i=1}^{N_\theta}
\end{aligned}
}
$$

### The Rejuvenation Step

When ESS drops below threshold, parameter particles are rejuvenated via a Particle MCMC move. For each resampled particle $\theta^{(i)}$:

1. Propose $\theta' \sim q(\cdot \mid \theta^{(i)})$
2. Run a **new** particle filter with $N_x$ particles under $\theta'$ to get $\hat{p}(y_{1:t} \mid \theta')$
3. Accept with probability:

$$
\alpha = \min\left(1, \frac{p(\theta') \cdot \hat{p}(y_{1:t} \mid \theta')}{p(\theta^{(i)}) \cdot \hat{p}(y_{1:t} \mid \theta^{(i)})} \cdot \frac{q(\theta^{(i)} \mid \theta')}{q(\theta' \mid \theta^{(i)})}\right)
$$

!!! warning "Computational cost of rejuvenation"
    Each rejuvenation step requires running a full particle filter from $t=1$ to the current time $t$ for each proposed $\theta'$. This cost grows linearly with $t$, making rejuvenation increasingly expensive as more data is observed.

---

## API Reference

### Constructor

```python
from particlefilterbox.smc import SMCSquared, SMCSquaredConfig

config = SMCSquaredConfig(
    n_theta=200,          # outer SMC particles (parameters)
    n_x=500,              # inner PF particles (states)
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
)

smc2 = SMCSquared(model=my_ssm, config=config)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_theta` | `int` | `200` | Number of parameter particles $N_\theta$ |
| `n_x` | `int` | `500` | Number of state particles $N_x$ per parameter |
| `resampling` | `str` | `"systematic"` | Resampling scheme for outer SMC |
| `ess_threshold` | `float` | `0.5` | ESS threshold for triggering rejuvenation |
| `n_mcmc_moves` | `int` | `3` | MCMC moves per rejuvenation |
| `inner_filter` | `str` | `"bootstrap"` | Filter type for inner PF: `"bootstrap"`, `"auxiliary"` |

### Running

```python
result = smc2.filter(observations)
```

### Result Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `theta_particles` | `(N_\theta, dim_\theta)` | Parameter particles |
| `theta_weights` | `(N_\theta,)` | Parameter weights |
| `log_marginal_likelihood` | scalar | $\log \hat{p}(y_{1:T})$ |
| `ess_history` | `(T,)` | Parameter ESS at each time step |
| `rejuvenation_times` | list | Time steps where rejuvenation occurred |
| `filtered_means` | `(T, dim_x)` | Mixture-averaged filtered state means |

---

## Examples

### Example: Online Parameter Estimation for Stochastic Volatility

Estimating the parameters $(\phi, \sigma, \beta)$ of a stochastic volatility model as data arrives:

```python
import numpy as np
from particlefilterbox.smc import SMCSquared, SMCSquaredConfig
from particlefilterbox.models import StochasticVolatility

# --- Simulate data ---
true_params = {"phi": 0.98, "sigma": 0.16, "beta": 0.65}
sv_true = StochasticVolatility(**true_params)

rng = np.random.default_rng(42)
T = 300
x_true, y_obs = sv_true.simulate(T, rng=rng)

# --- Define model with parameter priors ---
model = StochasticVolatility(
    phi_prior=("beta", 20, 1.5),        # Beta(20, 1.5) -> mode near 0.95
    sigma_prior=("half_normal", 0.5),    # Half-Normal(0.5)
    beta_prior=("half_normal", 1.0),     # Half-Normal(1.0)
)

# --- Run SMC² ---
config = SMCSquaredConfig(
    n_theta=200,
    n_x=500,
    ess_threshold=0.5,
    seed=42,
)

smc2 = SMCSquared(model=model, config=config)
result = smc2.filter(y_obs)

# --- Posterior parameter estimates ---
w = result.theta_weights
theta = result.theta_particles

param_names = ["phi", "sigma", "beta"]
true_values = [0.98, 0.16, 0.65]

print("Parameter | True  | Post. Mean | Post. Std")
print("-" * 47)
for j, (name, true_val) in enumerate(zip(param_names, true_values)):
    post_mean = np.average(theta[:, j], weights=w)
    post_std = np.sqrt(np.average((theta[:, j] - post_mean)**2, weights=w))
    print(f"  {name:7s} | {true_val:5.3f} | {post_mean:10.4f} | {post_std:9.4f}")

print(f"\nLog marginal likelihood: {result.log_marginal_likelihood:.2f}")
print(f"Rejuvenation steps: {len(result.rejuvenation_times)}")
print(f"Mean ESS: {result.ess_history.mean():.0f} / {config.n_theta}")
```

### Tracking Parameter Evolution

SMC$^2$ provides online parameter estimates. To track how parameter beliefs evolve:

```python
# --- Online parameter tracking ---
# Run step-by-step to record parameter evolution
smc2 = SMCSquared(model=model, config=config)
cloud = smc2.initialize(rng)

theta_means = np.zeros((T, 3))
theta_stds = np.zeros((T, 3))

for t in range(T):
    cloud = smc2.step(cloud, y_obs[t], t)
    w = cloud.theta_weights
    theta = cloud.theta_particles

    theta_means[t] = np.average(theta, weights=w, axis=0)
    theta_stds[t] = np.sqrt(
        np.average((theta - theta_means[t])**2, weights=w, axis=0)
    )

# theta_means[t, :] contains the online parameter estimates at time t
# theta_stds[t, :] contains the posterior uncertainty at time t
```

!!! tip "Convergence"
    Parameter estimates typically stabilize after observing enough data --- for a 3-parameter SV model, expect reasonable estimates after $T \approx 100$--$200$ observations. The posterior standard deviation should decrease roughly as $O(1/\sqrt{T})$.

---

## Computational Cost

SMC$^2$ is the most computationally expensive method in this section. Understanding the cost helps with budget allocation:

| Component | Cost per time step | Notes |
|-----------|:------------------:|-------|
| Inner PFs (filtering) | $O(N_\theta \cdot N_x)$ | $N_\theta$ particle filters, each with $N_x$ particles |
| Reweighting | $O(N_\theta)$ | One likelihood per parameter particle |
| Rejuvenation | $O(N_\theta \cdot N_x \cdot t)$ | Only at rejuvenation times; cost grows with $t$ |
| **Total (no rejuvenation)** | $O(N_\theta \cdot N_x)$ | |
| **Total (with rejuvenation)** | $O(N_\theta \cdot N_x \cdot t)$ | Dominates when triggered |

### Practical Budget Guidelines

| Scenario | $N_\theta$ | $N_x$ | Approx. cost per step |
|----------|:----------:|:------:|:---------------------:|
| Quick exploration | 50 | 200 | $10^4$ operations |
| Standard estimation | 200 | 500 | $10^5$ operations |
| High accuracy | 500 | 1000 | $5 \times 10^5$ operations |

!!! warning "Memory"
    Each parameter particle stores its own particle filter history. With $N_\theta = 200$ and $N_x = 500$, memory usage scales as $O(N_\theta \cdot N_x \cdot \text{dim}(x))$. For high-dimensional states, consider [IBIS](ibis.md) if the state can be analytically integrated out.

---

## SMC$^2$ vs. Alternatives

| Feature | SMC$^2$ | [IBIS](ibis.md) | [PMMH](../pmcmc/index.md) |
|---------|:-------:|:----------------:|:-------------------------:|
| Online | **Yes** | **Yes** | No |
| Handles latent states | **Yes** | Only if integrable | **Yes** |
| Parallelizable | **Yes** | **Yes** | Limited |
| Marginal likelihood | **Yes** | **Yes** | No |
| Computational cost | High | Lower | Medium |
| Ease of use | Moderate | Easy | Easy |

!!! tip "When to use IBIS instead"
    If the latent states can be integrated out analytically (e.g., linear Gaussian state dynamics), [IBIS](ibis.md) achieves similar results at a fraction of the cost because it avoids running inner particle filters.

---

## See Also

- [IBIS](ibis.md) --- a lighter alternative when states can be marginalized
- [SMC Sampler](smc-sampler.md) --- the general SMC framework that SMC$^2$ builds upon
- [Waste-Free SMC](waste-free.md) --- can be applied to the outer SMC layer to reduce waste

---

## References

- Chopin, N., Jacob, P.E. & Papaspiliopoulos, O. (2013). SMC$^2$: An Efficient Algorithm for Sequential Analysis of State-Space Models. *Journal of the Royal Statistical Society: Series B*, 75(3), 397--426.
- Fulop, A. & Li, J. (2013). Efficient Learning via Simulation: A Marginalized Resample-Move Approach. *Journal of Econometrics*, 176(2), 146--161.
- Doucet, A., Pitt, M.K., Deligiannidis, G. & Kohn, R. (2015). Efficient Implementation of Markov Chain Monte Carlo when Using an Unbiased Likelihood Estimator. *Biometrika*, 102(2), 295--313.
