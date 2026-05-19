---
title: "SMC\xB2 Online"
description: "SMC\xB2 as an online PMCMC method -- sequential Bayesian parameter estimation with particle rejuvenation"
---

# SMC$^2$ as Online PMCMC

While [PMMH](pmmh.md) and [Particle Gibbs](particle-gibbs.md) are **batch** methods that process the entire dataset at each iteration, **SMC$^2$** (Chopin, Jacob, & Papaspiliopoulos, 2013) provides a fully **online** approach: it processes observations one at a time, updating the parameter posterior sequentially. When particle degeneracy is detected, it rejuvenates the parameter particles using PMMH moves.

This makes SMC$^2$ the natural choice for **streaming data** and **real-time** applications where waiting for a full batch MCMC run is not feasible.

---

## The Idea

SMC$^2$ maintains two layers of particles:

- **$\theta$-particles** ($N_\theta$): represent the parameter posterior $p(\theta \mid y_{1:t})$
- **$x$-particles** ($N_x$ per $\theta$-particle): each $\theta$-particle runs its own particle filter to track $p(x_{0:t} \mid \theta, y_{1:t})$

As new observations arrive, both layers are updated sequentially:

```
                 θ-particle 1          θ-particle 2         ...    θ-particle N_θ
                 ┌──────────┐          ┌──────────┐                ┌──────────┐
    y_1  ───►    │ PF (N_x) │          │ PF (N_x) │                │ PF (N_x) │
    y_2  ───►    │ PF (N_x) │          │ PF (N_x) │                │ PF (N_x) │
    ...          │   ...     │          │   ...     │                │   ...     │
    y_t  ───►    │ PF (N_x) │          │ PF (N_x) │                │ PF (N_x) │
                 └──────────┘          └──────────┘                └──────────┘
                   w_t^(1)               w_t^(2)                     w_t^(N_θ)
```

Each $\theta$-particle's weight is its **marginal likelihood** estimate $\hat{p}(y_{1:t} \mid \theta^{(j)})$, computed by its internal particle filter.

---

## The Algorithm

### Sequential Update

When a new observation $y_t$ arrives:

$$
\boxed{
\begin{aligned}
&\textbf{Algorithm: SMC}^2 \textbf{ -- Online Step} \\
&\textbf{Input:} \text{ new observation } y_t, \text{ current } \theta\text{-particles } \{\theta^{(j)}, \text{PF}^{(j)}\}_{j=1}^{N_\theta} \\
&\text{1. For each } \theta\text{-particle } j = 1, \ldots, N_\theta: \\
&\quad \text{a. Run one step of PF}^{(j)}(y_t, \theta^{(j)}) \\
&\quad \text{b. Update incremental weight: } w_t^{(j)} \propto w_{t-1}^{(j)} \cdot \hat{p}(y_t \mid y_{1:t-1}, \theta^{(j)}) \\
&\text{2. Compute ESS of } \theta\text{-particles:} \\
&\quad \text{ESS}_t = \frac{\left(\sum_j w_t^{(j)}\right)^2}{\sum_j \left(w_t^{(j)}\right)^2} \\
&\text{3. If } \text{ESS}_t < \text{ESS}_{\text{threshold}} \cdot N_\theta: \\
&\quad \text{a. Resample } \theta\text{-particles} \\
&\quad \text{b. Rejuvenate via PMMH moves (see below)} \\
&\text{4. Return updated particles and weights}
\end{aligned}
}
$$

### Rejuvenation via PMMH Moves

When the ESS drops below the threshold, the $\theta$-particles have degenerated -- too few particles carry significant weight. Rejuvenation diversifies them using PMMH:

$$
\boxed{
\begin{aligned}
&\textbf{Rejuvenation Step} \\
&\text{For each } \theta\text{-particle } j = 1, \ldots, N_\theta: \\
&\quad \text{For } r = 1, \ldots, R \text{ (rejuvenation moves):} \\
&\quad\quad \text{a. Propose } \theta' \sim q(\theta' \mid \theta^{(j)}) \\
&\quad\quad \text{b. Run a fresh PF}(y_{1:t}, \theta', N_x) \to \hat{p}' \\
&\quad\quad \text{c. Accept/reject with MH ratio:} \\
&\quad\quad\quad \alpha = \min\!\left(1, \;\frac{\hat{p}' \cdot p(\theta')}{\hat{p}^{(j)} \cdot p(\theta^{(j)})} \cdot \frac{q(\theta^{(j)} \mid \theta')}{q(\theta' \mid \theta^{(j)})}\right) \\
&\quad\quad \text{d. If accepted: } \theta^{(j)} \leftarrow \theta', \; \text{PF}^{(j)} \leftarrow \text{new PF}
\end{aligned}
}
$$

!!! note "Computational cost of rejuvenation"
    Each rejuvenation move requires running a **full** particle filter over $y_{1:t}$, which costs $O(t \cdot N_x)$. As $t$ grows, rejuvenation becomes increasingly expensive. The ESS threshold controls how often this happens -- a higher threshold triggers more frequent but shorter rejuvenations.

---

## API

```python
from particlefilterbox.pmcmc import SMC2Online

smc2 = SMC2Online(
    model,                      # StateSpaceModel instance
    n_theta=200,                # number of θ-particles
    n_x=500,                    # particles per internal PF
    ess_threshold=0.5,          # rejuvenation trigger (fraction of N_θ)
    n_rejuvenation_moves=5,     # PMMH moves per rejuvenation
    proposal='adaptive',        # proposal for rejuvenation moves
    priors={
        'mu':    ('normal', 0.0, 1.0),
        'phi':   ('beta', 20.0, 1.5),
        'sigma': ('half_cauchy', 0.0, 1.0),
    },
    seed=42
)
```

### Online Processing

```python
# Process observations one at a time
for t, y_t in enumerate(observations):
    smc2.step(y_t)

    # Current parameter posterior estimate
    if (t + 1) % 100 == 0:
        print(f"t={t+1}: {smc2.summary()}")
```

### Batch Convenience

```python
# Or process all at once (still sequential internally)
smc2.run(observations)

# Access results
print(smc2.summary())
```

---

## Adaptive Number of Particles

A key advantage of SMC$^2$ is the ability to **adapt $N_x$** during the run. As the time series grows, the internal particle filters may need more particles to maintain accurate likelihood estimates.

### Automatic Adaptation

```python
smc2 = SMC2Online(
    model,
    n_theta=200,
    n_x=200,                    # initial N_x
    adaptive_n_x=True,          # enable adaptive N_x
    n_x_max=2000,               # upper bound on N_x
    ess_threshold=0.5,
    seed=42
)
```

When `adaptive_n_x=True`, SMC$^2$ monitors the variance of the log-likelihood estimates across $\theta$-particles. If the variance exceeds a threshold, $N_x$ is increased:

$$
\text{If } \; \text{Var}_j\bigl[\log \hat{p}(y_{1:t} \mid \theta^{(j)})\bigr] > \tau, \quad \text{then } N_x \leftarrow \lceil c \cdot N_x \rceil
$$

where $c > 1$ is a growth factor (default 1.5) and $\tau$ is the variance threshold (default 3.0).

!!! tip "When to enable adaptive $N_x$"
    Adaptive $N_x$ is most useful for:

    - **Long time series** ($T > 500$) where fixed $N_x$ may become insufficient
    - **Non-stationary models** where the filtering difficulty changes over time
    - **Exploratory analysis** where you don't know the right $N_x$ in advance

    For short series or well-understood models, a fixed $N_x$ is simpler and avoids the overhead of adaptation.

### Manual Adaptation

You can also monitor and adapt manually:

```python
for t, y_t in enumerate(observations):
    smc2.step(y_t)

    # Monitor likelihood variance
    if smc2.log_likelihood_variance > 5.0:
        smc2.increase_n_x(factor=2.0)
        print(f"t={t}: increased N_x to {smc2.n_x}")
```

---

## Example: Online Parameter Estimation in Streaming

A stochastic volatility model where observations arrive in real time:

```python
import numpy as np
from particlefilterbox.models import SVModel
from particlefilterbox.pmcmc import SMC2Online

# True model
true_model = SVModel(mu=0.0, phi=0.97, sigma=0.15)
np.random.seed(42)
y_stream, h_true = true_model.simulate(T=2000, return_states=True)

# Set up SMC² for online estimation
smc2 = SMC2Online(
    true_model,
    n_theta=200,
    n_x=300,
    ess_threshold=0.5,
    n_rejuvenation_moves=5,
    adaptive_n_x=True,
    priors={
        'mu':    ('normal', 0.0, 1.0),
        'phi':   ('beta', 20.0, 1.5),
        'sigma': ('half_cauchy', 0.0, 1.0),
    },
    seed=42
)

# Process data sequentially, tracking posterior evolution
posterior_means = []
for t, y_t in enumerate(y_stream):
    smc2.step(y_t)
    posterior_means.append(smc2.theta_mean)

    if (t + 1) % 500 == 0:
        print(f"\n--- t = {t+1} ---")
        print(smc2.summary())
        print(f"Rejuvenations so far: {smc2.n_rejuvenations}")
        print(f"Current N_x: {smc2.n_x}")
```

```text
--- t = 500 ---
Parameter    Mean     Std     2.5%    97.5%
---------  ------  ------  ------  -------
mu         -0.032   0.312  -0.648    0.581
phi         0.958   0.021   0.912    0.991
sigma       0.172   0.041   0.102    0.261
Rejuvenations so far: 3
Current N_x: 300

--- t = 1000 ---
Parameter    Mean     Std     2.5%    97.5%
---------  ------  ------  ------  -------
mu          0.008   0.225  -0.434    0.451
phi         0.965   0.014   0.935    0.989
sigma       0.159   0.028   0.110    0.220
Rejuvenations so far: 7
Current N_x: 450
```

### Visualizing Posterior Evolution

```python
import matplotlib.pyplot as plt

posterior_means = np.array(posterior_means)
param_names = ['mu', 'phi', 'sigma']
true_values = [0.0, 0.97, 0.15]

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for i, (ax, name, true_val) in enumerate(zip(axes, param_names, true_values)):
    ax.plot(posterior_means[:, i], label=f'SMC² estimate', linewidth=0.8)
    ax.axhline(true_val, color='red', linestyle='--', label='True value')
    ax.set_ylabel(name)
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Observations processed')
fig.suptitle('SMC²: Online Parameter Learning')
plt.tight_layout()
plt.show()
```

---

## Comparison with Batch PMMH

| Aspect | Batch PMMH | SMC$^2$ Online |
|---|---|---|
| **Data processing** | Full dataset each iteration | One observation at a time |
| **Output** | Posterior samples $\{\theta^{(m)}\}$ | Weighted particle set $\{(\theta^{(j)}, w^{(j)})\}$ |
| **Real-time capable** | No | Yes |
| **Cost per observation** | $O(M \cdot T \cdot N)$ | $O(N_\theta \cdot N_x)$ (amortized) |
| **Rejuvenation cost** | N/A | $O(R \cdot t \cdot N_x)$ per event |
| **Adaptive $N$** | Must restart | Built-in |
| **When to use** | Fixed dataset, need exact posterior | Streaming data, need online estimates |

!!! tip "Choosing between batch and online"
    - Use **batch PMMH** when you have a fixed dataset and want the most accurate posterior samples with convergence guarantees
    - Use **SMC$^2$ Online** when data arrives sequentially, you need real-time parameter updates, or you want to avoid the cost of re-processing the full dataset at each iteration

### Accuracy Comparison

For a fixed dataset of length $T$, running batch PMMH for enough iterations will generally produce a more accurate posterior than SMC$^2$ with the same computational budget. However, SMC$^2$ has two advantages:

1. **Anytime estimates**: you get posterior approximations at every time step, not just at the end
2. **Natural parallelism**: the $N_\theta$ particle filters are independent and can run in parallel

```python
# Batch PMMH for comparison
from particlefilterbox.pmcmc import PMMH

pmmh = PMMH(
    model,
    n_particles=500,
    n_iterations=10000,
    proposal='adaptive',
    burnin=2000,
    priors={
        'mu':    ('normal', 0.0, 1.0),
        'phi':   ('beta', 20.0, 1.5),
        'sigma': ('half_cauchy', 0.0, 1.0),
    }
)

# Batch: must process all data at once
batch_chain = pmmh.sample(y_stream)
print("Batch PMMH:")
print(batch_chain.summary())

# Online: already processed sequentially above
print("\nSMC² Online:")
print(smc2.summary())
```

---

## Diagnostics

### Monitoring ESS

The ESS of the $\theta$-particles is the primary diagnostic. Track it over time to understand when and how often rejuvenation is triggered:

```python
# ESS history
ess_history = smc2.ess_history

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 3))
plt.plot(ess_history, linewidth=0.8)
plt.axhline(smc2.ess_threshold * smc2.n_theta, color='red',
            linestyle='--', label='Rejuvenation threshold')
plt.xlabel('Time step')
plt.ylabel('ESS')
plt.title('θ-particle ESS over time')
plt.legend()
plt.show()
```

### Rejuvenation Acceptance Rates

```python
# Acceptance rates during rejuvenation events
for i, (t, acc_rate) in enumerate(smc2.rejuvenation_log):
    print(f"Rejuvenation {i+1} at t={t}: acceptance rate = {acc_rate:.2%}")
```

!!! warning "Frequent rejuvenation"
    If rejuvenation occurs at nearly every time step:

    1. **Increase $N_\theta$**: more $\theta$-particles maintain diversity longer
    2. **Increase $N_x$**: better likelihood estimates lead to more stable weights
    3. **Check priors**: overly vague priors may cause weight degeneracy

---

## What's Next?

<div class="grid cards" markdown>

- :material-arrow-right-bold: **[Tuning Guide](tuning.md)**

    Complete guide to tuning all PMCMC methods, including SMC$^2$

- :material-arrow-left-bold: **[Conditional SMC](conditional-smc.md)**

    The CSMC kernel used inside Particle Gibbs and PGAS

- :material-arrow-left-bold: **[PMCMC Overview](index.md)**

    Back to the framework overview and method comparison

</div>
