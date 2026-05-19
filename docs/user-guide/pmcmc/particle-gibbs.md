---
title: Particle Gibbs
description: Particle Gibbs sampler -- conditional SMC for joint inference over parameters and latent states
---

# Particle Gibbs Sampler

The **Particle Gibbs** (PG) sampler extends the PMCMC framework to sample **jointly** from the posterior over parameters $\theta$ and latent states $x_{0:T}$. It alternates between updating parameters given states and updating states given parameters, using a **conditional SMC** algorithm for the state update.

---

## The Algorithm

Particle Gibbs is a Gibbs sampler with two blocks:

1. **Update parameters**: sample $\theta^{(m)} \sim p(\theta \mid x_{0:T}^{(m-1)}, y_{1:T})$
2. **Update states**: sample $x_{0:T}^{(m)} \sim p(x_{0:T} \mid \theta^{(m)}, y_{1:T})$ using conditional SMC

The state update in step 2 is the key innovation: instead of running a standard particle filter, we run a **conditional** particle filter that forces one particle to follow the reference trajectory $x_{0:T}^{(m-1)}$ from the previous iteration.

### Pseudocode

$$
\boxed{
\begin{aligned}
&\textbf{Algorithm: Particle Gibbs} \\
&\textbf{Input:} \text{ observations } y_{1:T}, \text{ prior } p(\theta), \text{ number of particles } N \\
&\text{1. Initialize } \theta^{(0)}, \; x_{0:T}^{(0)} \text{ (e.g., from a pilot PF run)} \\
&\text{2. For } m = 1, \ldots, M: \\
&\quad \text{a. Sample } \theta^{(m)} \sim p(\theta \mid x_{0:T}^{(m-1)}, y_{1:T}) \\
&\quad \text{b. Run conditional SMC}\bigl(y_{1:T}, \theta^{(m)}, x_{0:T}^{(m-1)}, N\bigr): \\
&\quad\quad \text{i. Set } x_{0:T}^{(1)} = x_{0:T}^{(m-1)} \text{ (reference particle)} \\
&\quad\quad \text{ii. For } i = 2, \ldots, N: \text{ sample } x_0^{(i)} \sim p(x_0 \mid \theta^{(m)}) \\
&\quad\quad \text{iii. For } t = 1, \ldots, T: \\
&\quad\quad\quad \bullet \text{ For } i = 2, \ldots, N: \text{ propagate } x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)}, \theta^{(m)}) \\
&\quad\quad\quad \bullet \text{ Particle 1 follows: } x_t^{(1)} = x_t^{(m-1)} \\
&\quad\quad\quad \bullet \text{ Compute weights } w_t^{(i)} \propto p(y_t \mid x_t^{(i)}, \theta^{(m)}) \\
&\quad\quad\quad \bullet \text{ Resample (keeping particle 1 alive)} \\
&\quad\quad \text{iv. Sample trajectory } x_{0:T}^{(m)} \text{ from final particle set} \\
&\text{3. Return } \{(\theta^{(m)}, x_{0:T}^{(m)})\}_{m=1}^{M}
\end{aligned}
}
$$

---

## Conditional SMC: The Key Idea

In a standard particle filter, all $N$ particles are sampled freely. In **conditional SMC**, one particle (the reference) is **fixed** to follow a pre-specified trajectory $x_{0:T}^{\text{ref}}$, while the remaining $N - 1$ particles are sampled as usual.

```
Standard PF (all particles free):        Conditional PF (one particle fixed):

  t=0    t=1    t=2    t=3                 t=0    t=1    t=2    t=3
   ●──────●──────●──────●                  ★══════★══════★══════★  ← reference
   ●──────●──────●──────●                  ●──────●──────●──────●
   ●──────●──────●──────●                  ●──────●──────●──────●
   ●──────●──────●──────●                  ●──────●──────●──────●
   ●──────●──────●──────●                  ●──────●──────●──────●
```

The reference trajectory is never killed during resampling -- it always survives to the next time step. This ensures that the conditional PF produces a **valid Gibbs update** for the state trajectory.

!!! note "Why fix a reference particle?"
    Without the reference particle, the conditional distribution $p(x_{0:T} \mid \theta, y_{1:T})$
    is intractable -- we cannot sample from it directly. The conditional SMC trick provides
    a valid MCMC kernel that targets this distribution while being computationally tractable.

---

## API

```python
from particlefilterbox.pmcmc import ParticleGibbs

pg = ParticleGibbs(
    model,                    # StateSpaceModel instance
    n_particles=500,          # particles per conditional PF run
    n_iterations=5000,        # total Gibbs iterations
    burnin=1000,              # discard first 1000 samples
    thin=1,                   # thinning interval
    seed=42                   # random seed
)

chain = pg.sample(observations)
```

### Accessing Results

```python
# Parameter posterior
print(chain.summary())                # posterior summaries for theta
chain.plot_trace()                    # trace plots

# State trajectories
states = chain.states                 # (n_iterations, T, dim_x) array
mean_trajectory = chain.states_mean   # posterior mean of x_{0:T}
quantiles = chain.states_quantile([0.05, 0.95])  # credible bands
```

---

## Example: Stochastic Volatility with Latent States

```python
import numpy as np
from particlefilterbox.models import SVModel
from particlefilterbox.pmcmc import ParticleGibbs

# True parameters
model = SVModel(mu=0.0, phi=0.97, sigma=0.15)
np.random.seed(42)
y, h_true = model.simulate(T=500, return_states=True)

# Run Particle Gibbs
pg = ParticleGibbs(
    model,
    n_particles=500,
    n_iterations=8000,
    burnin=3000,
    priors={
        'mu':    ('normal', 0.0, 1.0),
        'phi':   ('beta', 20.0, 1.5),
        'sigma': ('half_cauchy', 0.0, 1.0),
    }
)

chain = pg.sample(y)

# Parameter estimates
print(chain.summary())
```

```text
Parameter    Mean     Std     2.5%    97.5%    ESS    R-hat
---------  ------  ------  ------  -------  -----  -------
mu         -0.005   0.210  -0.415    0.410   2100    1.002
phi         0.966   0.013   0.938    0.988   1800    1.003
sigma       0.160   0.026   0.114    0.217   1500    1.004
```

```python
# Smoothed states with credible bands
import matplotlib.pyplot as plt

h_mean = chain.states_mean[:, 0]
h_lower, h_upper = chain.states_quantile([0.05, 0.95])

plt.figure(figsize=(12, 4))
plt.fill_between(range(len(h_mean)), h_lower[:, 0], h_upper[:, 0],
                 alpha=0.3, label='90% credible band')
plt.plot(h_mean, label='Posterior mean', linewidth=1)
plt.plot(h_true, '--', label='True states', linewidth=0.8, alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Log-volatility $h_t$')
plt.legend()
plt.title('Particle Gibbs: Smoothed Latent States')
plt.show()
```

---

## Tuning and Diagnostics

### Number of Particles

Unlike PMMH, where particle count controls likelihood variance, in Particle Gibbs the particle count controls the **mixing quality** of the state trajectory update:

| $N$ | Mixing quality | Cost per iteration |
|---|---|---|
| 100 | Poor -- reference trajectory dominates | Low |
| 500 | Moderate -- some trajectory diversity | Medium |
| 1000 | Good -- new trajectory often different from reference | High |
| 2000 | Excellent -- for long time series | Very high |

!!! tip "Rule of thumb"
    Start with $N = 5 \sqrt{T}$ where $T$ is the series length. Increase if the
    state trajectory shows poor mixing (new trajectories rarely differ from the reference).

### Monitoring Mixing

```python
# Check parameter mixing
print(f"Parameter ESS: {chain.ess}")
print(f"Acceptance rate: {chain.acceptance_rate:.2%}")

# Check state trajectory mixing
# Compute fraction of time steps where x_t changed from previous iteration
trajectory_change_rate = chain.trajectory_change_rate
print(f"Mean trajectory change rate: {trajectory_change_rate:.2%}")
```

| Diagnostic | Target | Action if poor |
|---|---|---|
| **Parameter ESS** | > 1000 | Reparameterize or run longer |
| **Trajectory change rate** | > 50% | Increase $N$ |
| **State ESS** | > 500 per time step | Increase $N$ or switch to PGAS |

---

## Path Degeneracy

The main limitation of Particle Gibbs is **path degeneracy**: for long time series, the conditional particle filter tends to produce trajectories that are identical to the reference for the early time steps, with new particles only diverging near the end.

```
Conditional PF with path degeneracy (T = 100):

  t=0       t=20      t=40      t=60      t=80      t=100
   ★─────────★─────────★─────────★─────────★─────────★  ← reference
   ●─────────●─────────●─●───────●─●───────●─●───●───●
   ●─────────●─────────●─●───────●─●───────●─●───●───●
                                        ↑
                              new particles only diverge here
```

This happens because the resampling steps cause all particle genealogies to **coalesce** backward in time -- eventually, all particles share the same ancestor at $t = 0$. In the conditional PF, this means the reference particle's early history dominates.

### Consequences

- **Slow mixing of early states**: $x_0, x_1, \ldots$ change very slowly across iterations
- **Autocorrelation**: high autocorrelation in the state chain for early time steps
- **Worsens with $T$**: path degeneracy is more severe for longer series

### Mitigation

| Strategy | Effectiveness | Notes |
|---|---|---|
| **Increase $N$** | Moderate | Helps but $N$ must grow with $T$ |
| **Use PGAS** | Strong | Ancestor sampling resolves the issue -- see [PGAS](pgas.md) |
| **Backward sampling** | Strong | $O(N \cdot T)$ post-processing step |
| **Block sampling** | Moderate | Update states in blocks rather than full trajectory |

!!! warning "Path degeneracy in practice"
    For time series with $T > 200$, standard Particle Gibbs often requires
    prohibitively many particles to achieve good mixing. In these cases,
    **PGAS is strongly recommended** -- it resolves path degeneracy with
    minimal additional cost.

---

## Comparison with PMMH

| Aspect | PMMH | Particle Gibbs |
|---|---|---|
| **Output** | Parameters only | Parameters + state trajectories |
| **State estimation** | Must run separate smoother | Built-in |
| **Scalability in $T$** | Good (PF variance grows slowly) | Limited by path degeneracy |
| **Scalability in $d_\theta$** | Depends on MH proposal | Depends on conditional structure |
| **Tuning** | Proposal + $N$ | Mainly $N$ |
| **Best for** | Parameter estimation | Joint inference, short series |

---

## What's Next?

<div class="grid cards" markdown>

- :material-arrow-right-bold: **[PGAS](pgas.md)**

    Particle Gibbs with Ancestor Sampling -- resolves path degeneracy

- :material-arrow-left-bold: **[PMMH](pmmh.md)**

    Simpler parameter-only inference via Metropolis-Hastings

- :material-arrow-left-bold: **[PMCMC Overview](index.md)**

    Back to the framework overview and method comparison

</div>
