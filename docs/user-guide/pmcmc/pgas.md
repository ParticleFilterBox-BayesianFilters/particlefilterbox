---
title: Particle Gibbs with Ancestor Sampling (PGAS)
description: PGAS algorithm -- resolving path degeneracy in Particle Gibbs via ancestor sampling for efficient joint inference
---

# Particle Gibbs with Ancestor Sampling (PGAS)

**PGAS** (Lindsten, Jordan & Sch&ouml;n, 2014) is an extension of [Particle Gibbs](particle-gibbs.md) that adds an **ancestor sampling** step at each time point. This simple modification dramatically improves mixing by reconnecting the reference trajectory to the particle genealogy, effectively resolving the path degeneracy problem that limits standard Particle Gibbs.

PGAS is the **recommended default** for joint inference over parameters and latent states in particlefilterbox.

---

## The Path Degeneracy Problem (Recap)

In standard Particle Gibbs, the reference trajectory is kept alive throughout the conditional PF. Due to coalescence of particle genealogies, the early portion of the reference trajectory is almost never replaced:

```
Standard Particle Gibbs (path degeneracy):

  t=0       t=25      t=50      t=75      t=100
   ★═════════★═════════★═════════★═════════★  ← reference (early states frozen)
   ●─────────●────●────●────●────●────●────●
   ●─────────●────●────●────●────●────●────●
                                 ↑
                    only recent states get updated
```

This means the Gibbs sampler mixes very slowly for the early latent states.

---

## The Ancestor Sampling Solution

PGAS adds one extra step at each time point $t$: it **resamples the ancestor** of the reference particle. Instead of forcing the reference particle to always descend from itself at $t - 1$, we allow it to "adopt" any particle from the previous time step as its ancestor, with probability proportional to the transition density.

### Pseudocode

$$
\boxed{
\begin{aligned}
&\textbf{Algorithm: PGAS} \\
&\textbf{Input:} \text{ observations } y_{1:T}, \text{ prior } p(\theta), \text{ particles } N, \text{ reference } x_{0:T}^{\text{ref}} \\
&\text{1. Initialize } \theta^{(0)}, \; x_{0:T}^{(0)} \\
&\text{2. For } m = 1, \ldots, M: \\
&\quad \text{a. Sample } \theta^{(m)} \sim p(\theta \mid x_{0:T}^{(m-1)}, y_{1:T}) \\
&\quad \text{b. Run conditional SMC with ancestor sampling:} \\
&\quad\quad \text{i. Set } x_0^{(1)} = x_0^{\text{ref}}, \text{ sample } x_0^{(i)} \sim p(x_0 \mid \theta^{(m)}) \text{ for } i = 2, \ldots, N \\
&\quad\quad \text{ii. For } t = 1, \ldots, T: \\
&\quad\quad\quad \bullet \text{ For } i = 2, \ldots, N: \text{ propagate } x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)}, \theta^{(m)}) \\
&\quad\quad\quad \bullet \text{ Set } x_t^{(1)} = x_t^{\text{ref}} \\
&\quad\quad\quad \bullet \textbf{ Ancestor sampling: draw ancestor } a_t^{(1)} \text{ with probability} \\
&\quad\quad\quad\quad \tilde{w}_{t-1}^{(j)} \propto w_{t-1}^{(j)} \cdot p(x_t^{\text{ref}} \mid x_{t-1}^{(j)}, \theta^{(m)}) \\
&\quad\quad\quad \bullet \text{ Compute weights } w_t^{(i)} \propto p(y_t \mid x_t^{(i)}, \theta^{(m)}) \\
&\quad\quad\quad \bullet \text{ Resample (keeping particle 1 alive)} \\
&\quad\quad \text{iii. Sample trajectory } x_{0:T}^{(m)} \text{ from final particle set} \\
&\text{3. Return } \{(\theta^{(m)}, x_{0:T}^{(m)})\}_{m=1}^{M}
\end{aligned}
}
$$

### How Ancestor Sampling Works

At each time step $t$, instead of the reference particle always inheriting its own history, we sample a new ancestor $a_t^{(1)}$ from among all $N$ particles at time $t - 1$:

$$
P(a_t^{(1)} = j) \propto w_{t-1}^{(j)} \cdot p\!\left(x_t^{\text{ref}} \mid x_{t-1}^{(j)}, \theta\right)
$$

This probability balances two factors:

- **Weight** $w_{t-1}^{(j)}$: prefer high-quality particles
- **Transition density** $p(x_t^{\text{ref}} \mid x_{t-1}^{(j)}, \theta)$: prefer particles that could plausibly have generated the reference state at time $t$

```
PGAS (ancestor sampling reconnects history):

  t=0       t=25      t=50      t=75      t=100
   ●────●────●────●────●═══●════★═════════★  ← reference adopts new ancestors
   ●────●────●────●────●────●────●────●────●
   ●═══●════●════●═══●─●────●────●────●────●
   ↑                   ↑
   new ancestor        new ancestor
   at t=5              at t=48
```

The reference trajectory's history is **continuously refreshed** through ancestor sampling, preventing the frozen-history problem.

---

## API

```python
from particlefilterbox.pmcmc import PGAS

pgas = PGAS(
    model,                    # StateSpaceModel instance
    n_particles=200,          # fewer particles needed than PG!
    n_iterations=5000,        # total Gibbs iterations
    burnin=1000,              # discard first 1000 samples
    thin=1,                   # thinning interval
    seed=42                   # random seed
)

chain = pgas.sample(observations)
```

!!! tip "Fewer particles than Particle Gibbs"
    Because ancestor sampling resolves path degeneracy, PGAS typically requires
    **2--5x fewer particles** than standard Particle Gibbs for the same mixing quality.
    Start with $N = 100$--$200$ and increase only if needed.

---

## Example: SV Model -- PGAS vs Particle Gibbs

This example demonstrates the superior mixing of PGAS compared to standard Particle Gibbs on a stochastic volatility model.

```python
import numpy as np
from particlefilterbox.models import SVModel
from particlefilterbox.pmcmc import ParticleGibbs, PGAS

# Generate data
model = SVModel(mu=0.0, phi=0.97, sigma=0.15)
np.random.seed(42)
y, h_true = model.simulate(T=500, return_states=True)

priors = {
    'mu':    ('normal', 0.0, 1.0),
    'phi':   ('beta', 20.0, 1.5),
    'sigma': ('half_cauchy', 0.0, 1.0),
}

# --- Particle Gibbs (N=500 particles) ---
pg = ParticleGibbs(model, n_particles=500, n_iterations=5000,
                   burnin=1000, priors=priors)
chain_pg = pg.sample(y)

# --- PGAS (N=200 particles -- fewer needed!) ---
pgas = PGAS(model, n_particles=200, n_iterations=5000,
            burnin=1000, priors=priors)
chain_pgas = pgas.sample(y)
```

### Comparing Mixing Quality

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Trace plots for phi
axes[0, 0].plot(chain_pg.get('phi'), alpha=0.5, linewidth=0.3)
axes[0, 0].set_title(f'PG: phi (ESS={chain_pg.ess["phi"]:.0f})')
axes[0, 0].axhline(0.97, color='red', linestyle='--', label='True')

axes[0, 1].plot(chain_pgas.get('phi'), alpha=0.5, linewidth=0.3)
axes[0, 1].set_title(f'PGAS: phi (ESS={chain_pgas.ess["phi"]:.0f})')
axes[0, 1].axhline(0.97, color='red', linestyle='--', label='True')

# State trajectory at t=10 (early time step -- tests path degeneracy)
axes[1, 0].plot(chain_pg.states[:, 10, 0], alpha=0.5, linewidth=0.3)
axes[1, 0].set_title(f'PG: h_10 (ESS={chain_pg.state_ess[10]:.0f})')
axes[1, 0].axhline(h_true[10], color='red', linestyle='--')

axes[1, 1].plot(chain_pgas.states[:, 10, 0], alpha=0.5, linewidth=0.3)
axes[1, 1].set_title(f'PGAS: h_10 (ESS={chain_pgas.state_ess[10]:.0f})')
axes[1, 1].axhline(h_true[10], color='red', linestyle='--')

plt.suptitle('Particle Gibbs vs PGAS: Mixing Comparison')
plt.tight_layout()
plt.show()
```

### Expected Results

```text
                    Particle Gibbs (N=500)    PGAS (N=200)
                    ─────────────────────    ─────────────
phi ESS:                    1800                 3200
sigma ESS:                  1500                 2800
State ESS (t=10):             80                 2500
State ESS (t=250):          1200                 2900
Trajectory change rate:       35%                  92%
Cost per iteration:          High               Moderate
```

!!! note "Key takeaway"
    PGAS with 200 particles achieves **better mixing** than Particle Gibbs with 500 particles,
    at lower computational cost. The improvement is most dramatic for early latent states
    (compare state ESS at $t = 10$: 80 vs 2500).

---

## Tuning and Diagnostics

### Number of Particles

PGAS is remarkably robust to the choice of $N$. Unlike Particle Gibbs, where $N$ must grow with $T$ to combat path degeneracy, PGAS works well with a fixed, moderate $N$:

| Series length $T$ | Recommended $N$ | Notes |
|---|---|---|
| $T < 100$ | 50--100 | Very few particles needed |
| $100 \leq T < 500$ | 100--200 | Default range |
| $500 \leq T < 2000$ | 200--500 | May need more for highly nonlinear models |
| $T > 2000$ | 300--500 | PGAS scales well even for long series |

### Diagnostics Checklist

```python
# 1. Parameter diagnostics
print(chain.summary())
print(f"Parameter ESS: {chain.ess}")

# 2. State trajectory mixing
print(f"Trajectory change rate: {chain.trajectory_change_rate:.2%}")
print(f"State ESS (min/mean/max): "
      f"{chain.state_ess.min():.0f} / "
      f"{chain.state_ess.mean():.0f} / "
      f"{chain.state_ess.max():.0f}")

# 3. Ancestor sampling acceptance
print(f"Ancestor sampling rate: {chain.ancestor_sampling_rate:.2%}")
```

| Diagnostic | Target | Action if poor |
|---|---|---|
| **Parameter ESS** | > 1000 | Run longer or reparameterize |
| **Trajectory change rate** | > 80% | Increase $N$ |
| **State ESS (min)** | > 500 | Increase $N$ |
| **Ancestor sampling rate** | > 30% | Check model transition density |

!!! warning "Low ancestor sampling rate"
    If the ancestor sampling rate is very low (< 10%), it means the transition density
    $p(x_t^{\text{ref}} \mid x_{t-1}^{(j)}, \theta)$ is very small for most particles.
    This can happen when:

    - The transition model has very small noise (states don't move much)
    - The model is highly nonlinear with sharp transitions

    In these cases, increase $N$ or consider using a **locally-optimal proposal** inside
    the conditional PF.

---

## When to Use PGAS

PGAS should be your **default choice** for joint inference over parameters and states. Use it whenever:

- You need **posterior samples of latent states** $x_{0:T}$ (not just parameters)
- The time series is **moderate to long** ($T > 50$)
- You want **good mixing** without extensive tuning

### Decision Guide

| Situation | Recommended method |
|---|---|
| Only need $p(\theta \mid y_{1:T})$ | [PMMH](pmmh.md) |
| Need $p(\theta, x_{0:T} \mid y_{1:T})$, short $T$ | Particle Gibbs or PGAS |
| Need $p(\theta, x_{0:T} \mid y_{1:T})$, any $T$ | **PGAS** |
| Sequential/online parameter learning | SMC$^2$ or IBIS |
| Linear-Gaussian model | Use kalmanbox directly |

!!! tip "PGAS as the default"
    Unless you have a specific reason to use another method, start with PGAS.
    It combines the joint inference capability of Particle Gibbs with mixing quality
    that is comparable to (or better than) PMMH, while requiring fewer particles
    than standard Particle Gibbs.

---

## Advanced: Inside the Ancestor Sampling Step

For readers interested in the technical details, here is exactly what happens during the ancestor sampling step at time $t$.

Given the reference state $x_t^{\text{ref}}$ and the particle set $\{x_{t-1}^{(j)}, w_{t-1}^{(j)}\}_{j=1}^{N}$ at time $t - 1$, we compute ancestor weights:

$$
\tilde{w}_{t-1|t}^{(j)} = w_{t-1}^{(j)} \cdot p\!\left(x_t^{\text{ref}} \mid x_{t-1}^{(j)}, \theta\right)
$$

Normalize these weights and sample the ancestor index:

$$
a_t^{(1)} \sim \text{Categorical}\!\left(\frac{\tilde{w}_{t-1|t}^{(1)}}{\sum_k \tilde{w}_{t-1|t}^{(k)}}, \ldots, \frac{\tilde{w}_{t-1|t}^{(N)}}{\sum_k \tilde{w}_{t-1|t}^{(k)}}\right)
$$

The reference particle at time $t$ then inherits the **full history** of particle $a_t^{(1)}$ up to time $t - 1$, followed by $x_t^{\text{ref}}$ at time $t$. This creates a new trajectory that combines the reference's future with a potentially different particle's past.

!!! info "Computational cost"
    Ancestor sampling requires evaluating the transition density $p(x_t^{\text{ref}} \mid x_{t-1}^{(j)}, \theta)$
    for all $N$ particles at each time step. This adds $O(N \cdot T)$ operations per iteration --
    the same order as the conditional PF itself. In practice, the overhead is negligible.

---

## What's Next?

<div class="grid cards" markdown>

- :material-arrow-left-bold: **[Particle Gibbs](particle-gibbs.md)**

    Standard Particle Gibbs without ancestor sampling

- :material-arrow-left-bold: **[PMMH](pmmh.md)**

    Parameter-only inference via Metropolis-Hastings

- :material-arrow-left-bold: **[PMCMC Overview](index.md)**

    Back to the framework overview and method comparison

</div>
