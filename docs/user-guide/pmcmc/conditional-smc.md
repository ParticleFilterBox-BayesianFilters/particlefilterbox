---
title: Conditional Sequential Monte Carlo
description: Conditional SMC algorithm -- particle filtering with a fixed reference trajectory for use in Particle Gibbs and PGAS
---

# Conditional Sequential Monte Carlo

**Conditional SMC** (CSMC) is a particle filter in which one particle is forced to follow a pre-specified **reference trajectory** $x_{0:T}^{\text{ref}}$, while the remaining $N - 1$ particles evolve freely. This seemingly simple modification is the engine behind both [Particle Gibbs](particle-gibbs.md) and [PGAS](pgas.md), providing a valid MCMC kernel that targets the smoothing distribution $p(x_{0:T} \mid \theta, y_{1:T})$.

---

## Why Conditional SMC?

In standard particle filtering, we obtain an approximation to the filtering distribution but **cannot easily sample full trajectories** from the joint smoothing distribution. Conditional SMC solves this by:

1. **Guaranteeing a valid trajectory**: the reference particle survives all resampling steps
2. **Providing diversity**: the $N - 1$ free particles explore alternative trajectories
3. **Enabling Gibbs sampling**: by alternating between parameter updates and CSMC state updates, we obtain a valid Gibbs sampler (Particle Gibbs)

!!! note "Relation to importance sampling"
    Conditional SMC can be viewed as a form of **conditional importance sampling**. The reference trajectory defines a conditional distribution, and the free particles provide importance-weighted samples around it. The output trajectory is sampled from the full particle set (including the reference), with probability proportional to the final weights.

---

## The Algorithm

### Standard Conditional SMC

At each time step, the reference particle follows $x_t^{\text{ref}}$ exactly, while the other particles are propagated and resampled as in a standard particle filter. Crucially, the reference particle is **never killed** during resampling.

$$
\boxed{
\begin{aligned}
&\textbf{Algorithm: Conditional SMC} \\
&\textbf{Input:} \text{ observations } y_{1:T}, \text{ parameters } \theta, \text{ reference } x_{0:T}^{\text{ref}}, \text{ particles } N \\
&\text{1. Initialize:} \\
&\quad x_0^{(1)} = x_0^{\text{ref}} \\
&\quad \text{For } i = 2, \ldots, N: \text{ sample } x_0^{(i)} \sim p(x_0 \mid \theta) \\
&\quad \text{Compute weights } w_0^{(i)} \propto p(y_0 \mid x_0^{(i)}, \theta) \\
&\text{2. For } t = 1, \ldots, T: \\
&\quad \text{a. Resample indices } \{a_t^{(i)}\}_{i=2}^{N} \text{ from } \{w_{t-1}^{(i)}\}_{i=1}^{N} \\
&\quad \quad \text{Set } a_t^{(1)} = 1 \quad \text{(reference always survives)} \\
&\quad \text{b. Propagate:} \\
&\quad \quad x_t^{(1)} = x_t^{\text{ref}} \quad \text{(fixed)} \\
&\quad \quad \text{For } i = 2, \ldots, N: \text{ sample } x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(a_t^{(i)})}, \theta) \\
&\quad \text{c. Compute weights:} \\
&\quad \quad w_t^{(i)} \propto p(y_t \mid x_t^{(i)}, \theta) \\
&\text{3. Sample output trajectory } x_{0:T}^{\star} \text{ with } P(x_{0:T}^{\star} = x_{0:T}^{(i)}) \propto w_T^{(i)} \\
&\text{4. Return } x_{0:T}^{\star}
\end{aligned}
}
$$

### Key Details

**Ancestor assignment for the reference particle.** At step 2a, the ancestor of particle 1 is always set to 1. This means the reference trajectory is never broken by resampling -- it forms a continuous path from $t = 0$ to $t = T$.

**Weight computation for the reference particle.** The reference particle receives weights just like any other particle. Its weight at time $t$ is:

$$
w_t^{(1)} \propto p(y_t \mid x_t^{\text{ref}}, \theta)
$$

This is important: the reference particle is not guaranteed to be selected as the output -- it competes with the free particles on equal terms via the weights.

**Output sampling.** At the final step, one trajectory is sampled from the full particle set with probabilities proportional to $w_T^{(i)}$. This trajectory may or may not be the reference trajectory.

---

## Efficient Implementation

### Inserting the Reference Trajectory

The reference trajectory must be inserted at every time step without disrupting the standard particle filter operations. The key implementation pattern is:

```python
from particlefilterbox.pmcmc import ConditionalSMC

# Create CSMC with a reference trajectory
csmc = ConditionalSMC(
    model,                              # StateSpaceModel instance
    n_particles=500,                    # total particles (including reference)
    reference_trajectory=x_ref          # (T+1, dim_x) array
)

result = csmc.filter(observations)
```

### Step-by-Step Internals

The implementation handles three operations at each time step that differ from a standard PF:

=== "1. Ancestor Assignment"

    Before resampling, force the reference particle's ancestor:

    ```python
    # Standard resampling for particles 2, ..., N
    ancestors = resample(weights, n_particles - 1)

    # Force reference particle's ancestor
    ancestors = np.concatenate([[0], ancestors])  # particle 0 is always its own ancestor
    ```

=== "2. State Propagation"

    After resampling, override particle 1 with the reference value:

    ```python
    # Propagate free particles from transition
    for i in range(1, n_particles):
        particles[i] = model.transition(particles[ancestors[i]], theta)

    # Override reference particle
    particles[0] = x_ref[t]
    ```

=== "3. Weight Computation"

    Weights are computed identically for all particles, including the reference:

    ```python
    for i in range(n_particles):
        log_weights[i] = model.log_likelihood(y[t], particles[i], theta)
    ```

!!! tip "Memory layout"
    Store the reference trajectory contiguously in memory (e.g., as a NumPy array of shape `(T+1, dim_x)`). At each time step, copying a single row into the particle array is an $O(d_x)$ operation -- negligible compared to the $O(N \cdot d_x)$ cost of propagating the free particles.

---

## Weights of the Conditioned Particle

A natural question is whether the reference particle should receive **special** weights. The answer is no -- the reference particle is weighted exactly like all others:

$$
w_t^{(1)} = p(y_t \mid x_t^{\text{ref}}, \theta)
$$

This is essential for the theoretical validity of Conditional SMC as an MCMC kernel. If we gave the reference particle a different weight, the resulting Gibbs sampler would not target the correct posterior.

!!! warning "Common implementation mistake"
    Do **not** set the reference particle's weight to 1 or give it any preferential treatment in the weight computation. The reference particle must compete fairly with the free particles. Its only special treatment is that it survives resampling.

### What Happens to the Reference Particle's Weight?

In practice, the reference particle's weight behaves differently from the free particles:

- **Early time steps**: the reference trajectory was sampled from a previous CSMC run, so it tends to be a "good" trajectory. Its weight is often competitive.
- **Later time steps**: path degeneracy causes many free particles to coalesce with the reference, so the reference weight is one of many similar values.
- **At $t = T$**: the reference particle's selection probability $\propto w_T^{(1)}$ determines how often the output trajectory equals the reference -- this is the **acceptance rate** of the implicit MCMC move.

---

## As a Building Block

Conditional SMC is not typically used as a standalone method. Instead, it serves as the state-update kernel in two important PMCMC algorithms:

### In Particle Gibbs

[Particle Gibbs](particle-gibbs.md) alternates between:

1. Sampling $\theta^{(m)} \sim p(\theta \mid x_{0:T}^{(m-1)}, y_{1:T})$
2. Sampling $x_{0:T}^{(m)}$ via CSMC with reference $x_{0:T}^{(m-1)}$

```python
from particlefilterbox.pmcmc import ParticleGibbs

pg = ParticleGibbs(
    model,
    n_particles=500,
    n_iterations=5000,
    burnin=1000
)
chain = pg.sample(observations)
```

The CSMC in step 2 is called internally -- you do not need to instantiate `ConditionalSMC` manually.

### In PGAS

[PGAS](pgas.md) extends CSMC with **ancestor sampling**: at each time step, the reference particle's ancestor is resampled (rather than fixed to particle 1), dramatically improving mixing for long time series.

```python
from particlefilterbox.pmcmc import PGAS

pgas = PGAS(
    model,
    n_particles=200,       # PGAS needs fewer particles than PG
    n_iterations=5000,
    burnin=1000
)
chain = pgas.sample(observations)
```

### Standalone Usage

For advanced users who want direct control, `ConditionalSMC` can be used as a standalone building block:

```python
from particlefilterbox.pmcmc import ConditionalSMC
import numpy as np

# Initial reference trajectory (e.g., from a pilot PF run)
pilot_result = model.filter(observations, n_particles=1000)
x_ref = pilot_result.sample_trajectory()

# Run CSMC
csmc = ConditionalSMC(model, n_particles=500, reference_trajectory=x_ref)
result = csmc.filter(observations)

# The output trajectory
x_new = result.sample_trajectory()

# Use x_new as the next reference in a custom MCMC loop
for m in range(n_iterations):
    csmc.set_reference(x_ref)
    result = csmc.filter(observations)
    x_ref = result.sample_trajectory()
```

---

## Diagnostics

### Trajectory Diversity

The most important diagnostic for CSMC is **trajectory diversity**: how different is the output trajectory from the reference?

```python
# Run CSMC and compare output to reference
result = csmc.filter(observations)
x_new = result.sample_trajectory()

# Fraction of time steps where the output differs from the reference
change_rate = np.mean(np.any(x_new != x_ref, axis=-1))
print(f"Trajectory change rate: {change_rate:.2%}")
```

| Change rate | Interpretation | Action |
|---|---|---|
| < 10% | Output almost always equals reference | Increase $N$ significantly |
| 10--40% | Moderate diversity | May be acceptable; consider PGAS |
| 40--80% | Good diversity | Healthy CSMC behavior |
| > 80% | Excellent diversity | Typical of PGAS or short series |

### ESS of the Particle Set

The effective sample size at the final time step indicates how concentrated the weights are:

```python
print(f"Final ESS: {result.ess[-1]:.0f} / {csmc.n_particles}")
```

Low final ESS (e.g., < 10% of $N$) suggests that most particles have negligible weight, which reduces the diversity of the output trajectory.

### Path Degeneracy Visualization

```python
# Visualize particle genealogies
result.plot_genealogy()

# Check ancestor diversity at each time step
result.plot_ancestor_diversity()
```

!!! tip "Diagnosing path degeneracy"
    If the genealogy plot shows all particle paths coalescing to a single ancestor within the first few time steps, you have severe **path degeneracy**. Solutions:

    1. Increase $N$ (expensive, scales poorly with $T$)
    2. Switch to **PGAS** (recommended -- ancestor sampling breaks coalescence)
    3. Use **backward simulation** as a post-processing step

---

## Theoretical Properties

### Validity as an MCMC Kernel

Conditional SMC defines a Markov kernel $K(x_{0:T}^{\star} \mid x_{0:T}^{\text{ref}})$ that leaves the smoothing distribution $p(x_{0:T} \mid \theta, y_{1:T})$ invariant. This means:

$$
\int K(x_{0:T}^{\star} \mid x_{0:T}^{\text{ref}}) \, p(x_{0:T}^{\text{ref}} \mid \theta, y_{1:T}) \, dx_{0:T}^{\text{ref}} = p(x_{0:T}^{\star} \mid \theta, y_{1:T})
$$

This result (Andrieu et al., 2010) holds for **any** $N \geq 2$ -- even with just 2 particles, CSMC is a valid kernel, though mixing will be poor.

### Mixing Rate

The mixing rate of CSMC depends on:

- **$N$**: more particles $\Rightarrow$ higher probability that a free particle "beats" the reference
- **$T$**: longer series $\Rightarrow$ worse path degeneracy $\Rightarrow$ slower mixing
- **Model structure**: strong observations help particles concentrate on good trajectories

For standard CSMC (without ancestor sampling), the mixing time grows as $O(T)$ in the worst case. PGAS reduces this to $O(1)$ under regularity conditions.

---

## What's Next?

<div class="grid cards" markdown>

- :material-arrow-right-bold: **[SMC^2 Online](smc2-online.md)**

    Sequential parameter estimation with online particle adaptation

- :material-arrow-right-bold: **[Tuning Guide](tuning.md)**

    Complete guide to tuning all PMCMC methods

- :material-arrow-left-bold: **[PGAS](pgas.md)**

    Ancestor sampling extension that resolves path degeneracy

- :material-arrow-left-bold: **[Particle Gibbs](particle-gibbs.md)**

    The Gibbs sampler that uses CSMC internally

</div>
