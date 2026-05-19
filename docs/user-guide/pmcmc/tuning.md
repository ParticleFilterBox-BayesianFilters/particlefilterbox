---
title: PMCMC Tuning Guide
description: Complete guide to tuning particle MCMC methods -- particle count, proposals, diagnostics, burn-in, and troubleshooting
---

# PMCMC Tuning Guide

Getting good results from PMCMC requires careful tuning. This guide covers all the key decisions -- from choosing the number of particles to diagnosing convergence -- with practical rules of thumb for each method.

---

## 1. Choosing the Number of Particles ($N$)

The number of particles $N$ in the internal particle filter is the **most important** tuning parameter in PMCMC. It controls the variance of the likelihood estimate, which directly affects the efficiency of the sampler.

### The Fundamental Tradeoff

More particles give a more accurate likelihood estimate, but each MCMC iteration becomes more expensive:

$$
\text{Total cost} \propto M \times N \times T
$$

where $M$ is the number of MCMC iterations, $N$ is the number of particles, and $T$ is the time series length. The goal is to find the **smallest $N$** that gives acceptable mixing.

### The Variance Rule

For PMMH, the optimal $N$ is the one that makes the **variance of the log-likelihood** estimate approximately 1--3 (Doucet et al., 2015; Sherlock et al., 2015):

$$
\text{Var}\bigl[\log \hat{p}(y_{1:T} \mid \theta)\bigr] \approx 1 \text{--} 3
$$

This corresponds to an acceptance rate of roughly **15--30%** for random walk proposals.

!!! tip "How to estimate the variance"
    Run the particle filter **multiple times** at the same $\theta$ value and compute the variance of the log-likelihood estimates:

    ```python
    from particlefilterbox.diagnostics import likelihood_variance

    # Run PF 50 times at the posterior mode
    theta_star = chain.mode()  # or a reasonable parameter value
    var_ll = likelihood_variance(model, observations, theta_star,
                                  n_particles=500, n_replicates=50)
    print(f"Var[log p(y|θ)] = {var_ll:.2f}")
    ```

    | $\text{Var}[\log \hat{p}]$ | Interpretation | Action |
    |---|---|---|
    | < 0.5 | Too many particles (wasting computation) | Decrease $N$ |
    | 0.5 -- 1.0 | Slightly conservative (fine for production) | OK |
    | 1.0 -- 3.0 | Optimal range | Ideal |
    | 3.0 -- 10.0 | High variance, low acceptance | Increase $N$ |
    | > 10.0 | Extreme variance, chain will get stuck | Increase $N$ significantly |

### Rules of Thumb by Method

Each PMCMC method has different sensitivity to $N$:

=== "PMMH"

    The variance of $\log \hat{p}$ directly controls the acceptance rate.

    - **Starting point**: $N = 100$--$500$
    - **Target**: $\text{Var}[\log \hat{p}] \approx 1$--$3$
    - **Scaling**: $N$ typically needs to grow as $O(\sqrt{T})$

    ```python
    from particlefilterbox.pmcmc import PMMH

    # Pilot run with different N values
    for n in [100, 200, 500, 1000]:
        pmmh = PMMH(model, n_particles=n, n_iterations=1000,
                     proposal='adaptive')
        pilot = pmmh.sample(observations)
        print(f"N={n:4d}: acceptance={pilot.acceptance_rate:.2%}, "
              f"ESS={pilot.ess.mean():.0f}")
    ```

=== "Particle Gibbs"

    $N$ controls **trajectory diversity** rather than acceptance rate.

    - **Starting point**: $N = 5\sqrt{T}$
    - **Target**: trajectory change rate > 50%
    - **Scaling**: $N$ must grow with $T$ (path degeneracy)

    ```python
    from particlefilterbox.pmcmc import ParticleGibbs

    pg = ParticleGibbs(model, n_particles=500, n_iterations=2000)
    chain = pg.sample(observations)
    print(f"Trajectory change rate: {chain.trajectory_change_rate:.2%}")
    ```

=== "PGAS"

    Ancestor sampling reduces sensitivity to $N$, so fewer particles are needed.

    - **Starting point**: $N = 50$--$200$
    - **Target**: state ESS > 500 per time step
    - **Scaling**: $N$ does **not** need to grow with $T$

    ```python
    from particlefilterbox.pmcmc import PGAS

    pgas = PGAS(model, n_particles=100, n_iterations=3000)
    chain = pgas.sample(observations)
    print(f"State ESS (mean): {chain.state_ess.mean():.0f}")
    ```

=== "SMC$^2$"

    Two particle counts to tune: $N_\theta$ and $N_x$.

    - **$N_\theta$**: 100--500 (more = less frequent rejuvenation)
    - **$N_x$**: start at 200, enable adaptive if long series
    - **Scaling**: $N_x$ may need to grow with $t$ (use adaptive)

    ```python
    from particlefilterbox.pmcmc import SMC2Online

    smc2 = SMC2Online(model, n_theta=200, n_x=300,
                       adaptive_n_x=True, ess_threshold=0.5)
    smc2.run(observations)
    print(f"Final N_x: {smc2.n_x}")
    print(f"Rejuvenations: {smc2.n_rejuvenations}")
    ```

### Adaptive $N$ During Burn-in

A practical strategy is to start with fewer particles during burn-in (when the chain is far from the posterior) and increase $N$ for the production phase:

```python
# Phase 1: burn-in with fewer particles (fast exploration)
pmmh_burnin = PMMH(model, n_particles=100, n_iterations=2000,
                    proposal='adaptive')
burnin_chain = pmmh_burnin.sample(observations)

# Use burn-in to estimate good starting values and proposal covariance
theta_init = burnin_chain.mean()
cov_init = burnin_chain.covariance()

# Phase 2: production with more particles (accurate posterior)
pmmh_prod = PMMH(model, n_particles=500, n_iterations=10000,
                  proposal='adaptive', proposal_cov=cov_init,
                  initial_theta=theta_init)
chain = pmmh_prod.sample(observations)
```

!!! tip "Two-phase strategy"
    This two-phase approach can reduce total computation by 2--5x compared to running the full chain with the production $N$. The burn-in phase needs only rough exploration, not precise likelihood estimates.

---

## 2. Proposal Tuning

The proposal distribution determines how the MCMC chain explores the parameter space. Good proposals make large moves that are frequently accepted; poor proposals either take tiny steps or propose values that are almost always rejected.

### PMMH Proposal Strategies

=== "Random Walk"

    $$
    \theta' = \theta^{(m-1)} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \Sigma)
    $$

    - **Scale $\Sigma$**: controls step size. Too large $\to$ low acceptance. Too small $\to$ slow exploration.
    - **Target acceptance**: **20--30%** for low-dimensional ($d \leq 5$), **15--25%** for higher dimensions
    - **Optimal scaling**: $\Sigma = \frac{2.38^2}{d} \hat{\Sigma}_{\text{posterior}}$ (Roberts & Rosenthal, 2001)

    ```python
    pmmh = PMMH(model, n_particles=500, n_iterations=10000,
                proposal='random_walk', proposal_scale=0.1)
    ```

=== "Adaptive Metropolis (AM)"

    Automatically tunes $\Sigma$ using the chain history:

    $$
    \Sigma_m = \frac{2.38^2}{d} \hat{\Sigma}_m + \epsilon I_d
    $$

    - **Recommended default** -- removes manual tuning
    - `adaptation_start`: begin adapting after this many iterations (default 500)
    - The small $\epsilon I_d$ term prevents the covariance from becoming singular

    ```python
    pmmh = PMMH(model, n_particles=500, n_iterations=10000,
                proposal='adaptive', adaptation_start=500)
    ```

=== "MALA"

    Uses gradient information to propose in high-probability directions:

    $$
    \theta' = \theta^{(m-1)} + \frac{h}{2} \nabla_\theta \log \pi(\theta^{(m-1)}) + \sqrt{h} \, \eta, \quad \eta \sim \mathcal{N}(0, I)
    $$

    - **Step size $h$**: controls the strength of the gradient push
    - **Target acceptance**: **50--60%** (higher than random walk due to gradient guidance)
    - Best for **high-dimensional** problems ($d > 5$)

    ```python
    pmmh = PMMH(model, n_particles=500, n_iterations=10000,
                proposal='mala', step_size=0.01)
    ```

=== "HMC-within-PMMH"

    Hamiltonian Monte Carlo proposals for very high-dimensional parameter spaces:

    - **Leapfrog steps $L$**: number of integration steps (5--20)
    - **Step size $\epsilon$**: integration step size
    - **Target acceptance**: **65--80%**
    - Requires gradient of log-posterior

    ```python
    pmmh = PMMH(model, n_particles=500, n_iterations=5000,
                proposal='hmc', n_leapfrog=10, step_size=0.01)
    ```

### Acceptance Rate Targets

| Proposal | Optimal acceptance rate | Dimension regime |
|---|---|---|
| Random walk | 23.4% (asymptotic) | All |
| Adaptive (AM) | 20--30% | All |
| MALA | 57.4% (asymptotic) | $d > 5$ |
| HMC | 65--80% | $d > 10$ |

!!! warning "PMCMC acceptance rates are lower than pure MCMC"
    In standard MCMC, the likelihood is exact. In PMCMC, the **noisy** likelihood estimate adds extra randomness to the acceptance probability, systematically **lowering** the acceptance rate. The targets above already account for this, but if your acceptance rate is much lower than expected, the particle filter variance may be too high -- increase $N$ before adjusting the proposal.

### Reparameterization

When parameters are highly correlated or have very different scales, reparameterization can dramatically improve mixing:

```python
# Bad: correlated parameters with different scales
# mu ∈ (-∞, ∞), phi ∈ (0, 1), sigma ∈ (0, ∞)

# Better: transform to unconstrained space
# mu stays as-is, phi → logit(phi), sigma → log(sigma)
model_reparam = model.reparameterize({
    'phi': 'logit',      # maps (0,1) → (-∞, ∞)
    'sigma': 'log'       # maps (0, ∞) → (-∞, ∞)
})

pmmh = PMMH(model_reparam, n_particles=500, n_iterations=10000,
            proposal='adaptive')
chain = pmmh.sample(observations)

# Transform back to original parameterization
chain_original = chain.inverse_transform()
```

---

## 3. Chain Diagnostics

After running a PMCMC chain, you must check whether it has converged to the posterior. Never trust posterior summaries without verifying convergence.

### Trace Plots

The first diagnostic: visually inspect the parameter traces.

```python
chain.plot_trace()
```

**What to look for:**

- **Good mixing**: the trace looks like white noise around a stable mean
- **Poor mixing**: the trace shows long excursions, trends, or sticky regions
- **Non-stationarity**: the mean or variance changes over time (chain hasn't converged)

!!! tip "Multiple chains"
    Always run at least **2--4 chains** from different starting values. If they converge to the same region, you have more confidence in convergence.

    ```python
    chains = pmmh.sample(observations, n_chains=4)
    chains.plot_trace()  # overlaid trace plots from all chains
    ```

### Autocorrelation Function (ACF)

The ACF measures how correlated successive samples are:

```python
chain.plot_autocorr(max_lag=100)
```

- **Fast decay** (ACF drops to zero within 10--20 lags): good mixing
- **Slow decay** (ACF remains positive for 100+ lags): poor mixing, consider reparameterization or different proposal

### Effective Sample Size (ESS)

The ESS estimates how many **independent** samples the chain is equivalent to:

$$
\text{ESS} = \frac{M}{1 + 2\sum_{k=1}^{\infty} \rho_k}
$$

where $M$ is the chain length and $\rho_k$ is the autocorrelation at lag $k$.

```python
print(f"ESS per parameter: {chain.ess}")
print(f"ESS per second: {chain.ess / chain.runtime:.1f}")
```

| ESS | Interpretation |
|---|---|
| < 100 | Insufficient for reliable inference |
| 100 -- 500 | Marginal; increase iterations or improve mixing |
| 500 -- 2000 | Adequate for most applications |
| > 2000 | Excellent |

!!! tip "ESS per second"
    **ESS per second** is the best metric for comparing methods. A method with lower ESS per iteration but faster iterations may be more efficient overall.

    ```python
    # Compare PMMH vs PGAS efficiency
    print(f"PMMH:  ESS={chain_pmmh.ess.min():.0f}, "
          f"time={chain_pmmh.runtime:.1f}s, "
          f"ESS/s={chain_pmmh.ess.min()/chain_pmmh.runtime:.1f}")
    print(f"PGAS:  ESS={chain_pgas.ess.min():.0f}, "
          f"time={chain_pgas.runtime:.1f}s, "
          f"ESS/s={chain_pgas.ess.min()/chain_pgas.runtime:.1f}")
    ```

### Gelman-Rubin Diagnostic ($\hat{R}$)

The Gelman-Rubin statistic compares **between-chain** and **within-chain** variance across multiple chains:

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}
$$

where $\hat{V}$ is the pooled variance estimate and $W$ is the within-chain variance.

```python
# Run multiple chains
chains = pmmh.sample(observations, n_chains=4)

# Compute R-hat
print(f"R-hat per parameter: {chains.rhat}")
```

| $\hat{R}$ | Interpretation |
|---|---|
| < 1.01 | Excellent convergence |
| 1.01 -- 1.05 | Acceptable |
| 1.05 -- 1.10 | Questionable; run longer |
| > 1.10 | Not converged; investigate |

!!! warning "R-hat alone is not sufficient"
    $\hat{R} < 1.05$ is a **necessary** but not **sufficient** condition for convergence. A chain can have low $\hat{R}$ but still miss a mode of a multimodal posterior. Always combine $\hat{R}$ with trace plots and ESS.

### Geweke Diagnostic

The Geweke test compares the mean of the **first 10%** of the chain (after burn-in) with the **last 50%**:

$$
z = \frac{\bar{\theta}_A - \bar{\theta}_B}{\sqrt{\hat{\sigma}_A^2 + \hat{\sigma}_B^2}}
$$

Under convergence, $z$ follows a standard normal distribution.

```python
print(f"Geweke z-scores: {chain.geweke()}")
# Values in [-2, 2] indicate no evidence against convergence
```

---

## 4. Burn-in and Thinning

### Burn-in

**Burn-in** is the initial portion of the chain that is discarded because it may not have reached the stationary distribution. The chain starts from an arbitrary point and needs time to reach the high-probability region of the posterior.

**How much burn-in?**

- **Conservative rule**: discard the first **50%** of iterations
- **Moderate rule**: discard the first **20--30%** (if starting from a reasonable initial value)
- **Adaptive rule**: use the Geweke diagnostic to find where stationarity begins

```python
# Set burn-in at construction time
pmmh = PMMH(model, n_particles=500, n_iterations=10000, burnin=5000)
chain = pmmh.sample(observations)

# Or discard manually after sampling
chain_trimmed = chain.burn(5000)
```

!!! tip "Starting from a good initial value"
    You can dramatically reduce the needed burn-in by starting from a good initial value:

    ```python
    # Maximum likelihood estimate as starting point
    from scipy.optimize import minimize

    def neg_log_lik(theta):
        return -model.log_likelihood_estimate(observations, theta, n_particles=200)

    result = minimize(neg_log_lik, x0=[0.0, 0.95, 0.2])
    theta_init = result.x

    pmmh = PMMH(model, n_particles=500, n_iterations=10000,
                burnin=1000, initial_theta=theta_init)
    ```

### Thinning

**Thinning** keeps every $k$-th sample and discards the rest. This is primarily useful for **reducing memory** when storing or post-processing very long chains:

```python
# Keep every 10th sample
pmmh = PMMH(model, n_particles=500, n_iterations=100000,
            burnin=20000, thin=10)
chain = pmmh.sample(observations)
# chain contains 8000 samples (80000 post-burnin / 10)
```

!!! warning "Thinning is usually unnecessary"
    Thinning **discards information**. A chain of 10,000 correlated samples contains **more** information than 1,000 thinned samples. Only thin when:

    - Memory is a constraint (e.g., storing state trajectories from Particle Gibbs)
    - You need approximately independent samples for a downstream analysis
    - The chain is extremely long and storage is expensive

    In most cases, it is better to keep all samples and let posterior summaries account for autocorrelation.

---

## 5. Troubleshooting

### Acceptance Rate Too Low (< 5%)

The chain gets stuck at the same value for many iterations.

| Possible cause | Diagnosis | Fix |
|---|---|---|
| Too few particles | $\text{Var}[\log \hat{p}] > 5$ | Increase $N$ |
| Proposal scale too large | Large proposed jumps | Reduce `proposal_scale` or use `'adaptive'` |
| Poor parameterization | Highly correlated parameters | Reparameterize (logit, log transforms) |
| Misspecified model | Likelihood always very low | Check model specification |

```python
# Diagnose: is it the particles or the proposal?
# 1. Check likelihood variance
var_ll = likelihood_variance(model, observations, chain.mode(),
                              n_particles=500, n_replicates=50)
print(f"Var[log p(y|θ)] = {var_ll:.2f}")  # if > 5, increase N

# 2. Check if acceptance improves with more particles
for n in [200, 500, 1000, 2000]:
    pilot = PMMH(model, n_particles=n, n_iterations=500).sample(observations)
    print(f"N={n}: acceptance={pilot.acceptance_rate:.2%}")
```

### Acceptance Rate Too High (> 50%)

The chain moves at every iteration but takes very small steps.

| Possible cause | Diagnosis | Fix |
|---|---|---|
| Proposal scale too small | Tiny steps in trace plot | Increase `proposal_scale` |
| Too many particles | $\text{Var}[\log \hat{p}] < 0.5$ | Decrease $N$ (save computation) |
| Near-deterministic model | Very peaked likelihood | Normal behavior; check ESS |

### Poor Mixing

The chain moves but explores slowly -- high autocorrelation, low ESS.

```python
# Check autocorrelation
chain.plot_autocorr(max_lag=200)

# If ACF decays slowly:
print(f"ESS: {chain.ess}")
print(f"Acceptance rate: {chain.acceptance_rate:.2%}")
```

**Solutions:**

1. **Use adaptive proposals**: switch to `proposal='adaptive'` to learn the posterior covariance
2. **Reparameterize**: transform correlated parameters to reduce posterior correlation
3. **Use MALA or HMC**: gradient-informed proposals for high-dimensional problems
4. **Switch methods**: PGAS often mixes better than PMMH for joint state-parameter inference

!!! tip "Diagnosing correlation-induced slow mixing"
    Plot the **pairwise posterior** to see correlations:

    ```python
    chain.plot_pairs()  # scatter plots of all parameter pairs
    ```

    If you see strong banana-shaped or diagonal correlations, reparameterization will help significantly.

### Multimodality

The posterior has multiple separated modes, and the chain gets trapped in one.

**Symptoms:**

- Multiple chains converge to different values
- $\hat{R}$ is large even after long runs
- Trace plots show the chain staying in one region indefinitely

**Solutions:**

| Strategy | Description |
|---|---|
| **Multiple chains** | Run many chains from dispersed starting values |
| **Tempering** | Use parallel tempering to help chains jump between modes |
| **SMC$^2$** | The particle-based approach naturally handles multimodality |
| **Reparameterization** | Sometimes modes merge under a different parameterization |

```python
# Run chains from dispersed starting values
import numpy as np

initial_values = [
    {'mu': -1.0, 'phi': 0.8, 'sigma': 0.3},
    {'mu':  0.0, 'phi': 0.95, 'sigma': 0.1},
    {'mu':  1.0, 'phi': 0.99, 'sigma': 0.5},
    {'mu':  0.5, 'phi': 0.90, 'sigma': 0.2},
]

chains = pmmh.sample(observations, n_chains=4,
                      initial_values=initial_values)

# Check if all chains found the same mode
for i, c in enumerate(chains):
    print(f"Chain {i}: mean = {c.mean()}")
```

### Degeneracy

The particle filter collapses -- all weight concentrates on a single particle.

**Symptoms:**

- Very low PF ESS (< 2--3) at some time steps
- Erratic log-likelihood estimates
- Sudden jumps in acceptance rate

**Solutions:**

1. **Increase $N$**: more particles resist degeneracy
2. **Improve the proposal**: use a guided or locally optimal PF instead of bootstrap
3. **Resample more frequently**: adaptive resampling (resample when ESS < $N/2$)
4. **Shorten the series**: if possible, process data in blocks

```python
# Diagnose: check PF ESS at each time step
result = model.filter(observations, n_particles=500, theta=chain.mode())
print(f"Min PF ESS: {result.ess.min():.0f} at t={result.ess.argmin()}")
print(f"Mean PF ESS: {result.ess.mean():.0f}")

# If min ESS is very low, try a better proposal
from particlefilterbox.filters import GuidedPF

guided_model = model.with_filter(GuidedPF, guide='ekf')
pmmh = PMMH(guided_model, n_particles=200, n_iterations=10000)
chain = pmmh.sample(observations)
```

!!! warning "Degeneracy vs. poor mixing"
    **Degeneracy** is a problem of the internal particle filter (too few $x$-particles). **Poor mixing** is a problem of the external MCMC chain (bad proposals or high likelihood variance). The symptoms overlap -- both cause low acceptance rates -- but the fixes are different. Always check the PF ESS to distinguish them.

---

## Quick Reference: Tuning Checklist

Use this checklist when setting up a new PMCMC analysis:

| Step | What to check | Target |
|---|---|---|
| 1. Choose method | PMMH / PG / PGAS / SMC$^2$ | Based on [method comparison](index.md) |
| 2. Set $N$ | Run pilot, check $\text{Var}[\log \hat{p}]$ | 1--3 for PMMH |
| 3. Set proposal | Start with `'adaptive'` | 15--30% acceptance |
| 4. Run pilot chain | 1000--2000 iterations | Check trace plots |
| 5. Adjust $N$ and proposal | Based on pilot diagnostics | Iterate until satisfied |
| 6. Run production chain | 10,000+ iterations | -- |
| 7. Check ESS | `chain.ess` | > 1000 per parameter |
| 8. Check $\hat{R}$ | Multiple chains | < 1.05 |
| 9. Check trace plots | Visual inspection | Stationary, well-mixing |
| 10. Report results | `chain.summary()` | After verifying convergence |

---

## What's Next?

<div class="grid cards" markdown>

- :material-arrow-left-bold: **[PMMH](pmmh.md)**

    Parameter estimation with particle MH

- :material-arrow-left-bold: **[PGAS](pgas.md)**

    Joint inference with ancestor sampling

- :material-arrow-left-bold: **[SMC^2 Online](smc2-online.md)**

    Sequential online parameter estimation

- :material-arrow-left-bold: **[PMCMC Overview](index.md)**

    Back to the framework overview and method comparison

</div>
