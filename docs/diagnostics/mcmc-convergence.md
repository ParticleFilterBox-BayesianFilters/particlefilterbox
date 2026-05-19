---
title: MCMC Convergence Diagnostics
description: "Convergence diagnostics for PMCMC chains: Gelman-Rubin R-hat, Geweke, Heidelberger-Welch, and modern rank-normalized variants"
---

# MCMC Convergence Diagnostics

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `MCMCConvergence` |
    | **Import** | `from particlefilterbox.diagnostics import MCMCConvergence` |
    | **Input** | Chains from PMMH, Particle Gibbs, or PGAS |
    | **Key method** | `.summary()` |
    | **Goal** | Verify that PMCMC chains have converged to the target posterior |

## Overview

When running PMCMC algorithms --- PMMH, Particle Gibbs, or PGAS --- the output is a collection of MCMC chains sampling from the posterior distribution of model parameters. Before trusting these samples for inference, you must verify that the chains have **converged** to the stationary distribution.

The `MCMCConvergence` diagnostic provides a suite of classical and modern convergence tests:

1. **Gelman-Rubin $\hat{R}$** --- compares between-chain and within-chain variance across multiple chains
2. **Split-$\hat{R}$** --- splits each chain in half to detect within-chain non-stationarity
3. **Rank-normalized $\hat{R}$** --- robust modern variant that handles heavy tails and multimodality
4. **Geweke diagnostic** --- tests whether the first and last portion of a single chain come from the same distribution
5. **Heidelberger-Welch** --- automatic burn-in detection and stationarity test

---

## Basic Usage

```python
from particlefilterbox.diagnostics import MCMCConvergence
import numpy as np

# Assume we ran PMMH with 4 chains
# chains shape: (n_chains, n_iterations, n_parameters)
chains = pmmh.run(n_chains=4, n_iterations=10000)

# Create convergence diagnostic
diag = MCMCConvergence(chains)

# Full summary
print(diag.summary())
```

```text
=== MCMC Convergence Summary ===
Chains:      4
Iterations:  10000 (after burn-in: 5000)
Parameters:  3

Gelman-Rubin R-hat:
  mu        : 1.003  ✓
  phi       : 1.012  ✓
  sigma_v   : 1.008  ✓

Split-R-hat:
  mu        : 1.005  ✓
  phi       : 1.018  ✓
  sigma_v   : 1.011  ✓

Rank-normalized R-hat:
  mu        : 1.002  ✓
  phi       : 1.009  ✓
  sigma_v   : 1.007  ✓

Geweke z-scores (chain 0):
  mu        :  0.42  ✓
  phi       : -1.15  ✓
  sigma_v   :  0.87  ✓

Overall: CONVERGED (all R-hat < 1.05, all |z| < 2.0)
```

---

## Gelman-Rubin Diagnostic ($\hat{R}$)

### Theory

The Gelman-Rubin diagnostic (Gelman & Rubin, 1992) compares between-chain variance $B$ and within-chain variance $W$ for each parameter $\theta$:

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}, \qquad \hat{V} = \frac{n-1}{n} W + \frac{1}{n} B
$$

where $n$ is the number of post-burn-in iterations per chain and:

$$
B = \frac{n}{m-1} \sum_{j=1}^{m} (\bar{\theta}_{j\cdot} - \bar{\theta}_{\cdot\cdot})^2, \qquad
W = \frac{1}{m} \sum_{j=1}^{m} s_j^2
$$

with $m$ chains, $\bar{\theta}_{j\cdot}$ being the mean of chain $j$, and $s_j^2$ the variance of chain $j$.

### Computation

```python
# R-hat for all parameters
rhat = diag.gelman_rubin()
print(rhat)
```

```text
{'mu': 1.003, 'phi': 1.012, 'sigma_v': 1.008}
```

```python
# R-hat for a specific parameter
rhat_phi = diag.gelman_rubin(param="phi")
print(f"R-hat for phi: {rhat_phi:.4f}")
```

### Interpretation

!!! tip "R-hat thresholds"
    | $\hat{R}$ value | Interpretation | Action |
    |-----------------|---------------|--------|
    | $< 1.01$ | Excellent convergence | Safe to use samples |
    | $1.01 - 1.05$ | Acceptable convergence | Likely fine, consider longer chains |
    | $1.05 - 1.10$ | Marginal convergence | Run chains longer or check initialization |
    | $> 1.10$ | Not converged | Do **not** use these samples --- run longer or diagnose |

!!! warning "R-hat requires multiple chains"
    You need at least 2 chains (ideally 4+) to compute $\hat{R}$. If you ran a single chain, use Geweke or Heidelberger-Welch instead. Multiple chains also help detect multimodality --- if chains are stuck in different modes, $\hat{R}$ will be large.

---

## Split-$\hat{R}$

Split-$\hat{R}$ (Gelman et al., 2013) splits each chain in half and treats the halves as separate chains. This detects within-chain non-stationarity that the standard $\hat{R}$ might miss.

```python
# Split-R-hat
split_rhat = diag.split_rhat()
print(split_rhat)
```

```text
{'mu': 1.005, 'phi': 1.018, 'sigma_v': 1.011}
```

If a chain has not reached stationarity, the first half will differ systematically from the second half, inflating the split-$\hat{R}$.

!!! note "When split-$\hat{R}$ differs from $\hat{R}$"
    If $\hat{R} < 1.05$ but split-$\hat{R} > 1.05$, the chains may have reached similar regions but are still trending. This often indicates that the burn-in period was too short.

---

## Rank-Normalized $\hat{R}$

The rank-normalized $\hat{R}$ (Vehtari et al., 2021) replaces the raw parameter values with their ranks before computing $\hat{R}$. This addresses two limitations of the classical diagnostic:

1. **Heavy tails**: Classical $\hat{R}$ can miss convergence issues when the posterior has heavy tails
2. **Discrete parameters or multimodality**: Rank transformation makes the diagnostic robust to non-normal shapes

```python
# Rank-normalized R-hat (recommended as the primary diagnostic)
rank_rhat = diag.rank_rhat()
print(rank_rhat)
```

```text
{'mu': 1.002, 'phi': 1.009, 'sigma_v': 1.007}
```

!!! tip "Modern best practice"
    Vehtari et al. (2021) recommend using rank-normalized $\hat{R}$ as the default convergence diagnostic. It is more robust than the classical version and uses the same threshold ($< 1.01$ ideal, $< 1.05$ acceptable).

---

## Geweke Diagnostic

### Theory

The Geweke diagnostic (Geweke, 1992) tests whether the mean of the first $a$ fraction (default 10%) and the last $b$ fraction (default 50%) of a single chain are equal:

$$
z = \frac{\bar{\theta}_a - \bar{\theta}_b}{\sqrt{\hat{S}_a^2 + \hat{S}_b^2}}
$$

where $\hat{S}_a^2$ and $\hat{S}_b^2$ are spectral density estimates of the variance (accounting for autocorrelation). Under convergence, $z \sim \mathcal{N}(0, 1)$.

### Computation

```python
# Geweke z-scores for each parameter (single chain)
z_scores = diag.geweke(chain_index=0)
print(z_scores)
```

```text
{'mu': 0.42, 'phi': -1.15, 'sigma_v': 0.87}
```

```python
# Geweke with custom fractions
z_scores = diag.geweke(
    chain_index=0,
    first_frac=0.1,   # first 10%
    last_frac=0.5,     # last 50%
)

# Geweke plot: z-scores for successive segments
diag.geweke_plot(chain_index=0, n_segments=20)
```

### Interpretation

!!! tip "Geweke z-score thresholds"
    | $|z|$ value | Interpretation |
    |-------------|---------------|
    | $< 1.0$ | Strong evidence of stationarity |
    | $1.0 - 2.0$ | Acceptable |
    | $2.0 - 3.0$ | Marginal --- consider longer burn-in |
    | $> 3.0$ | Non-stationarity detected --- discard more burn-in or run longer |

The Geweke plot shows z-scores for successive segments of the chain. A converged chain shows z-scores scattered randomly around zero. A systematic trend indicates non-stationarity.

---

## Heidelberger-Welch Diagnostic

### Theory

The Heidelberger-Welch diagnostic (Heidelberger & Welch, 1983) performs two tests:

1. **Stationarity test**: Uses the Cramér-von Mises statistic to test whether the chain is stationary. If the full chain fails, it successively discards the first 10%, 20%, ..., 50% until it passes (automatic burn-in detection).
2. **Halfwidth test**: Checks whether the MCMC estimate of the mean has a sufficiently small confidence interval relative to the mean.

### Computation

```python
# Heidelberger-Welch diagnostic
hw = diag.heidelberger_welch(chain_index=0)
print(hw)
```

```text
=== Heidelberger-Welch Diagnostic ===
Parameter  | Stationarity | Start iter | Halfwidth test | Mean
-----------+--------------+------------+----------------+-------
mu         |    PASS      |     1      |     PASS       | 0.032
phi        |    PASS      |     501    |     PASS       | 0.975
sigma_v    |    PASS      |     1      |     PASS       | 0.158

Recommended burn-in: 501 iterations
```

### Automatic Burn-in

```python
# Use Heidelberger-Welch to determine burn-in
burnin = diag.heidelberger_welch_burnin(chain_index=0)
print(f"Recommended burn-in: {burnin} iterations")

# Apply burn-in and recompute diagnostics
diag_trimmed = MCMCConvergence(chains[:, burnin:, :])
print(diag_trimmed.summary())
```

!!! warning "When Heidelberger-Welch fails"
    If the stationarity test fails even after discarding 50% of the chain, the chain has likely not converged. Possible remedies:

    - Run the chain significantly longer (2x--5x current length)
    - Improve the PMMH proposal (reduce $N$ variance or tune step size)
    - Check for multimodality in the posterior
    - Consider reparameterization of the model

---

## Running All Diagnostics

```python
# One-line comprehensive diagnostic
diag = MCMCConvergence(chains)
report = diag.summary(
    include_rhat=True,
    include_split_rhat=True,
    include_rank_rhat=True,
    include_geweke=True,
    include_hw=True,
)

# Check convergence programmatically
if diag.has_converged(rhat_threshold=1.05, geweke_threshold=2.0):
    print("Chains have converged --- safe to proceed with inference")
else:
    print("WARNING: Convergence not achieved")
    print(diag.convergence_issues())
```

---

## Convergence Plots

### Trace Plot

```python
# Trace plot for visual inspection
diag.trace_plot(params=["mu", "phi", "sigma_v"])
```

The trace plot shows all chains overlaid for each parameter. Converged chains should:

- **Mix well**: Chains overlap and are indistinguishable
- **Be stationary**: No trends or drift
- **Explore the same region**: No chain stuck in a separate mode

### Running Mean Plot

```python
# Running mean for each chain
diag.running_mean_plot(params=["mu", "phi"])
```

The running mean should stabilize to the same value across all chains. Diverging running means indicate non-convergence.

---

## Complete Example: PMMH Convergence Check

```python
import numpy as np
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.pmcmc import PMMH
from particlefilterbox.diagnostics import MCMCConvergence

# Setup
model = StochasticVolatility(variant="basic")
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=200, rng=rng)

# Run PMMH with 4 chains from dispersed starting points
pmmh = PMMH(model, obs, n_particles=1000)
chains = pmmh.run(
    n_chains=4,
    n_iterations=10000,
    seed=42,
)

# 1. Full convergence diagnostic
diag = MCMCConvergence(chains)
print(diag.summary())

# 2. Visual inspection
diag.trace_plot(params=["mu", "phi", "sigma_v"])

# 3. Automatic burn-in detection
burnin = diag.heidelberger_welch_burnin(chain_index=0)
print(f"\nRecommended burn-in: {burnin}")

# 4. Apply burn-in and verify
diag_post = MCMCConvergence(chains[:, burnin:, :])
assert diag_post.has_converged(), "Chains did not converge!"

# 5. Report final diagnostics
print(diag_post.summary())
```

---

## API Summary

| Method | Description |
|--------|-------------|
| `MCMCConvergence(chains)` | Create diagnostic from chains array `(n_chains, n_iter, n_params)` |
| `.gelman_rubin(param=None)` | Classical $\hat{R}$ per parameter |
| `.split_rhat(param=None)` | Split-$\hat{R}$ (within-chain stationarity) |
| `.rank_rhat(param=None)` | Rank-normalized $\hat{R}$ (Vehtari et al., 2021) |
| `.geweke(chain_index, first_frac, last_frac)` | Geweke z-scores for a single chain |
| `.geweke_plot(chain_index, n_segments)` | Plot z-scores for successive segments |
| `.heidelberger_welch(chain_index)` | Stationarity + halfwidth tests |
| `.heidelberger_welch_burnin(chain_index)` | Automatic burn-in detection |
| `.summary(**kwargs)` | Comprehensive convergence report |
| `.has_converged(rhat_threshold, geweke_threshold)` | Programmatic convergence check |
| `.convergence_issues()` | List parameters that failed diagnostics |
| `.trace_plot(params)` | Trace plots for visual inspection |
| `.running_mean_plot(params)` | Running mean convergence plot |

---

## See Also

- [Mixing Diagnostics](mixing.md) --- ACF, ESS, and integrated autocorrelation time
- [Acceptance Rate](acceptance-rate.md) --- PMMH acceptance rate tuning
- [ESS Diagnostic](ess-diagnostic.md) --- particle filter ESS (not MCMC ESS)
- [Convergence Diagnostic](convergence.md) --- particle filter convergence with $N$
- [PMMH](../user-guide/pmcmc/pmmh.md) --- Particle Marginal Metropolis-Hastings algorithm
- [Particle Gibbs](../user-guide/pmcmc/particle-gibbs.md) --- Gibbs sampling with conditional SMC
- [PGAS](../user-guide/pmcmc/pgas.md) --- Particle Gibbs with Ancestor Sampling
- [PMCMC Tuning](../user-guide/pmcmc/tuning.md) --- practical tuning advice for PMCMC methods
- [Theory: PMCMC](../theory/pmcmc-theory.md) --- theoretical guarantees for PMCMC convergence
