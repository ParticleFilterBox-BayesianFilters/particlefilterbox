---
title: Acceptance Rate Diagnostics
description: "PMMH acceptance rate diagnostics: target rates, rolling monitoring, particle count tuning, and proposal adaptation"
---

# Acceptance Rate Diagnostics

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `AcceptanceRateDiagnostic` |
    | **Import** | `from particlefilterbox.diagnostics import AcceptanceRateDiagnostic` |
    | **Input** | Chain from PMMH with acceptance history |
    | **Key metric** | Overall and rolling acceptance rate |
    | **Target range** | 15--30% for PMMH (dimension-dependent) |

## Overview

The Particle Marginal Metropolis-Hastings (PMMH) algorithm uses a Metropolis-Hastings accept/reject step at each iteration. The **acceptance rate** --- the fraction of proposals that are accepted --- is a critical diagnostic for tuning the algorithm.

Unlike standard MCMC where the optimal acceptance rate is well-known (23.4% for random walk Metropolis in high dimensions), PMMH has an additional source of noise: the **particle filter estimate** of the log-likelihood. This makes the acceptance rate depend on both the proposal distribution and the number of particles $N$.

---

## Basic Usage

```python
from particlefilterbox.diagnostics import AcceptanceRateDiagnostic

# From a PMMH chain with recorded accept/reject decisions
diag = AcceptanceRateDiagnostic(chain)

print(diag.summary())
```

```text
=== Acceptance Rate Diagnostic ===
Total iterations:     10000
Accepted:             2187
Rejected:             7813
Overall rate:         0.219 (21.9%)

Phase analysis:
  Burn-in (0-2000):   0.185 (18.5%)
  Post burn-in:       0.227 (22.7%)

Verdict: GOOD (target: 15-30%)
```

---

## The Acceptance Rate in PMMH

### Theory

In PMMH, the acceptance probability for a proposal $\theta^* \sim q(\cdot \mid \theta)$ is:

$$
\alpha(\theta^*, \theta) = \min\!\left(1,\; \frac{\hat{p}(y \mid \theta^*)\, p(\theta^*)\, q(\theta \mid \theta^*)}{\hat{p}(y \mid \theta)\, p(\theta)\, q(\theta^* \mid \theta)}\right)
$$

where $\hat{p}(y \mid \theta)$ is the **unbiased** particle filter estimate of the likelihood. The key difference from standard MH is that the likelihood ratio $\hat{p}(y \mid \theta^*) / \hat{p}(y \mid \theta)$ is noisy --- its variance depends on the number of particles $N$.

### Sources of Rejection

Proposals in PMMH are rejected for two distinct reasons:

1. **Bad proposals**: $\theta^*$ is far from the posterior mode --- this is normal and desirable
2. **Noisy likelihood**: The particle filter estimate $\hat{p}(y \mid \theta^*)$ is noisy, causing random rejections even for good $\theta^*$ --- this is wasteful

!!! note "The trade-off"
    Increasing $N$ reduces noisy rejections (source 2) but is computationally expensive. The goal is to find the minimum $N$ that keeps the noise-induced rejection rate low enough for good mixing.

---

## Target Acceptance Rates

### Optimal Rates by Method

| Method | Target acceptance rate | Reason |
|--------|----------------------|--------|
| PMMH (low dim, $d \leq 3$) | 25--40% | Close to standard RWM optimal |
| PMMH (moderate dim, $d = 3\text{--}10$) | 15--30% | Noise from PF lowers optimal rate |
| PMMH (high dim, $d > 10$) | 10--20% | Higher dimensions need conservative proposals |
| Particle Gibbs / PGAS | N/A | No accept/reject step (Gibbs updates) |

!!! tip "The Pitt et al. (2012) rule"
    For PMMH, the optimal number of particles $N$ is the one that makes the standard deviation of $\log \hat{p}(y \mid \theta)$ approximately **1.0 to 1.7**. This corresponds to acceptance rates in the 15--30% range for typical models. If $\text{Std}[\log \hat{p}]$ is much larger, acceptance drops; if much smaller, you are wasting computation on unnecessary particles.

---

## Rolling Acceptance Rate

The overall acceptance rate can mask important dynamics. The **rolling acceptance rate** reveals how the chain behaves over time:

```python
# Rolling acceptance rate with window of 500 iterations
diag.plot_rolling(window=500)
```

```python
# Get rolling rate as array
rolling = diag.rolling_rate(window=500)
print(f"Min rolling rate:  {rolling.min():.3f}")
print(f"Max rolling rate:  {rolling.max():.3f}")
print(f"Final rolling rate: {rolling[-1]:.3f}")
```

```text
Min rolling rate:  0.142
Max rolling rate:  0.298
Final rolling rate: 0.224
```

### Interpreting Rolling Rate Patterns

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| Stable around target | Well-tuned algorithm | No action needed |
| High initially, drops to target | Burn-in transient | Normal; ensure burn-in is long enough |
| Steadily decreasing | Chain drifting to hard region | Check for multimodality or model issues |
| Large oscillations | Proposal scale mismatched | Adapt proposal covariance |
| Near zero for extended periods | Chain is stuck | Increase $N$ or widen proposal |

---

## Relationship Between $N$ and Acceptance Rate

The number of particles directly affects the acceptance rate through the noise in the likelihood estimate:

```python
# Acceptance rate as a function of N
diag.plot_acceptance_vs_n(
    n_values=[100, 250, 500, 1000, 2000, 5000],
    n_iterations=2000,
    seed=42,
)
```

```text
=== Acceptance Rate vs N ===
     N   | Acceptance Rate | Std(log-lik) | Time per iter
---------+-----------------+--------------+--------------
     100 |     0.032       |     8.42     |    0.02s
     250 |     0.087       |     3.91     |    0.05s
     500 |     0.148       |     2.14     |    0.10s
    1000 |     0.215       |     1.38     |    0.19s
    2000 |     0.261       |     0.87     |    0.38s
    5000 |     0.289       |     0.51     |    0.95s
```

!!! warning "Diminishing returns beyond the optimal $N$"
    Once $\text{Std}[\log \hat{p}] < 1.0$, further increasing $N$ barely improves the acceptance rate but linearly increases computation time. The sweet spot is $\text{Std}[\log \hat{p}] \approx 1.0\text{--}1.7$.

### Visualizing the Trade-off

```python
# Cost-efficiency plot: ESS per second vs N
diag.plot_efficiency(
    n_values=[100, 250, 500, 1000, 2000, 5000],
    n_iterations=2000,
    seed=42,
)
```

The efficiency plot shows the effective samples per second (accounting for both acceptance rate and computation time). This typically has a clear maximum at an intermediate $N$.

---

## Tuning Guide

### Step 1: Determine the Right $N$

```python
from particlefilterbox.diagnostics import AcceptanceRateDiagnostic, ConvergenceDiagnostic

# Use the convergence diagnostic to find optimal N
conv = ConvergenceDiagnostic(model, obs)
for n in [250, 500, 1000, 2000]:
    report = conv.inter_run_variance(n_particles=n, n_runs=20, seed=42)
    print(f"N={n:5d}: std(log-lik) = {report['ll_std']:.3f}")
```

Target: $\text{Std}[\log \hat{p}] \approx 1.0\text{--}1.7$.

### Step 2: Tune the Proposal Scale

```python
# Run short pilot chains with different proposal scales
from particlefilterbox.pmcmc import PMMH

for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
    pmmh = PMMH(model, obs, n_particles=1000, proposal_scale=scale)
    chain = pmmh.run(n_iterations=2000, seed=42)
    diag = AcceptanceRateDiagnostic(chain)
    print(f"Scale={scale:.1f}: acceptance={diag.overall_rate:.3f}")
```

```text
Scale=0.1: acceptance=0.672   # too high - proposals too small
Scale=0.5: acceptance=0.341   # slightly high
Scale=1.0: acceptance=0.219   # good
Scale=2.0: acceptance=0.098   # too low
Scale=5.0: acceptance=0.012   # way too low
```

### Step 3: Use Adaptive PMMH

```python
# Adaptive PMMH learns the proposal during burn-in
pmmh = PMMH(
    model, obs,
    n_particles=1000,
    adaptive=True,           # adapt proposal covariance
    adaptation_window=2000,  # adapt during first 2000 iterations
)
chain = pmmh.run(n_iterations=10000, seed=42)

diag = AcceptanceRateDiagnostic(chain)
diag.plot_rolling(window=500)  # should stabilize after adaptation window
```

!!! tip "Adaptation checklist"
    1. Run a short pilot chain (1000--2000 iterations) with a conservative proposal
    2. Use the pilot chain covariance to initialize the full run
    3. Enable adaptation for the first ~20% of iterations
    4. Verify that the rolling acceptance rate stabilizes in the target range
    5. Discard the adaptation period as burn-in

---

## Diagnosing Problems

### Acceptance Rate Too Low ($< 10\%$)

| Possible Cause | Diagnostic | Fix |
|----------------|-----------|-----|
| Proposal too wide | High rejection, low acceptance | Reduce `proposal_scale` |
| Too few particles | High $\text{Std}[\log \hat{p}]$ | Increase $N$ |
| Prior-posterior conflict | Many proposals have very low likelihood | Rethink priors |
| Multimodal posterior | Chain stuck in one mode | Use tempered or parallel-tempered PMMH |

### Acceptance Rate Too High ($> 40\%$)

| Possible Cause | Diagnostic | Fix |
|----------------|-----------|-----|
| Proposal too narrow | Small moves, high autocorrelation | Increase `proposal_scale` |
| Flat likelihood | Data is uninformative | Check model specification |
| Too many particles | $\text{Std}[\log \hat{p}] \ll 1$ | Reduce $N$ (save computation) |

---

## Complete Example

```python
import numpy as np
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.pmcmc import PMMH
from particlefilterbox.diagnostics import AcceptanceRateDiagnostic

# Setup
model = StochasticVolatility(variant="basic")
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=200, rng=rng)

# 1. Find optimal N
from particlefilterbox.diagnostics import ConvergenceDiagnostic
conv = ConvergenceDiagnostic(model, obs)
for n in [500, 1000, 2000]:
    report = conv.inter_run_variance(n_particles=n, n_runs=20, seed=42)
    print(f"N={n}: std(log-lik) = {report['ll_std']:.3f}")

# 2. Run PMMH with adaptive proposal
pmmh = PMMH(model, obs, n_particles=1000, adaptive=True)
chain = pmmh.run(n_iterations=10000, seed=42)

# 3. Diagnose acceptance rate
diag = AcceptanceRateDiagnostic(chain)
print(diag.summary())
diag.plot_rolling(window=500)

# 4. Verify target range
assert 0.10 < diag.overall_rate < 0.40, (
    f"Acceptance rate {diag.overall_rate:.3f} outside target range"
)
```

---

## API Summary

| Method | Description |
|--------|-------------|
| `AcceptanceRateDiagnostic(chain)` | Create diagnostic from PMMH chain |
| `.overall_rate` | Overall acceptance rate |
| `.summary()` | Comprehensive acceptance rate report |
| `.rolling_rate(window)` | Rolling acceptance rate array |
| `.plot_rolling(window, **kwargs)` | Plot rolling acceptance rate |
| `.plot_acceptance_vs_n(n_values, **kwargs)` | Acceptance rate as function of $N$ |
| `.plot_efficiency(n_values, **kwargs)` | ESS per second vs $N$ |
| `.phase_analysis(burnin)` | Compare burn-in vs post-burn-in rates |

---

## See Also

- [MCMC Convergence](mcmc-convergence.md) --- $\hat{R}$, Geweke, and Heidelberger-Welch
- [Mixing Diagnostics](mixing.md) --- ACF, ESS, and autocorrelation time
- [Convergence Diagnostic](convergence.md) --- choosing $N$ for the particle filter
- [PMMH](../user-guide/pmcmc/pmmh.md) --- the algorithm whose acceptance rate this diagnostic monitors
- [PMCMC Tuning](../user-guide/pmcmc/tuning.md) --- comprehensive guide to tuning PMMH including acceptance rate
- [Acceleration: Numba](../acceleration/numba.md) --- speed up the particle filter inside PMMH for faster iteration
- [Acceleration: GPU](../acceleration/gpu.md) --- GPU-accelerated likelihood evaluation for large $N$ in PMMH
