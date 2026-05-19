---
title: Mixing Diagnostics
description: "MCMC chain mixing diagnostics: autocorrelation function, effective sample size, integrated autocorrelation time, and PMMH vs PGAS comparison"
---

# Mixing Diagnostics

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `MixingDiagnostic` |
    | **Import** | `from particlefilterbox.diagnostics import MixingDiagnostic` |
    | **Input** | Single MCMC chain from PMMH, Particle Gibbs, or PGAS |
    | **Key metrics** | ACF, ESS (chain), IAT |
    | **Goal** | Assess how efficiently the chain explores the posterior |

## Overview

Convergence tells you whether the chain has reached the right distribution. **Mixing** tells you how efficiently it explores that distribution once it has converged. A chain can be converged (stable $\hat{R}$) but poorly mixed (high autocorrelation, low effective sample size).

The mixing diagnostic answers three questions:

1. **How correlated are successive samples?** --- Autocorrelation Function (ACF)
2. **How many independent samples do we effectively have?** --- Effective Sample Size (ESS)
3. **How many iterations between independent samples?** --- Integrated Autocorrelation Time (IAT)

!!! note "MCMC ESS vs Particle Filter ESS"
    The ESS in this page refers to the **MCMC chain** ESS --- how many independent posterior samples the chain contains. This is different from the particle filter ESS ([ESS Diagnostic](ess-diagnostic.md)), which measures particle diversity at each time step within a single filter run.

---

## Basic Usage

```python
from particlefilterbox.diagnostics import MixingDiagnostic

# From a single post-burn-in MCMC chain
# chain shape: (n_iterations, n_parameters)
diag = MixingDiagnostic(chain)

print(diag.summary())
```

```text
=== Mixing Diagnostic Summary ===
Iterations:    8000 (post burn-in)
Parameters:    3

Parameter  |    ESS    | ESS/iter |   IAT   | ACF(lag=1) | ACF(lag=10)
-----------+-----------+----------+---------+------------+------------
mu         |   2847.3  |  0.356   |   2.81  |   0.641    |   0.087
phi        |    892.1  |  0.112   |   8.97  |   0.884    |   0.412
sigma_v    |   1523.6  |  0.190   |   5.25  |   0.782    |   0.218

Overall ESS (minimum): 892.1  (phi)
```

---

## Autocorrelation Function (ACF)

### Theory

The autocorrelation function at lag $k$ measures the correlation between samples $k$ steps apart:

$$
\rho(k) = \frac{\text{Cov}(\theta_t, \theta_{t+k})}{\text{Var}(\theta_t)}
$$

For a well-mixing chain, $\rho(k)$ decays quickly to zero. For a poorly-mixing chain, $\rho(k)$ stays positive for many lags.

### Computing and Plotting ACF

```python
# ACF for all parameters
acf_values = diag.acf(max_lag=100)

# ACF plot
diag.acf_plot(max_lag=100)
```

```python
# ACF for a specific parameter
diag.acf_plot(param="phi", max_lag=200, show_ci=True)
```

The ACF plot shows:

- **Bars**: Autocorrelation at each lag
- **Blue dashed lines**: 95% confidence interval for white noise ($\pm 1.96 / \sqrt{n}$)
- A well-mixing chain: ACF drops below the confidence band within 20--50 lags
- A poorly-mixing chain: ACF stays above the band for hundreds of lags

### Interpreting ACF

!!! tip "ACF rules of thumb"
    | ACF behavior | Mixing quality | Typical cause |
    |-------------|---------------|---------------|
    | Drops to 0 by lag 10--20 | Excellent | Well-tuned proposal |
    | Drops to 0 by lag 50--100 | Acceptable | Moderate correlations, increase iterations |
    | Stays positive beyond lag 100 | Poor | Proposal too narrow, or highly correlated posterior |
    | Oscillates around 0 | Good but noisy | Normal for finite chains |
    | Alternates sign (negative at odd lags) | Over-dispersed proposals | Reduce proposal scale slightly |

---

## Effective Sample Size (ESS)

### Theory

The effective sample size accounts for autocorrelation in the chain. For $n$ MCMC iterations with integrated autocorrelation time $\tau$:

$$
\text{ESS} = \frac{n}{1 + 2\sum_{k=1}^{\infty} \rho(k)} = \frac{n}{\tau}
$$

where $\tau = 1 + 2\sum_{k=1}^{\infty} \rho(k)$ is the **integrated autocorrelation time**.

If samples were independent ($\rho(k) = 0$ for $k > 0$), then $\text{ESS} = n$. In practice, $\text{ESS} < n$ due to autocorrelation.

### Computing ESS

```python
# ESS for all parameters
ess = diag.ess()
print(ess)
```

```text
{'mu': 2847.3, 'phi': 892.1, 'sigma_v': 1523.6}
```

```python
# ESS per iteration (efficiency metric)
ess_per_iter = diag.ess_per_iteration()
print(ess_per_iter)
```

```text
{'mu': 0.356, 'phi': 0.112, 'sigma_v': 0.190}
```

### Interpreting ESS

!!! tip "ESS thresholds for inference"
    | ESS value | Quality | Interpretation |
    |-----------|---------|---------------|
    | $> 1000$ | Excellent | Reliable posterior summaries and credible intervals |
    | $400 - 1000$ | Good | Point estimates reliable, tail quantiles may be noisy |
    | $100 - 400$ | Marginal | Posterior mean reliable, credible intervals imprecise |
    | $< 100$ | Poor | Estimates unreliable --- run chain longer |

!!! warning "Report the minimum ESS across parameters"
    The bottleneck parameter (lowest ESS) determines the reliability of joint inference. Always report $\text{ESS}_{\min} = \min_i \text{ESS}(\theta_i)$ as the effective sample size of your analysis.

---

## Integrated Autocorrelation Time (IAT)

### Theory

The integrated autocorrelation time (IAT) is the number of iterations needed to produce one effectively independent sample:

$$
\tau = 1 + 2\sum_{k=1}^{K} \rho(k)
$$

The sum is truncated at some cutoff $K$ to avoid noise from high-lag estimates. Common choices:

- **Initial positive sequence** (Geyer, 1992): Truncate when consecutive pairs of autocorrelations sum to a negative value
- **Window method**: $K = c \cdot \hat{\tau}$ where $c = 5$ (Sokal, 1997)

### Computing IAT

```python
# IAT for all parameters
iat = diag.iat()
print(iat)
```

```text
{'mu': 2.81, 'phi': 8.97, 'sigma_v': 5.25}
```

```python
# IAT with specific truncation method
iat = diag.iat(method="geyer")    # initial positive sequence
iat = diag.iat(method="sokal")    # Sokal's windowed estimator
```

### Interpreting IAT

| IAT value | Interpretation | Implications |
|-----------|---------------|-------------|
| $\tau < 5$ | Excellent mixing | 1000 iterations $\approx$ 200+ independent samples |
| $5 \leq \tau < 20$ | Good mixing | Typical for well-tuned PMMH |
| $20 \leq \tau < 100$ | Moderate mixing | May need longer chains |
| $\tau > 100$ | Poor mixing | Fundamental tuning issue |

!!! note "IAT relates ESS and chain length"
    The three quantities are linked: $\text{ESS} = n / \tau$. So if you need $\text{ESS} = 1000$ and $\tau = 10$, you need $n = 10{,}000$ iterations.

---

## Comparing PMMH vs PGAS Mixing

PMMH and PGAS have fundamentally different mixing properties:

- **PMMH**: Proposes new parameters and runs a fresh particle filter. Mixing depends on proposal scale and $\text{Std}[\log \hat{p}]$.
- **PGAS**: Uses ancestor sampling to update the trajectory jointly. Often mixes faster for the latent states but may be slower for parameters.

```python
from particlefilterbox.diagnostics import MixingDiagnostic

# Compare mixing from two algorithms
diag_pmmh = MixingDiagnostic(chain_pmmh)
diag_pgas = MixingDiagnostic(chain_pgas)

# Side-by-side comparison
MixingDiagnostic.compare(
    {"PMMH": chain_pmmh, "PGAS": chain_pgas},
    metrics=["ess", "iat", "acf"],
)
```

```text
=== Mixing Comparison ===
            |     PMMH      |     PGAS
Parameter   | ESS   | IAT   | ESS   | IAT
------------+-------+-------+-------+-------
mu          | 2847  |  2.81 | 3124  |  2.56
phi         |  892  |  8.97 | 2456  |  3.26
sigma_v     | 1524  |  5.25 | 1987  |  4.03

Min ESS:    |  892  |       | 1987  |
ESS/iter:   | 0.112 |       | 0.248 |
```

### When to Use Each

!!! tip "PMMH vs PGAS: mixing trade-offs"
    | Scenario | Better algorithm | Why |
    |----------|-----------------|-----|
    | Few parameters, long time series | PGAS | Ancestor sampling avoids path degeneracy |
    | Many parameters | PMMH | Gibbs updates in PG can be slow for many parameters |
    | Highly correlated parameters | PMMH with block proposals | Can propose correlated moves |
    | Multimodal posterior | Neither --- use tempered methods | Both can get stuck |

---

## What To Do With Poor Mixing

### Diagnosis Tree

```python
# Identify the bottleneck
diag = MixingDiagnostic(chain)
bottleneck = diag.worst_parameter()
print(f"Worst mixing: {bottleneck['name']} (ESS={bottleneck['ess']:.0f}, IAT={bottleneck['iat']:.1f})")
```

| Symptom | Likely Cause | Remedy |
|---------|-------------|--------|
| All parameters have high IAT | Global proposal issue | Adapt proposal covariance, check $N$ |
| One parameter much worse than others | Strong posterior correlation with that parameter | Reparameterize or use block proposals |
| ESS near chain length (very low IAT) | Proposal is too narrow | Increase proposal scale |
| ESS very low, high IAT | Proposal is too wide or $N$ too small | Decrease proposal scale or increase $N$ |
| PMMH ESS low but PGAS ESS high | Likelihood noise dominating | Increase $N$ for PMMH or switch to PGAS |

### Improving Mixing: Step by Step

1. **Check acceptance rate**: Should be 15--30% for PMMH. See [Acceptance Rate](acceptance-rate.md).
2. **Tune proposal scale**: Use pilot runs to find the optimal scale.
3. **Adapt proposal covariance**: Use the empirical posterior covariance from a pilot chain.
4. **Increase $N$**: Reduce likelihood noise (but costs computation).
5. **Reparameterize**: Non-centered parameterizations can dramatically improve mixing.
6. **Switch algorithm**: If PMMH mixes poorly, try PGAS (or vice versa).

```python
# Example: reparameterization for better mixing
# Instead of sampling (mu, phi, sigma) directly, sample
# transformed parameters and map back

model_reparam = StochasticVolatility(
    variant="basic",
    parameterization="non_centered",  # x_t = mu + sigma * z_t
)
```

---

## Complete Example

```python
import numpy as np
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.pmcmc import PMMH, PGAS
from particlefilterbox.diagnostics import MixingDiagnostic

# Setup
model = StochasticVolatility(variant="basic")
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=200, rng=rng)

# Run PMMH
pmmh = PMMH(model, obs, n_particles=1000, adaptive=True)
chain_pmmh = pmmh.run(n_iterations=10000, seed=42)

# Run PGAS
pgas = PGAS(model, obs, n_particles=500)
chain_pgas = pgas.run(n_iterations=10000, seed=42)

# Mixing diagnostics (discard burn-in)
diag_pmmh = MixingDiagnostic(chain_pmmh[2000:])
diag_pgas = MixingDiagnostic(chain_pgas[2000:])

# 1. Summary
print("=== PMMH ===")
print(diag_pmmh.summary())
print("\n=== PGAS ===")
print(diag_pgas.summary())

# 2. ACF comparison
diag_pmmh.acf_plot(max_lag=100)
diag_pgas.acf_plot(max_lag=100)

# 3. Side-by-side comparison
MixingDiagnostic.compare(
    {"PMMH": chain_pmmh[2000:], "PGAS": chain_pgas[2000:]},
)
```

---

## API Summary

| Method | Description |
|--------|-------------|
| `MixingDiagnostic(chain)` | Create diagnostic from chain array `(n_iter, n_params)` |
| `.acf(max_lag, param)` | Autocorrelation function values |
| `.acf_plot(max_lag, param, **kwargs)` | Plot ACF with confidence bands |
| `.ess(param)` | Effective sample size per parameter |
| `.ess_per_iteration(param)` | ESS / number of iterations |
| `.iat(param, method)` | Integrated autocorrelation time |
| `.summary()` | Comprehensive mixing report |
| `.worst_parameter()` | Parameter with lowest ESS |
| `.compare(chains_dict, metrics)` | Compare mixing across algorithms |

---

## See Also

- [MCMC Convergence](mcmc-convergence.md) --- has the chain converged?
- [Acceptance Rate](acceptance-rate.md) --- PMMH accept/reject tuning
- [ESS Diagnostic](ess-diagnostic.md) --- particle filter ESS (per time step)
- [Convergence Diagnostic](convergence.md) --- choosing the number of particles $N$
- [PMMH](../user-guide/pmcmc/pmmh.md) --- Particle Marginal Metropolis-Hastings algorithm
- [PGAS](../user-guide/pmcmc/pgas.md) --- Particle Gibbs with Ancestor Sampling (often better mixing)
- [PMCMC Tuning](../user-guide/pmcmc/tuning.md) --- step-by-step tuning guide for PMCMC
- [Acceleration: Parallel](../acceleration/parallel.md) --- run multiple MCMC chains in parallel for faster convergence assessment
