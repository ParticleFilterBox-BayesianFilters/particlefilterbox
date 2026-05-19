---
title: Marginal Likelihood Estimation
description: "Marginal likelihood estimation for model comparison: particle filter estimates, bridge sampling, Bayes factors, and model selection"
---

# Marginal Likelihood Estimation

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `MarginalLikelihood` |
    | **Import** | `from particlefilterbox.diagnostics import MarginalLikelihood` |
    | **Input** | Model and observed data (optionally, MCMC chain) |
    | **Key method** | `.particle_filter_estimate(n_particles, n_repeats)` |
    | **Goal** | Estimate $\log p(y_{1:T})$ for model comparison |

## Overview

The marginal likelihood (or **evidence**) is the probability of the observed data under a model, integrating over all parameters:

$$
p(y_{1:T}) = \int p(y_{1:T} \mid \theta)\, p(\theta)\, d\theta
$$

This quantity is central to **Bayesian model selection**: models that explain the data well with fewer degrees of freedom (Occam's razor) receive higher marginal likelihood. The ratio of marginal likelihoods between two models is the **Bayes factor**.

In state-space models, the likelihood $p(y_{1:T} \mid \theta)$ itself requires integrating over latent states, making marginal likelihood estimation doubly challenging. The `MarginalLikelihood` class provides multiple estimation methods.

---

## Particle Filter Estimate

### Theory

The particle filter provides an **unbiased** estimate of the likelihood $p(y_{1:T} \mid \theta)$ as a byproduct of filtering:

$$
\hat{p}(y_{1:T} \mid \theta) = \prod_{t=1}^{T} \hat{p}(y_t \mid y_{1:t-1}, \theta) = \prod_{t=1}^{T} \frac{1}{N} \sum_{i=1}^{N} w_t^{(i)}
$$

where $w_t^{(i)}$ are the unnormalized importance weights. Taking logarithms:

$$
\log \hat{p}(y_{1:T} \mid \theta) = \sum_{t=1}^{T} \log\!\left(\frac{1}{N}\sum_{i=1}^{N} w_t^{(i)}\right)
$$

!!! note "Unbiased on the natural scale, biased on the log scale"
    The particle filter estimate is unbiased for $p(y_{1:T} \mid \theta)$, but $\log \hat{p}$ is **biased downward** by Jensen's inequality: $\mathbb{E}[\log \hat{p}] \leq \log p$. The bias is $O(1/N)$ and negligible for large $N$.

### Basic Usage

```python
from particlefilterbox.diagnostics import MarginalLikelihood

ml = MarginalLikelihood(model, obs)

# Single estimate at fixed parameters
log_ml = ml.particle_filter_estimate(
    theta=theta_mle,        # parameter values
    n_particles=5000,
    seed=42,
)
print(f"Log marginal likelihood: {log_ml:.2f}")
```

```text
Log marginal likelihood: -412.29
```

### Averaging Over Multiple Runs

Because each particle filter run is noisy, averaging over multiple independent runs reduces variance:

```python
# Multiple runs for a reliable estimate
result = ml.particle_filter_estimate(
    theta=theta_mle,
    n_particles=5000,
    n_repeats=20,
    seed=42,
)
print(result)
```

```text
=== Particle Filter Marginal Likelihood Estimate ===
Particles:    5000
Repeats:      20

Log-likelihood estimates:
  Mean:       -412.29
  Std:         0.063
  Std error:   0.014  (std / sqrt(n_repeats))
  95% CI:     [-412.32, -412.26]
```

### Integrating Over the Posterior

For the full marginal likelihood $p(y_{1:T})$, we need to average over the posterior distribution of $\theta$:

```python
# Estimate marginal likelihood using posterior samples
result = ml.particle_filter_estimate(
    chain=chain[2000:],     # posterior samples (post burn-in)
    n_particles=5000,
    n_samples=100,          # use 100 posterior samples
    seed=42,
)
print(f"Log p(y): {result['log_ml']:.2f} ± {result['std_error']:.3f}")
```

---

## Bridge Sampling

### Theory

Bridge sampling (Meng & Wong, 1996) provides a more efficient estimate of the marginal likelihood by using samples from both the prior and the posterior. The bridge sampling identity:

$$
p(y_{1:T}) = \frac{\mathbb{E}_{\text{posterior}}\!\left[h(\theta)\, p(\theta)\right]}{\mathbb{E}_{\text{prior}}\!\left[h(\theta)\, p(y_{1:T} \mid \theta)\right]}
$$

for an optimal bridge function $h(\theta)$. In practice, this is solved iteratively.

### Usage

```python
# Bridge sampling requires posterior samples
result = ml.bridge_sampling(
    chain=chain[2000:],     # posterior samples
    n_prior_samples=5000,   # samples from the prior
    n_particles=2000,       # for likelihood evaluation
    max_iter=100,           # bridge sampling iterations
    seed=42,
)
print(result)
```

```text
=== Bridge Sampling Estimate ===
Log marginal likelihood: -410.87
Standard error:           0.042
Iterations to converge:   23
Relative efficiency:      3.2x vs particle filter estimate
```

!!! tip "When to use bridge sampling"
    Bridge sampling is more efficient than the particle filter estimate when:

    - You have a good MCMC chain with many effective samples
    - The prior and posterior overlap substantially
    - You need a precise estimate (e.g., for Bayes factors between similar models)

    It is less reliable when the prior-posterior overlap is small or the posterior is multimodal.

---

## Harmonic Mean Estimator

### Theory

The harmonic mean estimator (Newton & Raftery, 1994) uses only posterior samples:

$$
\hat{p}(y_{1:T}) = \left[\frac{1}{S}\sum_{s=1}^{S} \frac{1}{p(y_{1:T} \mid \theta^{(s)})}\right]^{-1}
$$

!!! warning "Use with extreme caution"
    The harmonic mean estimator is **notoriously unstable** --- it has infinite variance when the posterior has lighter tails than the likelihood. It is included for completeness but should **not** be the primary method for model comparison. Use the particle filter estimate or bridge sampling instead.

```python
# Harmonic mean (for comparison only)
result = ml.harmonic_mean(
    chain=chain[2000:],
    n_particles=2000,
    seed=42,
)
print(f"Harmonic mean estimate: {result['log_ml']:.2f} (SE: {result['std_error']:.3f})")
```

---

## Bayes Factors

### Definition

The Bayes factor comparing model $\mathcal{M}_1$ to model $\mathcal{M}_2$ is:

$$
\text{BF}_{12} = \frac{p(y_{1:T} \mid \mathcal{M}_1)}{p(y_{1:T} \mid \mathcal{M}_2)}
$$

On the log scale:

$$
\log \text{BF}_{12} = \log p(y_{1:T} \mid \mathcal{M}_1) - \log p(y_{1:T} \mid \mathcal{M}_2)
$$

### Computing Bayes Factors

```python
from particlefilterbox.models import StochasticVolatility, LocalLevel
from particlefilterbox.diagnostics import MarginalLikelihood

# Two competing models
model1 = StochasticVolatility(variant="basic")
model2 = LocalLevel()

# Estimate marginal likelihoods
ml1 = MarginalLikelihood(model1, obs)
ml2 = MarginalLikelihood(model2, obs)

result1 = ml1.particle_filter_estimate(n_particles=5000, n_repeats=20, seed=42)
result2 = ml2.particle_filter_estimate(n_particles=5000, n_repeats=20, seed=42)

# Bayes factor
bf = MarginalLikelihood.bayes_factor(result1, result2)
print(bf)
```

```text
=== Bayes Factor ===
Model 1: StochasticVolatility (basic)
Model 2: LocalLevel

Log ML (Model 1):  -412.29 ± 0.063
Log ML (Model 2):  -438.51 ± 0.087

Log Bayes Factor:   26.22
Bayes Factor:       2.42e+11

Interpretation: Decisive evidence in favor of Model 1
```

### Interpretation Scale

!!! tip "Kass & Raftery (1995) scale for Bayes factors"
    | $\log_{10} \text{BF}_{12}$ | $2 \ln \text{BF}_{12}$ | Evidence for $\mathcal{M}_1$ |
    |---------------------------|------------------------|------------------------------|
    | $0 - 0.5$ | $0 - 2$ | Barely worth mentioning |
    | $0.5 - 1$ | $2 - 6$ | Positive |
    | $1 - 2$ | $6 - 10$ | Strong |
    | $> 2$ | $> 10$ | Decisive |

### Model Comparison Table

```python
# Compare multiple models
models = {
    "SV-basic": StochasticVolatility(variant="basic"),
    "SV-leverage": StochasticVolatility(variant="leverage"),
    "SV-jump": StochasticVolatility(variant="jump"),
    "LocalLevel": LocalLevel(),
}

table = MarginalLikelihood.compare_models(
    models=models,
    observations=obs,
    n_particles=5000,
    n_repeats=20,
    seed=42,
)
print(table)
```

```text
=== Model Comparison ===
Model          | Log ML     | SE    | BF vs best | Rank
---------------+------------+-------+------------+-----
SV-leverage    | -405.12    | 0.071 |    1.00    |  1
SV-jump        | -408.34    | 0.083 |    0.04    |  2
SV-basic       | -412.29    | 0.063 |    0.001   |  3
LocalLevel     | -438.51    | 0.087 |   <0.001   |  4
```

---

## Standard Error of the Estimate

### Sources of Error

The marginal likelihood estimate has two sources of uncertainty:

1. **Monte Carlo error** from the particle filter (depends on $N$)
2. **Posterior sampling error** from finite MCMC chains (depends on ESS)

```python
# Detailed error analysis
error_analysis = ml.error_analysis(
    chain=chain[2000:],
    n_particles_values=[1000, 2000, 5000],
    n_repeats=20,
    seed=42,
)
print(error_analysis)
```

```text
=== Error Analysis ===
     N   | Mean log ML | MC Std  | SE of mean | Total uncertainty
---------+-------------+---------+------------+------------------
    1000 |   -412.38   |  0.312  |   0.070    |     0.320
    2000 |   -412.32   |  0.158  |   0.035    |     0.162
    5000 |   -412.29   |  0.063  |   0.014    |     0.065
```

### Reliability for Model Comparison

!!! warning "Ensure the standard error is small relative to the Bayes factor"
    If $\log \text{BF}_{12} = 3.5$ but the standard error of each log ML is $2.0$, the Bayes factor is unreliable. A rule of thumb:

    $$
    \text{SE}[\log \text{BF}_{12}] = \sqrt{\text{SE}_1^2 + \text{SE}_2^2} \ll |\log \text{BF}_{12}|
    $$

    If the standard error of the log Bayes factor exceeds half the log Bayes factor itself, increase $N$ or $n_{\text{repeats}}$.

---

## Complete Example

```python
import numpy as np
from particlefilterbox.models import StochasticVolatility, LocalLevel
from particlefilterbox.pmcmc import PMMH
from particlefilterbox.diagnostics import MarginalLikelihood

# Generate data from the true model
true_model = StochasticVolatility(variant="leverage")
rng = np.random.default_rng(42)
states, obs = true_model.simulate(n_obs=300, rng=rng)

# Define competing models
models = {
    "SV-basic": StochasticVolatility(variant="basic"),
    "SV-leverage": StochasticVolatility(variant="leverage"),
    "LocalLevel": LocalLevel(),
}

# 1. Quick comparison via particle filter estimates
table = MarginalLikelihood.compare_models(
    models=models,
    observations=obs,
    n_particles=5000,
    n_repeats=20,
    seed=42,
)
print(table)

# 2. More precise estimate for the top two models via bridge sampling
for name in ["SV-leverage", "SV-basic"]:
    model = models[name]
    pmmh = PMMH(model, obs, n_particles=1000, adaptive=True)
    chain = pmmh.run(n_iterations=10000, seed=42)
    
    ml = MarginalLikelihood(model, obs)
    result = ml.bridge_sampling(chain=chain[2000:], seed=42)
    print(f"{name}: log ML = {result['log_ml']:.2f} ± {result['std_error']:.3f}")

# 3. Final Bayes factor
bf = MarginalLikelihood.bayes_factor(result_leverage, result_basic)
print(f"\nLog BF (SV-leverage vs SV-basic): {bf['log_bf']:.2f}")
print(f"Interpretation: {bf['interpretation']}")
```

---

## API Summary

| Method | Description |
|--------|-------------|
| `MarginalLikelihood(model, obs)` | Create estimator for a model and observations |
| `.particle_filter_estimate(theta, n_particles, n_repeats)` | PF-based log-likelihood estimate at fixed $\theta$ |
| `.particle_filter_estimate(chain, n_particles, n_samples)` | PF-based estimate averaged over posterior samples |
| `.bridge_sampling(chain, n_prior_samples, **kwargs)` | Bridge sampling estimate using posterior chain |
| `.harmonic_mean(chain, n_particles)` | Harmonic mean estimator (use with caution) |
| `.error_analysis(chain, n_particles_values)` | Uncertainty decomposition |
| `.bayes_factor(result1, result2)` | Bayes factor from two ML estimates |
| `.compare_models(models, observations, **kwargs)` | Compare multiple models in a table |

---

## See Also

- [Posterior Predictive Checks](predictive-checks.md) --- qualitative model assessment
- [Convergence Diagnostic](convergence.md) --- calibrating $N$ for reliable log-likelihood estimates
- [MCMC Convergence](mcmc-convergence.md) --- ensure chains are reliable before bridge sampling
- [PMMH](../user-guide/pmcmc/pmmh.md) --- generate posterior chains for marginal likelihood estimation
- [SMC²](../user-guide/smc/smc-squared.md) --- alternative approach to marginal likelihood via SMC
- [Models Overview](../user-guide/models/index.md) --- pre-built models for model comparison studies
- [Theory: PMCMC](../theory/pmcmc-theory.md) --- theoretical foundations of particle MCMC methods
