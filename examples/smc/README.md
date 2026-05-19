# SMC (Sequential Monte Carlo) Samplers - Examples

This directory contains notebooks and validation scripts for SMC sampler
algorithms implemented in `particlefilterbox`.

## Overview

SMC samplers generalize particle filters to sample from static distributions
via tempering. Instead of filtering latent states over time, SMC samplers
construct a sequence of intermediate distributions bridging a prior to a
posterior, using importance sampling, resampling, and MCMC mutation steps.

## Algorithms

### SMC Sampler (Del Moral et al., 2006)

Constructs a sequence of tempered distributions
$\pi_t(\theta) \propto p(\theta) \, p(y \mid \theta)^{\gamma_t}$
where $0 = \gamma_0 < \gamma_1 < \cdots < \gamma_T = 1$. Particles are
propagated through this sequence using importance weighting and MCMC
mutation kernels (typically random-walk Metropolis-Hastings). The tempering
schedule can be fixed or chosen adaptively to control the effective sample
size (ESS).

### SMC-squared (Chopin et al., 2013)

SMC$^2$ combines two layers of SMC: an outer SMC sampler over the parameter
space and an inner particle filter for each parameter particle to estimate
the likelihood $p(y_{1:t} \mid \theta)$. This enables fully Bayesian
inference in state-space models with unknown static parameters. When ESS
drops below a threshold, parameter particles are rejuvenated via PMCMC
(particle MCMC) moves.

### IBIS - Iterated Batch Importance Sampling (Chopin, 2002)

IBIS processes observations sequentially: at each step, particle weights are
updated by incorporating the new data point's likelihood, and when the ESS
drops below a threshold, particles are rejuvenated via MCMC moves targeting
the current posterior $p(\theta \mid y_{1:t})$. IBIS is an online algorithm
that naturally handles streaming data.

### Waste-free SMC (Dau & Chopin, 2022)

Standard SMC discards particles that are not resampled, wasting computational
effort. Waste-free SMC reuses all MCMC intermediate states (not just the
final ones) as the particle population for the next iteration. This leads to
improved efficiency with reduced variance of normalizing constant estimates,
at no additional computational cost.

## Directory Structure

```
smc/
├── README.md                 # This file
├── data/                     # Datasets (symlinks to bootstrap_sir/data)
│   ├── simulated_sv.csv
│   └── simulated_linear_gaussian.csv
├── notebooks/                # Tutorial notebooks
├── solutions/                # Reference solutions
├── R_validation/             # Cross-validation with R packages
└── stata_validation/         # Cross-validation with Stata
```

## Datasets

Both datasets are shared with the Bootstrap/SIR examples (FASE 1):

- **simulated_sv.csv** - Simulated stochastic volatility data with columns
  `t`, `h_true` (log-volatility), `y_obs` (observed returns).
- **simulated_linear_gaussian.csv** - Simulated linear-Gaussian state-space
  data with columns `t`, `x_true` (latent state), `y_obs` (observation).

## References

- Chopin, N. (2002). A sequential particle filter method for static models.
  *Biometrika*, 89(3), 539-552.
- Del Moral, P., Doucet, A., & Jasra, A. (2006). Sequential Monte Carlo
  samplers. *JRSS-B*, 68(3), 411-436.
- Chopin, N., Jacob, P. E., & Papaspiliopoulos, O. (2013). SMC$^2$: an
  efficient algorithm for sequential analysis of state space models.
  *JRSS-B*, 75(3), 397-426.
- Dau, H. D. & Chopin, N. (2022). Waste-free Sequential Monte Carlo.
  *JRSS-B*, 84(1), 114-148.
