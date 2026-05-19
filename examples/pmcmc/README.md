# PMCMC Examples (Particle Markov Chain Monte Carlo)

This directory contains examples and notebooks for three Particle MCMC algorithms
used for Bayesian parameter estimation in nonlinear state-space models.

## Algorithms

### PMMH (Particle Marginal Metropolis-Hastings)

Andrieu, Doucet & Holenstein (2010). Uses a particle filter to estimate the
marginal likelihood within a Metropolis-Hastings step. The PF provides an unbiased
estimate of p(y_{1:T} | theta), which is plugged into the MH acceptance ratio.
This yields exact (in the limit) samples from the posterior p(theta | y_{1:T}).

### Particle Gibbs (PG)

Andrieu, Doucet & Holenstein (2010). A Gibbs sampler where the latent states are
updated using a Conditional SMC (CSMC) sweep that conditions on a reference
trajectory from the previous iteration. Parameters are sampled from their full
conditionals given the current state trajectory. Guarantees the target posterior
as its invariant distribution.

### PGAS (Particle Gibbs with Ancestor Sampling)

Lindsten, Jordan & Schon (2014). Extends Particle Gibbs with ancestor sampling,
which improves mixing by allowing the reference trajectory's ancestry to be
reconnected at each time step. This breaks the path degeneracy problem that
can cause slow mixing in standard Particle Gibbs.

## Datasets

- `data/simulated_sv.csv` - Standard stochastic volatility model (from bootstrap_sir examples)
- `data/simulated_sv_leverage.csv` - SV model with leverage effect (rho = -0.5)

### SV with Leverage Model

The leverage model introduces correlation between return and volatility innovations:

```
h_t = mu + phi * (h_{t-1} - mu) + sigma_h * eta_t
y_t = exp(h_t / 2) * eps_t
Corr(eta_t, eps_t) = rho
```

Parameters (default):
- `mu = -1.0` (log-variance level)
- `phi = 0.97` (persistence)
- `sigma_h = 0.15` (volatility of volatility)
- `rho = -0.5` (leverage effect: negative correlation between returns and volatility)
- `seed = 42` (reproducibility)

## Directory Structure

```
pmcmc/
├── README.md
├── data/               # Datasets and generation scripts
├── notebooks/          # Tutorial notebooks
├── solutions/          # Complete solution notebooks
├── R_validation/       # Cross-validation with R (pomp, nimbleSMC)
└── stata_validation/   # Reference comparisons with Stata
```

## References

- Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov chain Monte
  Carlo methods. *Journal of the Royal Statistical Society: Series B*, 72(3), 269-342.
- Lindsten, F., Jordan, M. I., & Schon, T. B. (2014). Particle Gibbs with
  ancestor sampling. *Journal of Machine Learning Research*, 15, 2145-2184.
