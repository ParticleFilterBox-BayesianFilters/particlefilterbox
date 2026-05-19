# Stata Validation: SMC (Reference Only)

## Overview

This directory contains a Stata reference script for benchmarking SMC parameter
estimation results. **Stata does not have native SMC samplers**, so this
validation is limited to Maximum Likelihood Estimation (MLE) via the `sspace`
command as a benchmark.

## Limitations

- **No SMC samplers**: Stata does not provide Sequential Monte Carlo samplers.
- **No IBIS or waste-free SMC**: These advanced SMC variants are not available.
- **MLE only**: The `sspace` command estimates parameters via the Kalman filter
  and numerical optimization (MLE), not Bayesian posterior sampling.
- **Stochastic Volatility**: For SV models, `sspace` can only handle linearized
  approximations, which introduces significant bias compared to proper
  particle-based inference.

## What This Provides

The `sspace` MLE estimates serve as a **point of comparison** for SMC results:

- Under a vague (diffuse) prior, SMC posterior means should converge to MLE
  estimates as the number of particles and data grow.
- The MLE log-likelihood provides a reference value for marginal likelihood
  approximations from SMC.
- AIC/BIC from MLE can be compared against SMC-based model selection criteria.

## Files

| File | Description |
|------|-------------|
| `benchmark_mle_reference.do` | Stata script using `sspace` to estimate linear-Gaussian model via MLE |
| `results_stata_mle.csv` | Exported MLE parameter estimates (generated after running the `.do` file) |

## Usage

```stata
do benchmark_mle_reference.do
```

Requires Stata 14+ with the `sspace` command available.

## Comparison with particlefilterbox

| Feature | Stata (`sspace`) | particlefilterbox (SMC) |
|---------|------------------|------------------------|
| Estimation method | MLE (Kalman filter) | Bayesian (SMC sampler) |
| Linear-Gaussian | Full support | Full support |
| Stochastic Volatility | Linearized approx. only | Full nonlinear support |
| Posterior distribution | Point estimate + SE | Full posterior samples |
| Model comparison | AIC/BIC | Marginal likelihood |
| SMC variants | None | IBIS, waste-free, standard |
