# Stata Validation: PMCMC (Reference Only)

## Overview

This directory contains a Stata reference script for stochastic volatility (SV)
model parameter estimation. **Stata does not have native PMCMC support**, so
this validation uses an approximate linearization approach via `sspace`.

## Approach

The standard SV model is:

```
y_t = exp(h_t / 2) * eps_t,    eps_t ~ N(0, 1)
h_t = mu + phi * (h_{t-1} - mu) + sigma_h * eta_t,    eta_t ~ N(0, 1)
```

The linearized approximation transforms the observation equation by squaring
and taking logs:

```
log(y_t^2) = h_t + log(eps_t^2)
```

where `log(eps_t^2) ~ log-chi2(1)` with mean -1.27 and variance pi^2/2 ~ 4.93.

This linearization is then estimated via Stata's `sspace` command as a linear
state-space model using the Kalman filter.

## Limitations and Biases

**Important**: Results from this approach are **approximate** and should NOT be
treated as ground truth. Known limitations include:

1. **No PMCMC in Stata**: Stata lacks Particle MCMC methods (PMMH, Particle
   Gibbs, PGAS). This script provides only an MLE-based approximate benchmark.

2. **Linearization bias**: The log-squared transformation introduces significant
   bias because `log(eps_t^2)` is non-Gaussian (log-chi-squared). The Kalman
   filter assumes Gaussian errors, leading to biased parameter estimates.

3. **Handling of zeros**: Observations near zero require a small constant
   (`+0.001`) to avoid `log(0)`, introducing additional approximation error.

4. **No mixture approximation**: The Kim, Shephard & Chib (1998) mixture-of-
   Gaussians approximation for `log(eps_t^2)` is not used here, which would
   improve accuracy but is not straightforward in `sspace`.

5. **Parameter interpretation**: The `sspace` parameterization maps indirectly
   to the SV model parameters (mu, phi, sigma_h), requiring care in
   interpreting the estimated coefficients.

## Files

| File | Description |
|------|-------------|
| `benchmark_sv_approximation.do` | Stata script for approximate SV estimation via `sspace` |
| `results_stata_sv_approx.csv` | Output data (generated after running the `.do` file) |

## Usage

```stata
do benchmark_sv_approximation.do
```

Requires: Stata 14+ with `sspace` command.

## Recommended Alternatives

For accurate SV model estimation and PMCMC, use:

- **R**: `pomp`, `nimbleSMC`, or `stochvol` packages
- **Python**: `particlefilterbox` (this library)

These tools implement proper particle filtering and PMCMC methods that handle
the non-linear, non-Gaussian nature of the SV model without linearization.
