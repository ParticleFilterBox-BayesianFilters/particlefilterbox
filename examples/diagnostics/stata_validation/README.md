# Stata Validation: Particle Filter Diagnostics (Reference Limited)

## Overview

This directory contains a Stata reference script that computes **Kalman filter
diagnostics** for a linear-Gaussian state-space model. Stata has **no native
particle filter support**, so it cannot validate particle-filter-specific
diagnostics (ESS, weight degeneracy, resampling, Monte Carlo convergence).

What Stata *can* provide is the set of diagnostics naturally produced by the
Kalman recursion - namely **innovations** (prediction errors), **innovation
variance**, and the **log-likelihood**. In the linear-Gaussian model these
coincide (in expectation) with the corresponding particle filter quantities,
so they serve as a partial benchmark.

## Files

| File | Description |
|------|-------------|
| `benchmark_kalman_diagnostics.do` | Stata script: fits `sspace` model, computes innovations, normality test, autocorrelation, log-likelihood |
| `results_stata_diagnostics.csv` | Output CSV with time, y, innovation (generated after running the .do file) |

## How to Run

```stata
do benchmark_kalman_diagnostics.do
```

Requires Stata 14+ with the `sspace` command (StataBase / StataSE / StataMP).

## What the Script Computes

1. **Model fit** - local-level state-space model via `sspace`:
   - State:       `x_t = x_{t-1} + w_t`,  `w_t ~ N(0, Q)`
   - Observation: `y_t = x_t    + v_t`,   `v_t ~ N(0, R)`

2. **Innovations** - `innovation = y - y_pred` where `y_pred` is the
   one-step-ahead Kalman prediction.

3. **Innovation diagnostics**:
   - Mean / std / min / max (should be approximately zero-mean)
   - **Skewness-kurtosis test** (`sktest`) - innovations should be Gaussian
     if the model is well specified
   - **Autocorrelation** (`corrgram`) - innovations should be white noise

4. **Log-likelihood** (`e(ll)`) from the Kalman recursion. This is the
   exact likelihood in the linear-Gaussian case and is the target that the
   particle filter estimator `loglik_hat` should approximate.

5. **CSV export** for cross-validation against Python diagnostics.

## Why Stata Is Not a Full Diagnostics Reference

| Diagnostic                    | Stata `sspace` | particlefilterbox |
|-------------------------------|:--------------:|:-----------------:|
| Innovations / log-likelihood  |       Yes      |        Yes        |
| Innovation normality test     |       Yes      |        Yes        |
| Effective Sample Size (ESS)   |     **No**     |        Yes        |
| Weight degeneracy diagnostics |     **No**     |        Yes        |
| Resampling diagnostics        |     **No**     |        Yes        |
| MC convergence (vs N)         |     **No**     |        Yes        |
| Variance of log-lik estimator |     **No**     |        Yes        |
| Non-linear / non-Gaussian     |     **No**     |        Yes        |

**ESS, weight diagnostics, and resampling metrics are fundamentally
particle-filter concepts** - they do not exist inside the Kalman recursion,
so Stata cannot produce them.

## Cross-Validation Strategy

Use Stata for what it *can* benchmark:

1. Fit the same linear-Gaussian model in Python (via `KalmanFilter` *and*
   via a bootstrap particle filter).
2. Compare the Stata `log-likelihood` (`e(ll)`) against both.
3. Compare Stata innovations against the filter's one-step-ahead
   prediction residuals.

Everything particle-filter-specific (ESS trajectories, weight entropy,
degeneracy-based resampling triggers, MC variance studies) must be
cross-validated against **R (`pomp`)** - see `../R_validation/` - or
against the analytical Kalman ground truth.

## References

- Durbin & Koopman (2012). *Time Series Analysis by State Space Methods*.
- Stata Manual: `[TS] sspace`, `[TS] sspace postestimation`.
