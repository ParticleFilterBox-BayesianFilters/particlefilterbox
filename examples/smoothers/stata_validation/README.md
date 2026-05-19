# Stata Validation: Kalman Smoother Benchmark

## Overview

This directory contains a Stata reference script that computes the **Kalman smoother**
for a linear-Gaussian state-space model using Stata's `sspace` command. The Kalman
smoother provides the **exact analytical solution** that particle smoothers
(FFBSm, FFBSi, two-filter) should approximate in the linear-Gaussian case.

## Files

| File | Description |
|------|-------------|
| `benchmark_kalman_smoother.do` | Stata script: fits sspace model, computes Kalman filter and smoother estimates |
| `results_stata_smoother.csv` | Output CSV with filtered and smoothed state estimates (generated after running the .do file) |

## How to Run

```stata
do benchmark_kalman_smoother.do
```

Requires Stata 14+ with the `sspace` command available.

## Kalman Smoother vs Kalman Filter

The Kalman **filter** estimates the state using observations up to time *t*:

    E[x_t | y_1, ..., y_t]

The Kalman **smoother** uses **all** observations (past and future):

    E[x_t | y_1, ..., y_T]

Because the smoother incorporates more information, it always achieves
**RMSE(smoother) <= RMSE(filter)** for linear-Gaussian models. This property
is used as a sanity check in the script.

## Limitations of Stata for Particle Smoothing

1. **No particle smoothers**: Stata does not implement FFBSm (Forward Filtering
   Backward Smoothing), FFBSi (Forward Filtering Backward Simulation),
   two-filter smoothers, or any other sequential Monte Carlo smoothing algorithm.

2. **Linear-Gaussian only**: The `sspace` command is restricted to linear-Gaussian
   state-space models. Non-linear models (e.g., stochastic volatility) cannot
   be estimated or smoothed with `sspace`.

3. **No stochastic volatility smoothing**: The SV model requires non-linear
   filtering/smoothing methods. Stata has no built-in support for this.

4. **Reference only**: This script serves as an analytical benchmark for the
   `particlefilterbox` library. The Kalman smoother RMSE is the theoretical
   lower bound that particle smoothers should approach as the number of
   particles increases.

## Comparison with particlefilterbox

| Feature | Stata `sspace` | particlefilterbox |
|---------|----------------|-------------------|
| Kalman filter | Yes | Yes (via KalmanFilter) |
| Kalman smoother | Yes (`predict, sstate`) | Yes (via RTS smoother) |
| Bootstrap PF smoother | No | Yes (FFBSm, FFBSi) |
| SIR smoother | No | Yes |
| Two-filter smoother | No | Yes |
| Stochastic volatility | No | Yes |
| Non-linear models | No | Yes |
