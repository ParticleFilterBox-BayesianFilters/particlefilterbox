# Stochastic Volatility Examples

Particle filtering applied to stochastic volatility (SV) models in finance.
These notebooks demonstrate estimation of latent log-volatility using
sequential Monte Carlo methods.

## Models

### SV Basic

The canonical stochastic volatility model:

```
h_t = mu + phi * (h_{t-1} - mu) + sigma_h * eta_t,   eta_t ~ N(0,1)
y_t = exp(h_t / 2) * eps_t,                           eps_t ~ N(0,1)
```

Parameters: persistence `phi`, volatility-of-volatility `sigma_h`, level `mu`.

### SV with Leverage

Extends the basic model by correlating return and volatility shocks:

```
Corr(eta_t, eps_t) = rho,   typically rho < 0
```

Negative `rho` captures the leverage effect: negative returns tend to
increase future volatility.

### SV with Jumps

Adds rare, large price movements via a Poisson jump component:

```
y_t = exp(h_t / 2) * eps_t + J_t * Z_t
J_t ~ Bernoulli(lambda),  Z_t ~ N(mu_j, sigma_j^2)
```

### Factor SV

Multivariate extension with common latent volatility factors driving
multiple asset return series simultaneously. Useful for portfolio risk
and cross-asset dependence modeling.

## Datasets

| File | Description |
|------|-------------|
| `data/sp500_returns.csv` | 2500 synthetic daily returns calibrated to S&P 500 (~1% daily vol) |
| `data/simulated_sv.csv` | Simulated SV basic data (from Phase 1) |
| `data/simulated_sv_leverage.csv` | Simulated SV leverage data (from Phase 5) |

## Directory Structure

```
stochastic_volatility/
├── data/               # Datasets and generation scripts
├── notebooks/          # Jupyter notebooks (exercises)
├── solutions/          # Notebook solutions
├── R_validation/       # R scripts for cross-validation (stochvol, bsvars)
└── stata_validation/   # Stata scripts for cross-validation (sspace)
```
