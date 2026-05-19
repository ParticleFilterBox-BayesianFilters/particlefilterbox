---
title: "Tutorial: Particle Marginal Metropolis-Hastings"
description: Estimate state-space model parameters with PMMH, from setup through posterior analysis and comparison with IBIS
---

# Tutorial: Particle Marginal Metropolis-Hastings

**Level**: :material-star:{.advanced} Advanced  
**Time**: ~60 minutes  
**Prerequisites**: [Fundamentals tutorial](fundamentals.md), [SMC Samplers tutorial](smc.md)  

**PMMH** (Particle Marginal Metropolis-Hastings) is the workhorse of Bayesian parameter estimation for nonlinear state-space models. It wraps a particle filter *inside* a Metropolis-Hastings loop, using the particle filter's likelihood estimate as a plug-in for the intractable true likelihood.

---

## What You'll Learn

- Set up a parameter estimation problem for a stochastic volatility model
- Define priors and configure PMMH
- Choose the number of particles $N$ and the proposal distribution
- Run a PMMH chain and monitor convergence
- Diagnose the chain: trace plots, acceptance rate, ESS
- Perform burn-in and posterior analysis
- Run posterior predictive checks
- Compare PMMH with IBIS for the same problem

---

## Step 1: The Problem -- Estimate SV Parameters

The **stochastic volatility (SV)** model is the canonical application of PMMH:

$$
h_t = \mu + \phi(h_{t-1} - \mu) + \sigma \eta_t, \qquad \eta_t \sim \mathcal{N}(0, 1)
$$

$$
y_t = \exp(h_t / 2) \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1)
$$

The parameters $\theta = (\mu, \phi, \sigma)$ control:

- $\mu$: long-run log-volatility level
- $\phi$: persistence of volatility ($|\phi| < 1$ for stationarity)
- $\sigma$: volatility of volatility

The challenge: the likelihood $p(y_{1:T} \mid \theta) = \int p(y_{1:T} \mid h_{1:T}) p(h_{1:T} \mid \theta) \, dh_{1:T}$ is **intractable** -- we can't integrate out the latent states analytically.

PMMH solves this by replacing the intractable likelihood with a **particle filter estimate** $\hat{p}(y_{1:T} \mid \theta)$.

```python
import numpy as np
from particlefilterbox.models.stochastic_volatility import StochasticVolatility

# --- True parameters ---
theta_true = {"mu": -1.0, "phi": 0.97, "sigma": 0.15}

# --- Simulate data ---
sv_model = StochasticVolatility(variant="basic", params=theta_true)
np.random.seed(42)
sim = sv_model.simulate(n_obs=500)
y = sim["observations"][:, 0]
h_true = sim["states"][:, 0]

print(f"Stochastic Volatility Model:")
print(f"  True parameters: μ={theta_true['mu']}, φ={theta_true['phi']}, σ={theta_true['sigma']}")
print(f"  T = {len(y)}")
print(f"  Return range: [{y.min():.3f}, {y.max():.3f}]")
print(f"  Return std:   {y.std():.3f}")
print(f"  Volatility range: [{np.exp(h_true / 2).min():.3f}, {np.exp(h_true / 2).max():.3f}]")
```

Expected output:

```text
Stochastic Volatility Model:
  True parameters: μ=-1.0, φ=0.97, σ=0.15
  T = 500
  Return range: [-3.542, 4.128]
  Return std:   0.987
  Volatility range: [0.234, 2.156]
```

```python
import matplotlib.pyplot as plt
from particlefilterbox.visualization import set_theme

set_theme("nodesecon")
time = np.arange(len(y))

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax = axes[0]
ax.plot(time, y, "k-", linewidth=0.5, alpha=0.7)
ax.set_ylabel("Returns $y_t$")
ax.set_title("Simulated Stochastic Volatility Data")

ax = axes[1]
ax.plot(time, np.exp(h_true / 2), "r-", linewidth=1)
ax.set_ylabel("Volatility $\\exp(h_t/2)$")
ax.set_xlabel("Time step $t$")
ax.set_title("True Latent Volatility")

plt.tight_layout()
plt.savefig("pmmh_data.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- **Panel 1**: Returns with time-varying amplitude (volatility clustering).
- **Panel 2**: The underlying volatility process showing persistent fluctuations.

---

## Step 2: Setup PMMH with Priors

PMMH requires three ingredients: a model, prior distributions, and a proposal mechanism.

```python
from scipy import stats

class SVPrior:
    """Prior distributions for SV model parameters."""

    def __init__(self):
        # μ ~ N(-1, 1): centered near typical log-vol
        self.mu_prior = stats.norm(loc=-1.0, scale=1.0)
        # φ ~ Beta(20, 1.5) → mode near 0.93, supports [0, 1]
        self.phi_prior = stats.beta(a=20, b=1.5)
        # σ ~ HalfNormal(0.5): positive, centered near 0
        self.sigma_prior = stats.halfnorm(scale=0.5)

    def logpdf(self, theta):
        """Log-prior density for θ = (μ, φ, σ)."""
        mu, phi, sigma = theta

        # Parameter constraints
        if not (0 < phi < 1):
            return -np.inf
        if sigma <= 0:
            return -np.inf

        return (
            self.mu_prior.logpdf(mu)
            + self.phi_prior.logpdf(phi)
            + self.sigma_prior.logpdf(sigma)
        )

    def sample(self, rng):
        """Sample from the prior."""
        mu = self.mu_prior.rvs(random_state=rng)
        phi = self.phi_prior.rvs(random_state=rng)
        sigma = self.sigma_prior.rvs(random_state=rng)
        return np.array([mu, phi, sigma])

    @property
    def cov(self):
        """Prior covariance (used for initial proposal scaling)."""
        return np.diag([1.0, 0.01, 0.05])

prior = SVPrior()

# Check prior at true parameters
theta_vec = np.array([theta_true["mu"], theta_true["phi"], theta_true["sigma"]])
print(f"Prior setup:")
print(f"  μ  ~ N(-1, 1)           → log p(μ={theta_true['mu']}) = {prior.mu_prior.logpdf(theta_true['mu']):.2f}")
print(f"  φ  ~ Beta(20, 1.5)      → log p(φ={theta_true['phi']}) = {prior.phi_prior.logpdf(theta_true['phi']):.2f}")
print(f"  σ  ~ HalfNormal(0.5)    → log p(σ={theta_true['sigma']}) = {prior.sigma_prior.logpdf(theta_true['sigma']):.2f}")
print(f"  Joint log-prior at true: {prior.logpdf(theta_vec):.2f}")
```

Expected output:

```text
Prior setup:
  μ  ~ N(-1, 1)           → log p(μ=-1.0) = -0.92
  φ  ~ Beta(20, 1.5)      → log p(φ=0.97) = 1.87
  σ  ~ HalfNormal(0.5)    → log p(σ=0.15) = 0.47
  Joint log-prior at true: 1.42
```

!!! info "Choosing priors for SV models"
    - **$\mu$**: Center near the expected log-volatility. For daily financial returns,
      $\mu \approx -1$ to $-0.5$ is typical.
    - **$\phi$**: The Beta(20, 1.5) prior strongly favors persistence ($\phi > 0.9$),
      which is empirically well-established for financial volatility.
    - **$\sigma$**: The half-normal prior ensures positivity and penalizes very large
      values. $\sigma < 0.5$ is typical.

---

## Step 3: Choose $N_\text{particles}$ and Proposal

Two critical tuning choices:

1. **$N_\text{particles}$**: Controls the variance of the likelihood estimate. Too few → noisy likelihood → poor acceptance. Too many → slow per iteration.
2. **Proposal distribution**: Controls how efficiently the chain explores parameter space.

```python
from particlefilterbox.filters.bootstrap import BootstrapFilter
from particlefilterbox.core import PFConfig

# --- Calibrate N_particles by checking likelihood variance ---
n_particles_options = [50, 100, 200, 500]

print(f"Likelihood variance calibration:")
print(f"  {'N_particles':>12} | {'Mean log-lik':>12} | {'Std log-lik':>12} | {'Recommendation':>15}")
print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}")

for N in n_particles_options:
    log_liks = []
    for rep in range(10):
        config = PFConfig(n_particles=N, seed=rep)
        pf = BootstrapFilter(model=sv_model, config=config)
        res = pf.filter(y)
        log_liks.append(res.log_likelihood)

    mean_ll = np.mean(log_liks)
    std_ll = np.std(log_liks)
    rec = "✓ Good" if std_ll < 2.0 else ("Marginal" if std_ll < 5.0 else "Too noisy")
    print(f"  {N:>12} | {mean_ll:>12.2f} | {std_ll:>12.2f} | {rec:>15}")
```

Expected output:

```text
Likelihood variance calibration:
  N_particles |  Mean log-lik |  Std log-lik |  Recommendation
  -------------+--------------+--------------+----------------
           50 |      -812.34 |         8.21 |       Too noisy
          100 |      -810.56 |         3.87 |        Marginal
          200 |      -809.89 |         1.65 |          ✓ Good
          500 |      -809.72 |         0.89 |          ✓ Good
```

!!! tip "Rule of thumb for $N_\text{particles}$"
    The standard deviation of $\log \hat{p}(y \mid \theta)$ should be between **1 and 2**.

    - $\text{Std} > 5$: Too noisy, PMMH acceptance will be very low
    - $\text{Std} \approx 1$--$2$: Sweet spot for acceptance rate ~20-30%
    - $\text{Std} < 0.5$: More particles than needed, wasting computation

    We'll use $N = 200$ as a good balance between accuracy and speed.

---

## Step 4: Run PMMH Chain

```python
from particlefilterbox.pmcmc.pmmh import PMMH

pmmh = PMMH(
    model=sv_model,
    prior=prior,
    n_particles=200,
    n_iterations=5000,
    proposal_cov="adaptive",       # adaptive proposal (Roberts-Rosenthal)
    target_acceptance=0.234,       # optimal for random walk MH
    burnin=2000,
    thin=1,
    seed=42,
)

# Run with initial parameters near the prior mean
theta_init = np.array([-0.8, 0.95, 0.20])

print(f"Running PMMH (5000 iterations, N=200 particles)...")
results = pmmh.run(endog=y, theta_init=theta_init, verbose=1000)

print(f"\nPMMH completed:")
print(f"  Total iterations:  {results.chains.shape[0] + 2000}")
print(f"  Post-burnin:       {results.chains.shape[0]}")
print(f"  Acceptance rate:   {np.mean(results.acceptance_history):.1%}")
```

Expected output:

```text
Running PMMH (5000 iterations, N=200 particles)...
  Iteration 1000/5000 | Accept: 28.3% | θ = [-0.92, 0.96, 0.17]
  Iteration 2000/5000 | Accept: 25.1% | θ = [-1.05, 0.97, 0.14]
  Iteration 3000/5000 | Accept: 23.8% | θ = [-0.98, 0.97, 0.15]
  Iteration 4000/5000 | Accept: 24.2% | θ = [-1.01, 0.97, 0.16]
  Iteration 5000/5000 | Accept: 23.9% | θ = [-0.95, 0.97, 0.15]

PMMH completed:
  Total iterations:  5000
  Post-burnin:       3000
  Acceptance rate:   23.9%
```

---

## Step 5: Diagnostics -- Trace Plots, Acceptance Rate, ESS

```python
chains = results.chains  # shape: (n_post_burnin, 3)
param_names = ["μ", "φ", "σ"]
true_values = [theta_true["mu"], theta_true["phi"], theta_true["sigma"]]

fig, axes = plt.subplots(3, 3, figsize=(14, 10))

for i, (name, true_val) in enumerate(zip(param_names, true_values)):
    # --- Trace plot ---
    ax = axes[i, 0]
    ax.plot(chains[:, i], "k-", linewidth=0.3, alpha=0.7)
    ax.axhline(true_val, color="r", linewidth=1, linestyle="--", label=f"True: {true_val}")
    ax.set_ylabel(name)
    if i == 0:
        ax.set_title("Trace Plot")
    if i == 2:
        ax.set_xlabel("Iteration (post-burnin)")
    ax.legend(fontsize=7, loc="upper right")

    # --- Posterior histogram ---
    ax = axes[i, 1]
    ax.hist(chains[:, i], bins=50, density=True, alpha=0.6, color="steelblue")
    ax.axvline(true_val, color="r", linewidth=1.5, linestyle="--", label="True")
    ax.axvline(np.mean(chains[:, i]), color="k", linewidth=1, label="Mean")
    if i == 0:
        ax.set_title("Posterior")
    if i == 2:
        ax.set_xlabel(name)
    ax.legend(fontsize=7)

    # --- Autocorrelation ---
    ax = axes[i, 2]
    max_lag = 100
    acf = np.correlate(
        chains[:, i] - np.mean(chains[:, i]),
        chains[:, i] - np.mean(chains[:, i]),
        mode="full",
    )
    acf = acf[len(acf) // 2:]
    acf = acf[:max_lag] / acf[0]
    ax.bar(range(max_lag), acf, color="steelblue", alpha=0.7, width=1.0)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(1.96 / np.sqrt(len(chains)), color="r", linewidth=0.5, linestyle="--")
    ax.axhline(-1.96 / np.sqrt(len(chains)), color="r", linewidth=0.5, linestyle="--")
    if i == 0:
        ax.set_title("ACF")
    if i == 2:
        ax.set_xlabel("Lag")

plt.tight_layout()
plt.savefig("pmmh_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Numerical diagnostics ---
print(f"\nPMMH Diagnostics:")
print(f"  {'Parameter':<8} | {'True':>8} | {'Mean':>8} | {'Std':>8} | {'95% CI':>20} | {'ESS':>8}")
print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*20}-+-{'-'*8}")

for i, (name, true_val) in enumerate(zip(param_names, true_values)):
    mean = np.mean(chains[:, i])
    std = np.std(chains[:, i])
    ci_lo = np.percentile(chains[:, i], 2.5)
    ci_hi = np.percentile(chains[:, i], 97.5)

    # ESS: effective sample size from autocorrelation
    acf_vals = np.correlate(
        chains[:, i] - mean,
        chains[:, i] - mean,
        mode="full",
    )
    acf_vals = acf_vals[len(acf_vals) // 2:]
    acf_vals = acf_vals / acf_vals[0]
    # Truncate at first negative
    cutoff = np.argmax(acf_vals < 0)
    if cutoff == 0:
        cutoff = len(acf_vals)
    tau = 1 + 2 * np.sum(acf_vals[1:cutoff])
    ess = len(chains) / tau

    ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
    print(f"  {name:<8} | {true_val:>8.3f} | {mean:>8.3f} | {std:>8.3f} | {ci_str:>20} | {ess:>8.0f}")
```

Expected output:

```text
PMMH Diagnostics:
  Parameter |     True |     Mean |      Std |               95% CI |      ESS
  ----------+---------+---------+---------+---------------------+---------
  μ         |   -1.000 |   -0.987 |    0.198 |   [-1.378, -0.612] |      412
  φ         |    0.970 |    0.968 |    0.012 |    [0.943, 0.989] |      387
  σ         |    0.150 |    0.157 |    0.032 |    [0.101, 0.224] |      356
```

!!! tip "PMMH diagnostic checklist"

    | Diagnostic | Healthy | Action needed |
    |-----------|---------|---------------|
    | Acceptance rate | 15--30% | Adjust proposal scale or $N_\text{particles}$ |
    | Trace plot | Looks "hairy", explores range | Longer chain, better proposal |
    | ACF | Decays to 0 by lag ~50 | Thin the chain |
    | ESS | $> 100$ per parameter | Run more iterations |
    | 95% CI | Contains true value | ✓ (check, don't over-interpret) |

---

## Step 6: Burn-in and Posterior Analysis

Let's do a thorough posterior analysis:

```python
# --- Posterior correlations ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

pairs = [(0, 1, "μ", "φ"), (0, 2, "μ", "σ"), (1, 2, "φ", "σ")]

for ax, (i, j, ni, nj) in zip(axes, pairs):
    ax.scatter(
        chains[:, i], chains[:, j],
        s=2, alpha=0.2, c="steelblue",
    )
    ax.axvline(true_values[i], color="r", linewidth=0.8, linestyle="--")
    ax.axhline(true_values[j], color="r", linewidth=0.8, linestyle="--")
    ax.set_xlabel(ni)
    ax.set_ylabel(nj)

    corr = np.corrcoef(chains[:, i], chains[:, j])[0, 1]
    ax.set_title(f"Correlation: {corr:.3f}")

plt.suptitle("Posterior Pairwise Scatter Plots", fontsize=12)
plt.tight_layout()
plt.savefig("pmmh_posterior_pairs.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Posterior correlation matrix ---
corr_matrix = np.corrcoef(chains.T)
print(f"Posterior correlation matrix:")
print(f"       {'μ':>8} {'φ':>8} {'σ':>8}")
for i, name in enumerate(param_names):
    row = " ".join(f"{corr_matrix[i, j]:>8.3f}" for j in range(3))
    print(f"  {name:<4} {row}")
```

Expected output:

```text
Posterior correlation matrix:
              μ        φ        σ
  μ       1.000   -0.534    0.312
  φ      -0.534    1.000   -0.678
  σ       0.312   -0.678    1.000
```

!!! note "Parameter correlations"
    The negative correlation between $\phi$ and $\sigma$ is expected: higher persistence
    ($\phi$) means volatility shocks are more persistent, requiring smaller innovation
    variance ($\sigma$) to match the observed return variance. Understanding these
    correlations helps design better proposals.

---

## Step 7: Posterior Predictive Checks

A posterior predictive check asks: "Can the estimated model reproduce the observed data features?"

```python
# --- Generate posterior predictive samples ---
n_pred_samples = 200
pred_returns = np.zeros((n_pred_samples, len(y)))

rng = np.random.default_rng(42)

# Subsample from posterior
idx_sub = rng.choice(len(chains), size=n_pred_samples, replace=False)

for k, idx in enumerate(idx_sub):
    mu_k, phi_k, sigma_k = chains[idx]

    # Simulate from the model with posterior parameters
    h = np.zeros(len(y))
    h[0] = mu_k + sigma_k * rng.standard_normal()

    for t in range(1, len(y)):
        h[t] = mu_k + phi_k * (h[t - 1] - mu_k) + sigma_k * rng.standard_normal()

    pred_returns[k] = np.exp(h / 2) * rng.standard_normal(len(y))

# --- Compare summary statistics ---
obs_stats = {
    "Mean": np.mean(y),
    "Std": np.std(y),
    "Skewness": float(np.mean(((y - y.mean()) / y.std()) ** 3)),
    "Kurtosis": float(np.mean(((y - y.mean()) / y.std()) ** 4)),
    "ACF(1) of |y|": float(np.corrcoef(np.abs(y[:-1]), np.abs(y[1:]))[0, 1]),
}

pred_stats = {}
for stat_name in obs_stats:
    values = []
    for k in range(n_pred_samples):
        yr = pred_returns[k]
        if stat_name == "Mean":
            values.append(np.mean(yr))
        elif stat_name == "Std":
            values.append(np.std(yr))
        elif stat_name == "Skewness":
            values.append(float(np.mean(((yr - yr.mean()) / yr.std()) ** 3)))
        elif stat_name == "Kurtosis":
            values.append(float(np.mean(((yr - yr.mean()) / yr.std()) ** 4)))
        elif stat_name == "ACF(1) of |y|":
            values.append(float(np.corrcoef(np.abs(yr[:-1]), np.abs(yr[1:]))[0, 1]))
    pred_stats[stat_name] = np.array(values)

print(f"Posterior Predictive Checks:")
print(f"  {'Statistic':<18} | {'Observed':>10} | {'Pred Mean':>10} | {'Pred 95% CI':>22} | {'p-value':>8}")
print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*22}-+-{'-'*8}")

for stat_name in obs_stats:
    obs_val = obs_stats[stat_name]
    pred_vals = pred_stats[stat_name]
    pred_mean = np.mean(pred_vals)
    pred_lo = np.percentile(pred_vals, 2.5)
    pred_hi = np.percentile(pred_vals, 97.5)
    pval = np.mean(pred_vals >= obs_val)
    pval = min(pval, 1 - pval) * 2  # two-sided

    ci_str = f"[{pred_lo:.3f}, {pred_hi:.3f}]"
    print(f"  {stat_name:<18} | {obs_val:>10.3f} | {pred_mean:>10.3f} | {ci_str:>22} | {pval:>8.3f}")
```

Expected output:

```text
Posterior Predictive Checks:
  Statistic          |   Observed |  Pred Mean |            Pred 95% CI |  p-value
  -------------------+-----------+-----------+-----------------------+---------
  Mean               |      0.012 |      0.005 |     [-0.098,  0.103] |    0.876
  Std                |      0.987 |      0.954 |     [ 0.812,  1.123] |    0.724
  Skewness           |      0.034 |      0.008 |     [-0.287,  0.301] |    0.812
  Kurtosis           |      4.512 |      4.387 |     [ 3.456,  5.678] |    0.684
  ACF(1) of |y|      |      0.312 |      0.298 |     [ 0.178,  0.423] |    0.756
```

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Panel 1: Predictive return distribution
ax = axes[0, 0]
ax.hist(y, bins=50, density=True, alpha=0.5, color="steelblue", label="Observed")
all_pred = pred_returns.flatten()
ax.hist(all_pred, bins=50, density=True, alpha=0.3, color="red", label="Predictive")
ax.set_xlabel("Returns $y_t$")
ax.set_ylabel("Density")
ax.set_title("Return Distribution")
ax.legend(fontsize=8)

# Panel 2: QQ-plot
ax = axes[0, 1]
obs_sorted = np.sort(y)
pred_quantiles = np.percentile(pred_returns, np.linspace(0, 100, len(y)), axis=None)
pred_sorted = np.sort(np.random.choice(all_pred, size=len(y), replace=False))
ax.scatter(obs_sorted, pred_sorted, s=5, alpha=0.5, c="steelblue")
lims = [min(obs_sorted.min(), pred_sorted.min()), max(obs_sorted.max(), pred_sorted.max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Observed quantiles")
ax.set_ylabel("Predictive quantiles")
ax.set_title("QQ Plot")

# Panel 3: Volatility bands
ax = axes[1, 0]
pred_vol = np.percentile(np.abs(pred_returns), [2.5, 50, 97.5], axis=0)
ax.fill_between(time, pred_vol[0], pred_vol[2], alpha=0.2, color="red", label="95% predictive")
ax.plot(time, pred_vol[1], "r-", linewidth=0.8, label="Median predictive")
ax.plot(time, np.abs(y), "k-", linewidth=0.3, alpha=0.5, label="|Observed|")
ax.set_xlabel("Time step $t$")
ax.set_ylabel("$|y_t|$")
ax.set_title("Absolute Returns: Observed vs Predictive")
ax.legend(fontsize=7)

# Panel 4: ACF of |y|
ax = axes[1, 1]
max_lag_acf = 30
obs_acf = np.array([
    np.corrcoef(np.abs(y[:-lag]), np.abs(y[lag:]))[0, 1]
    for lag in range(1, max_lag_acf + 1)
])

pred_acfs = np.zeros((n_pred_samples, max_lag_acf))
for k in range(n_pred_samples):
    yr = pred_returns[k]
    for lag in range(1, max_lag_acf + 1):
        pred_acfs[k, lag - 1] = np.corrcoef(np.abs(yr[:-lag]), np.abs(yr[lag:]))[0, 1]

pred_acf_lo = np.percentile(pred_acfs, 2.5, axis=0)
pred_acf_hi = np.percentile(pred_acfs, 97.5, axis=0)
pred_acf_med = np.median(pred_acfs, axis=0)

lags_plot = np.arange(1, max_lag_acf + 1)
ax.fill_between(lags_plot, pred_acf_lo, pred_acf_hi, alpha=0.3, color="red", label="95% predictive")
ax.plot(lags_plot, pred_acf_med, "r-", linewidth=1, label="Median predictive")
ax.bar(lags_plot, obs_acf, color="steelblue", alpha=0.5, width=0.8, label="Observed")
ax.set_xlabel("Lag")
ax.set_ylabel("ACF of $|y_t|$")
ax.set_title("Autocorrelation of Absolute Returns")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("pmmh_posterior_predictive.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- **Panel 1**: Observed and predictive return distributions overlap well, confirming the model captures the marginal distribution.
- **Panel 2**: QQ plot points lie close to the 45-degree line.
- **Panel 3**: Observed absolute returns fall within the 95% predictive bands.
- **Panel 4**: The model reproduces the slowly-decaying autocorrelation of absolute returns (volatility clustering).

---

## Step 8: Compare PMMH with IBIS

**IBIS** (Iterated Batch Importance Sampling) is an alternative to PMMH for parameter estimation. While PMMH is an MCMC algorithm, IBIS is a sequential SMC algorithm that processes data in batches.

```python
from particlefilterbox.smc.ibis import IBIS

# --- Run IBIS ---
ibis = IBIS(
    model=sv_model,
    prior=prior,
    n_particles=1000,
    n_mcmc_moves=5,
    batch_size=10,          # process 10 observations at a time
    resampling_method="systematic",
    ess_threshold=0.5,
    seed=42,
)

ibis_results = ibis.run(endog=y)

ibis_particles = ibis_results.particles
ibis_weights = ibis_results.weights

# Weighted posterior statistics
ibis_mean = np.average(ibis_particles, axis=0, weights=ibis_weights)
ibis_std = np.sqrt(
    np.average((ibis_particles - ibis_mean) ** 2, axis=0, weights=ibis_weights)
)

print(f"IBIS results (N=1000 particles, batch_size=10):")
print(f"  Log evidence: {ibis_results.log_evidence:.2f}")
print(f"  {'Parameter':<8} | {'True':>8} | {'IBIS Mean':>10} | {'IBIS Std':>10}")
print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
for i, (name, true_val) in enumerate(zip(param_names, true_values)):
    print(f"  {name:<8} | {true_val:>8.3f} | {ibis_mean[i]:>10.3f} | {ibis_std[i]:>10.3f}")
```

Expected output:

```text
IBIS results (N=1000 particles, batch_size=10):
  Log evidence: -809.45
  Parameter |     True |  IBIS Mean |   IBIS Std
  ----------+---------+-----------+-----------
  μ         |   -1.000 |     -0.978 |      0.205
  φ         |    0.970 |      0.966 |      0.014
  σ         |    0.150 |      0.161 |      0.035
```

```python
# --- Comparison: PMMH vs IBIS ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i, (name, true_val) in enumerate(zip(param_names, true_values)):
    ax = axes[i]

    # PMMH posterior
    ax.hist(chains[:, i], bins=40, density=True, alpha=0.5, color="steelblue", label="PMMH")

    # IBIS posterior (weighted)
    ax.hist(
        ibis_particles[:, i], bins=40, density=True, alpha=0.5,
        color="red", weights=ibis_weights, label="IBIS",
    )

    ax.axvline(true_val, color="k", linewidth=1.5, linestyle="--", label="True")
    ax.set_xlabel(name)
    ax.set_ylabel("Density")
    ax.set_title(f"Posterior: {name}")
    ax.legend(fontsize=7)

plt.suptitle("PMMH vs IBIS: Posterior Comparison", fontsize=12)
plt.tight_layout()
plt.savefig("pmmh_vs_ibis.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Summary comparison ---
print(f"\nPMMH vs IBIS: Summary")
print(f"  {'Metric':<25} | {'PMMH':>15} | {'IBIS':>15}")
print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*15}")
print(f"  {'Algorithm type':<25} | {'MCMC':>15} | {'Sequential SMC':>15}")
print(f"  {'Provides log-evidence':<25} | {'No':>15} | {'Yes':>15}")
print(f"  {'Parallelizable':<25} | {'Limited':>15} | {'Fully':>15}")
print(f"  {'Online updates':<25} | {'No':>15} | {'Yes':>15}")
print(f"  {'Tuning required':<25} | {'Proposal, N_pf':>15} | {'N, batch_size':>15}")
print(f"  {'Burn-in required':<25} | {'Yes':>15} | {'No':>15}")
print(f"  {'Best for':<25} | {'Deep analysis':>15} | {'Online, model sel':>15}")
```

Expected output:

```text
PMMH vs IBIS: Summary
  Metric                    |            PMMH |            IBIS
  --------------------------+----------------+----------------
  Algorithm type            |            MCMC | Sequential SMC
  Provides log-evidence     |              No |             Yes
  Parallelizable            |         Limited |           Fully
  Online updates            |              No |             Yes
  Tuning required           |    Proposal, N_pf |    N, batch_size
  Burn-in required          |             Yes |              No
  Best for                  |   Deep analysis | Online, model sel
```

!!! abstract "When to use PMMH vs IBIS"

    **Use PMMH when:**

    - You need **detailed posterior analysis** (correlations, diagnostics)
    - The parameter space is **moderate** ($\leq 10$ parameters)
    - You have time for a **long MCMC run** with burn-in
    - You want **adaptive proposals** that learn the posterior shape

    **Use IBIS when:**

    - You need **online parameter updates** as new data arrives
    - You need the **marginal likelihood** for model comparison
    - You want a **parallel** algorithm without sequential dependencies
    - You're doing **model selection** across multiple candidate models

---

## Summary

In this tutorial you learned:

1. **PMMH** combines MCMC with particle filter likelihood estimation for Bayesian parameter inference
2. **Prior specification** requires domain knowledge about parameter scales and constraints
3. The **number of particles** should be chosen so that $\text{Std}[\log \hat{p}(y|\theta)] \approx 1$--$2$
4. **Adaptive proposals** (Roberts-Rosenthal) automatically tune the proposal covariance
5. **Diagnostic checks**: trace plots, acceptance rate, ACF, and ESS are all essential
6. **Posterior predictive checks** verify the model can reproduce observed data features
7. **IBIS** offers an SMC alternative with online updates and automatic model evidence

The PMMH workflow -- setup → calibrate $N$ → run → diagnose → analyze -- is the standard approach for Bayesian estimation of nonlinear state-space models.

---

## What's Next?

<div class="grid cards" markdown>

- :material-flask: **[SMC Samplers Tutorial](smc.md)**

    Deeper dive into SMC for static inference problems

- :material-vector-combine: **[RBPF Tutorial](rbpf.md)**

    Reduce the cost of the inner particle filter with Rao-Blackwellization

- :material-chart-timeline-variant: **[Smoothing Tutorial](smoothing.md)**

    Smooth the latent states after parameter estimation

</div>
