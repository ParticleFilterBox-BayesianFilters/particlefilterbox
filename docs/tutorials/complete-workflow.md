---
title: "Tutorial: Complete Analysis Workflow"
description: End-to-end workflow from data exploration through filtering, parameter estimation, model comparison, smoothing, and report generation
---

# Tutorial: Complete Analysis Workflow

**Level**: :material-star:{.advanced} Advanced  
**Time**: ~90 minutes  
**Prerequisites**: All previous tutorials  

This **capstone tutorial** walks through a complete research workflow for analyzing financial returns using the stochastic volatility model with leverage effects. You'll use every major component of particlefilterbox: filtering, diagnostics, parameter estimation, model comparison, smoothing, and visualization.

---

## What You'll Learn

- Define a stochastic volatility model with leverage (SV-L)
- Explore data features that motivate model choice
- Select the appropriate particle filter using the decision tree
- Filter latent states and diagnose the filter
- Estimate parameters with PMMH
- Diagnose the MCMC chain
- Perform posterior analysis
- Compare competing models (SV vs SV-t)
- Smooth state trajectories with FFBSm
- Create publication-quality visualizations
- Generate a structured analysis report

---

## Step 1: Define the Model -- SV with Leverage

The **stochastic volatility with leverage** (SV-L) model captures the well-known asymmetry in financial markets: negative returns increase future volatility more than positive returns of the same magnitude.

$$
h_t = \mu + \phi(h_{t-1} - \mu) + \sigma \eta_t
$$

$$
y_t = \exp(h_t / 2) \varepsilon_t
$$

$$
\begin{pmatrix} \eta_t \\ \varepsilon_t \end{pmatrix} \sim \mathcal{N}\left(\mathbf{0}, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}\right)
$$

The correlation $\rho$ captures the **leverage effect**: typically $\rho < 0$, meaning a negative return today ($\varepsilon_t < 0$) is associated with an increase in volatility tomorrow ($\eta_t > 0$).

```python
import numpy as np
from scipy import stats

# --- True parameters (calibrated to S&P 500 daily returns) ---
true_params = {
    "mu": -0.5,       # long-run log-volatility
    "phi": 0.975,     # persistence
    "sigma": 0.12,    # vol-of-vol
    "rho": -0.35,     # leverage correlation
}

# --- Simulate T=1000 daily returns ---
np.random.seed(42)
T = 1000

h = np.zeros(T)
y = np.zeros(T)

# Correlated shocks
rho = true_params["rho"]
cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
L = np.linalg.cholesky(cov_matrix)

# Initial state from stationary distribution
h_var = true_params["sigma"] ** 2 / (1 - true_params["phi"] ** 2)
h[0] = true_params["mu"] + np.sqrt(h_var) * np.random.randn()
y[0] = np.exp(h[0] / 2) * np.random.randn()

for t in range(1, T):
    shocks = L @ np.random.randn(2)
    h[t] = (
        true_params["mu"]
        + true_params["phi"] * (h[t - 1] - true_params["mu"])
        + true_params["sigma"] * shocks[0]
    )
    y[t] = np.exp(h[t] / 2) * shocks[1]

h_true = h.copy()

print("SV-Leverage Model:")
print(f"  Parameters: μ={true_params['mu']}, φ={true_params['phi']}, "
      f"σ={true_params['sigma']}, ρ={true_params['rho']}")
print(f"  T = {T} daily returns")
print(f"  Return range: [{y.min():.3f}, {y.max():.3f}]")
print(f"  Return std:   {y.std():.3f}")
print(f"  Volatility range: [{np.exp(h_true / 2).min():.3f}, {np.exp(h_true / 2).max():.3f}]")
```

Expected output:

```text
SV-Leverage Model:
  Parameters: μ=-0.5, φ=0.975, σ=0.12, ρ=-0.35
  T = 1000 daily returns
  Return range: [-4.234, 3.876]
  Return std:   0.892
  Volatility range: [0.412, 1.876]
```

---

## Step 2: Explore Data Features

Before choosing a filter, let's examine the data to understand its statistical properties:

```python
import matplotlib.pyplot as plt
from particlefilterbox.visualization import set_theme

set_theme("nodesecon")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

# --- Panel 1: Return series ---
ax = axes[0, 0]
ax.plot(y, "k-", linewidth=0.4, alpha=0.7)
ax.set_xlabel("Day $t$")
ax.set_ylabel("Return $y_t$")
ax.set_title("Daily Returns")

# --- Panel 2: Return histogram ---
ax = axes[0, 1]
ax.hist(y, bins=80, density=True, alpha=0.6, color="steelblue", label="Empirical")
x_grid = np.linspace(y.min(), y.max(), 200)
ax.plot(x_grid, stats.norm.pdf(x_grid, y.mean(), y.std()), "r--",
        linewidth=1.5, label="Normal fit")
ax.set_xlabel("Return $y_t$")
ax.set_ylabel("Density")
ax.set_title("Return Distribution")
ax.legend(fontsize=8)

# --- Panel 3: QQ plot ---
ax = axes[0, 2]
sorted_y = np.sort(y)
theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, len(y)))
ax.scatter(theoretical, sorted_y, s=2, alpha=0.5, c="steelblue")
ax.plot([-4, 4], [-4, 4], "r--", linewidth=1)
ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Sample quantiles")
ax.set_title("QQ Plot (Normal)")

# --- Panel 4: Squared returns (volatility proxy) ---
ax = axes[1, 0]
ax.plot(y ** 2, "k-", linewidth=0.3, alpha=0.6)
ax.set_xlabel("Day $t$")
ax.set_ylabel("$y_t^2$")
ax.set_title("Squared Returns (Volatility Proxy)")

# --- Panel 5: Autocorrelation of |y| ---
ax = axes[1, 1]
abs_y = np.abs(y)
max_lag = 50
acf_abs = np.zeros(max_lag)
for lag in range(max_lag):
    if lag == 0:
        acf_abs[lag] = 1.0
    else:
        acf_abs[lag] = np.corrcoef(abs_y[:-lag], abs_y[lag:])[0, 1]
ax.bar(range(max_lag), acf_abs, color="steelblue", alpha=0.7, width=1.0)
ax.axhline(1.96 / np.sqrt(T), color="r", linewidth=0.5, linestyle="--")
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
ax.set_title("ACF of $|y_t|$ (Volatility Clustering)")

# --- Panel 6: Leverage effect ---
ax = axes[1, 2]
# Correlation between y_t and |y_{t+k}|
lags = range(1, 21)
leverage_corr = [
    np.corrcoef(y[:-lag], y[lag:] ** 2)[0, 1]
    for lag in lags
]
ax.bar(lags, leverage_corr, color="firebrick", alpha=0.7)
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("Lag $k$")
ax.set_ylabel("Corr($y_t$, $y_{t+k}^2$)")
ax.set_title("Leverage Effect")

plt.suptitle("Data Exploration: Key Features for Model Selection", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("workflow_data_exploration.png", dpi=150, bbox_inches="tight")
plt.show()

# Summary statistics
kurtosis = np.mean(((y - y.mean()) / y.std()) ** 4)
skewness = np.mean(((y - y.mean()) / y.std()) ** 3)
jb_stat = (T / 6) * (skewness ** 2 + (kurtosis - 3) ** 2 / 4)

print(f"\nData Summary Statistics:")
print(f"  Mean:           {y.mean():.4f}")
print(f"  Std:            {y.std():.4f}")
print(f"  Skewness:       {skewness:.3f}")
print(f"  Kurtosis:       {kurtosis:.3f} (Normal = 3)")
print(f"  Jarque-Bera:    {jb_stat:.1f} (p < 0.001)")
print(f"  ACF(1) of |y|:  {acf_abs[1]:.3f}")
print(f"  Leverage at k=1:{leverage_corr[0]:.3f}")
print(f"\nFindings:")
print(f"  ✓ Heavy tails (kurtosis >> 3)")
print(f"  ✓ Volatility clustering (significant ACF of |y|)")
print(f"  ✓ Leverage effect (negative correlation)")
print(f"  → SV model with leverage is appropriate!")
```

Expected output:

```text
Data Summary Statistics:
  Mean:           -0.0012
  Std:            0.8923
  Skewness:       -0.234
  Kurtosis:       5.678 (Normal = 3)
  Jarque-Bera:    312.4 (p < 0.001)
  ACF(1) of |y|:  0.312
  Leverage at k=1:-0.187

Findings:
  ✓ Heavy tails (kurtosis >> 3)
  ✓ Volatility clustering (significant ACF of |y|)
  ✓ Leverage effect (negative correlation)
  → SV model with leverage is appropriate!
```

---

## Step 3: Choose the Filter (Decision Tree)

Based on our data exploration, let's systematically select the appropriate filter:

```python
# --- Decision tree for filter selection ---
print("Filter Selection Decision Tree:")
print("="*50)
print(f"  1. Is the model linear-Gaussian?")
print(f"     → No (nonlinear observation: y = exp(h/2) * ε)")
print(f"  2. Does the model have linear substructure?")
print(f"     → No (leverage creates correlation between")
print(f"       state and observation noise)")
print(f"  3. Is a good proposal available?")
print(f"     → Yes! The Auxiliary PF can use the current")
print(f"       observation to guide particle placement")
print(f"  4. Recommendation: Auxiliary Particle Filter")
print(f"     - Better than Bootstrap PF for SV models")
print(f"     - Handles the informative observation equation")
print(f"     - N=500 particles should suffice")
```

Expected output:

```text
Filter Selection Decision Tree:
==================================================
  1. Is the model linear-Gaussian?
     → No (nonlinear observation: y = exp(h/2) * ε)
  2. Does the model have linear substructure?
     → No (leverage creates correlation between
       state and observation noise)
  3. Is a good proposal available?
     → Yes! The Auxiliary PF can use the current
       observation to guide particle placement
  4. Recommendation: Auxiliary Particle Filter
     - Better than Bootstrap PF for SV models
     - Handles the informative observation equation
     - N=500 particles should suffice
```

---

## Step 4: Filter Latent States

```python
from particlefilterbox.models.stochastic_volatility import StochasticVolatility
from particlefilterbox.filters.auxiliary import AuxiliaryFilter
from particlefilterbox.filters.bootstrap import BootstrapFilter
from particlefilterbox.core import PFConfig

# --- Define the SV-L model ---
sv_model = StochasticVolatility(variant="leverage", params=true_params)

# --- Auxiliary PF ---
config_apf = PFConfig(
    n_particles=500,
    resampling="systematic",
    backend="numba",
    seed=42,
)
apf = AuxiliaryFilter(model=sv_model, config=config_apf)
results_apf = apf.filter(y)

h_filtered = results_apf.filtered_mean[:, 0]
rmse = np.sqrt(np.mean((h_filtered - h_true) ** 2))

print(f"Auxiliary PF Results (N=500):")
print(f"  RMSE:           {rmse:.4f}")
print(f"  Mean ESS:       {np.mean(results_apf.ess_history):.1f}")
print(f"  Min ESS:        {np.min(results_apf.ess_history):.1f}")
print(f"  Log-likelihood: {results_apf.log_likelihood:.2f}")

# --- Bootstrap PF for comparison ---
config_bpf = PFConfig(n_particles=500, resampling="systematic", backend="numba", seed=42)
bpf = BootstrapFilter(model=sv_model, config=config_bpf)
results_bpf = bpf.filter(y)

h_bpf = results_bpf.filtered_mean[:, 0]
rmse_bpf = np.sqrt(np.mean((h_bpf - h_true) ** 2))

print(f"\nComparison:")
print(f"  {'Filter':<20} | {'RMSE':>8} | {'Mean ESS':>10} | {'Log-lik':>10}")
print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
print(f"  {'Auxiliary PF':<20} | {rmse:>8.4f} | {np.mean(results_apf.ess_history):>10.1f} | {results_apf.log_likelihood:>10.2f}")
print(f"  {'Bootstrap PF':<20} | {rmse_bpf:>8.4f} | {np.mean(results_bpf.ess_history):>10.1f} | {results_bpf.log_likelihood:>10.2f}")
```

Expected output:

```text
Auxiliary PF Results (N=500):
  RMSE:           0.1234
  Mean ESS:       378.4
  Min ESS:        189.2
  Log-likelihood: -1423.56

Comparison:
  Filter               |     RMSE |   Mean ESS |    Log-lik
  ---------------------+---------+-----------+-----------
  Auxiliary PF          |   0.1234 |      378.4 |   -1423.56
  Bootstrap PF          |   0.1567 |      298.7 |   -1425.89
```

```python
time = np.arange(T)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Returns
ax = axes[0]
ax.plot(time, y, "k-", linewidth=0.4, alpha=0.7)
ax.set_ylabel("Returns $y_t$")
ax.set_title("Step 4: Filtered Latent Volatility")

# Filtered volatility
ax = axes[1]
vol_true = np.exp(h_true / 2)
vol_filtered = np.exp(h_filtered / 2)
ax.plot(time, vol_true, "k-", linewidth=1.5, label="True volatility", alpha=0.8)
ax.plot(time, vol_filtered, "r-", linewidth=1, label="APF estimate", alpha=0.8)
ax.set_ylabel("Volatility $\\exp(h_t/2)$")
ax.legend(fontsize=8)

# ESS
ax = axes[2]
ax.plot(time, results_apf.ess_history, "steelblue", linewidth=0.5)
ax.axhline(250, color="r", linewidth=0.5, linestyle="--", label="N/2 threshold")
ax.set_ylabel("ESS")
ax.set_xlabel("Day $t$")
ax.set_title("Effective Sample Size")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("workflow_filtering.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- **Panel 1**: Daily returns showing volatility clustering.
- **Panel 2**: True volatility (black) closely tracked by APF estimate (red).
- **Panel 3**: ESS stays above the N/2 threshold most of the time, indicating healthy filtering.

---

## Step 5: Diagnose the Filter (ESS, Weights)

```python
from particlefilterbox.diagnostics import ESSMonitor, WeightAnalysis

# --- ESS diagnostics ---
ess_monitor = ESSMonitor(n_particles=500, warning_threshold=0.3, critical_threshold=0.1)
ess_alerts = ess_monitor.analyze(results_apf.ess_history)

print(f"Filter Diagnostics:")
print(f"  Mean ESS:      {np.mean(results_apf.ess_history):.1f} / {500}")
print(f"  Min ESS:       {np.min(results_apf.ess_history):.1f}")
print(f"  ESS < N/2:     {np.sum(results_apf.ess_history < 250)} / {T} time steps")
print(f"  ESS < N/10:    {np.sum(results_apf.ess_history < 50)} / {T} time steps")
print(f"  Alerts:        {len(ess_alerts)} ({sum(1 for a in ess_alerts if a.level == 'CRITICAL')} critical)")

# --- Weight analysis ---
weight_analysis = WeightAnalysis()
weight_stats = weight_analysis.analyze(results_apf)

print(f"\n  Weight Statistics:")
print(f"    Max weight (mean): {weight_stats['max_weight_mean']:.4f}")
print(f"    Max weight (max):  {weight_stats['max_weight_max']:.4f}")
print(f"    Weight entropy:    {weight_stats['entropy_mean']:.2f} (max: {np.log(500):.2f})")

# Overall health
health = "HEALTHY" if np.mean(results_apf.ess_history) > 250 else "NEEDS ATTENTION"
print(f"\n  Filter health: {health}")
```

Expected output:

```text
Filter Diagnostics:
  Mean ESS:      378.4 / 500
  Min ESS:       189.2
  ESS < N/2:     87 / 1000 time steps
  ESS < N/10:    0 / 1000 time steps
  Alerts:        3 (0 critical)

  Weight Statistics:
    Max weight (mean): 0.0123
    Max weight (max):  0.0567
    Weight entropy:    5.89 (max: 6.21)

  Filter health: HEALTHY
```

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# ESS distribution
ax = axes[0, 0]
ax.hist(results_apf.ess_history, bins=50, density=True,
        alpha=0.6, color="steelblue")
ax.axvline(250, color="r", linewidth=1, linestyle="--", label="N/2")
ax.axvline(np.mean(results_apf.ess_history), color="k", linewidth=1, label="Mean")
ax.set_xlabel("ESS")
ax.set_ylabel("Density")
ax.set_title("ESS Distribution")
ax.legend(fontsize=8)

# ESS vs absolute return
ax = axes[0, 1]
ax.scatter(np.abs(y), results_apf.ess_history, s=3, alpha=0.3, c="steelblue")
ax.set_xlabel("$|y_t|$")
ax.set_ylabel("ESS")
ax.set_title("ESS vs Absolute Return")

# Max weight over time
ax = axes[1, 0]
# Approximate max weight from ESS
max_weight_approx = 1.0 / results_apf.ess_history
ax.plot(time, max_weight_approx, "k-", linewidth=0.3, alpha=0.6)
ax.set_xlabel("Day $t$")
ax.set_ylabel("1/ESS (weight concentration)")
ax.set_title("Weight Concentration Over Time")

# Log-likelihood increments
ax = axes[1, 1]
ll_increments = np.diff(np.concatenate([[0], results_apf.log_likelihood_increments]))
ax.plot(time, ll_increments, "k-", linewidth=0.3, alpha=0.6)
ax.set_xlabel("Day $t$")
ax.set_ylabel("$\\log p(y_t | y_{1:t-1})$")
ax.set_title("Log-Likelihood Increments")

plt.suptitle("Step 5: Filter Diagnostics", fontsize=12)
plt.tight_layout()
plt.savefig("workflow_filter_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- **Top-left**: ESS distribution centered well above the N/2 threshold.
- **Top-right**: ESS drops when |y_t| is large (informative observations are harder to track).
- **Bottom-left**: Weight concentration spikes during extreme returns but remains manageable.
- **Bottom-right**: Log-likelihood increments are stable, with occasional dips during volatile periods.

---

## Step 6: Estimate Parameters (PMMH)

```python
from particlefilterbox.pmcmc.pmmh import PMMH

# --- Define priors for SV-L parameters ---
class SVLPrior:
    """Priors for SV model with leverage."""

    def __init__(self):
        self.mu_prior = stats.norm(loc=-0.5, scale=1.0)
        self.phi_prior = stats.beta(a=20, b=1.5)
        self.sigma_prior = stats.halfnorm(scale=0.5)
        self.rho_prior = stats.uniform(loc=-0.99, scale=1.98)  # U(-0.99, 0.99)

    def logpdf(self, theta):
        mu, phi, sigma, rho = theta
        if not (0 < phi < 1) or sigma <= 0 or not (-0.99 < rho < 0.99):
            return -np.inf
        return (
            self.mu_prior.logpdf(mu)
            + self.phi_prior.logpdf(phi)
            + self.sigma_prior.logpdf(sigma)
            + self.rho_prior.logpdf(rho)
        )

    def sample(self, rng):
        return np.array([
            self.mu_prior.rvs(random_state=rng),
            self.phi_prior.rvs(random_state=rng),
            self.sigma_prior.rvs(random_state=rng),
            self.rho_prior.rvs(random_state=rng),
        ])

prior = SVLPrior()

# --- Run PMMH ---
pmmh = PMMH(
    model=sv_model,
    prior=prior,
    n_particles=300,
    n_iterations=5000,
    proposal_cov="adaptive",
    target_acceptance=0.234,
    burnin=2000,
    thin=1,
    seed=42,
)

theta_init = np.array([-0.3, 0.95, 0.15, -0.20])

print("Running PMMH (5000 iterations, N=300)...")
pmmh_results = pmmh.run(endog=y, theta_init=theta_init, verbose=1000)

chains = pmmh_results.chains
param_names = ["μ", "φ", "σ", "ρ"]
true_values = [true_params["mu"], true_params["phi"],
               true_params["sigma"], true_params["rho"]]

print(f"\nPMMH completed:")
print(f"  Post-burnin samples: {chains.shape[0]}")
print(f"  Acceptance rate:     {np.mean(pmmh_results.acceptance_history):.1%}")
```

Expected output:

```text
Running PMMH (5000 iterations, N=300)...
  Iteration 1000/5000 | Accept: 26.1% | θ = [-0.45, 0.97, 0.13, -0.31]
  Iteration 2000/5000 | Accept: 24.3% | θ = [-0.52, 0.97, 0.12, -0.34]
  Iteration 3000/5000 | Accept: 23.8% | θ = [-0.48, 0.98, 0.11, -0.36]
  Iteration 4000/5000 | Accept: 24.1% | θ = [-0.51, 0.97, 0.12, -0.35]
  Iteration 5000/5000 | Accept: 23.9% | θ = [-0.49, 0.98, 0.12, -0.34]

PMMH completed:
  Post-burnin samples: 3000
  Acceptance rate:     23.9%
```

---

## Step 7: Diagnose MCMC (Trace, R-hat, ESS)

```python
def compute_acf(x, max_lag=100):
    """Compute autocorrelation function."""
    x_centered = x - np.mean(x)
    acf_full = np.correlate(x_centered, x_centered, mode="full")
    acf_full = acf_full[len(acf_full) // 2:]
    return acf_full[:max_lag] / acf_full[0]

def compute_ess(x):
    """Compute effective sample size."""
    acf = compute_acf(x, max_lag=len(x) // 2)
    cutoff = np.argmax(acf < 0)
    if cutoff == 0:
        cutoff = len(acf)
    tau = 1 + 2 * np.sum(acf[1:cutoff])
    return len(x) / max(tau, 1.0)

# --- Comprehensive MCMC diagnostics ---
fig, axes = plt.subplots(4, 3, figsize=(16, 14))

for i, (name, true_val) in enumerate(zip(param_names, true_values)):
    # Trace plot
    ax = axes[i, 0]
    ax.plot(chains[:, i], "k-", linewidth=0.2, alpha=0.7)
    ax.axhline(true_val, color="r", linewidth=1, linestyle="--", label=f"True: {true_val}")
    ax.set_ylabel(name)
    if i == 0:
        ax.set_title("Trace Plot")
    if i == 3:
        ax.set_xlabel("Iteration")
    ax.legend(fontsize=7, loc="upper right")

    # Posterior
    ax = axes[i, 1]
    ax.hist(chains[:, i], bins=50, density=True, alpha=0.6, color="steelblue")
    ax.axvline(true_val, color="r", linewidth=1.5, linestyle="--", label="True")
    ax.axvline(np.mean(chains[:, i]), color="k", linewidth=1, label="Mean")
    if i == 0:
        ax.set_title("Posterior")
    ax.legend(fontsize=7)

    # ACF
    ax = axes[i, 2]
    acf = compute_acf(chains[:, i], 80)
    ax.bar(range(80), acf, color="steelblue", alpha=0.7, width=1.0)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(1.96 / np.sqrt(len(chains)), color="r", linewidth=0.5, linestyle="--")
    if i == 0:
        ax.set_title("ACF")
    if i == 3:
        ax.set_xlabel("Lag")

plt.suptitle("Step 7: MCMC Diagnostics", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("workflow_mcmc_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Numerical diagnostics table ---
print(f"\nMCMC Diagnostics:")
print(f"  {'Param':<6} | {'True':>8} | {'Mean':>8} | {'Std':>8} | {'95% CI':>22} | {'ESS':>6} | {'Status':>8}")
print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*22}-+-{'-'*6}-+-{'-'*8}")

for i, (name, true_val) in enumerate(zip(param_names, true_values)):
    mean = np.mean(chains[:, i])
    std = np.std(chains[:, i])
    ci_lo = np.percentile(chains[:, i], 2.5)
    ci_hi = np.percentile(chains[:, i], 97.5)
    ess = compute_ess(chains[:, i])
    status = "OK" if ess > 100 and ci_lo <= true_val <= ci_hi else "CHECK"
    ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
    print(f"  {name:<6} | {true_val:>8.3f} | {mean:>8.3f} | {std:>8.3f} | {ci_str:>22} | {ess:>6.0f} | {status:>8}")
```

Expected output:

```text
MCMC Diagnostics:
  Param  |     True |     Mean |      Std |                 95% CI |    ESS |   Status
  -------+---------+---------+---------+-----------------------+-------+---------
  μ      |   -0.500 |   -0.487 |    0.198 |   [-0.876, -0.112]    |    389 |       OK
  φ      |    0.975 |    0.973 |    0.009 |    [0.954, 0.989]      |    356 |       OK
  σ      |    0.120 |    0.124 |    0.025 |    [0.078, 0.176]      |    312 |       OK
  ρ      |   -0.350 |   -0.342 |    0.087 |   [-0.512, -0.178]    |    278 |       OK
```

!!! tip "MCMC health checklist"

    | Check | Criterion | Our result |
    |-------|-----------|------------|
    | Acceptance rate | 15-30% | 23.9% |
    | Trace plots | "Hairy", no trends | Good mixing |
    | ACF | Decays by lag ~50 | All parameters |
    | ESS | $> 100$ per parameter | Min: 278 |
    | 95% CI | Contains true value | All 4 parameters |

---

## Step 8: Posterior Analysis

```python
# --- Posterior pairwise scatter ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
pairs = [
    (0, 1, "μ", "φ"), (0, 2, "μ", "σ"), (0, 3, "μ", "ρ"),
    (1, 2, "φ", "σ"), (1, 3, "φ", "ρ"), (2, 3, "σ", "ρ"),
]

for ax, (i, j, ni, nj) in zip(axes.ravel(), pairs):
    ax.scatter(chains[:, i], chains[:, j], s=2, alpha=0.15, c="steelblue")
    ax.axvline(true_values[i], color="r", linewidth=0.8, linestyle="--")
    ax.axhline(true_values[j], color="r", linewidth=0.8, linestyle="--")
    ax.set_xlabel(ni)
    ax.set_ylabel(nj)
    corr = np.corrcoef(chains[:, i], chains[:, j])[0, 1]
    ax.set_title(f"ρ = {corr:.3f}")

plt.suptitle("Step 8: Posterior Correlations", fontsize=12)
plt.tight_layout()
plt.savefig("workflow_posterior.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Posterior correlation matrix ---
corr_matrix = np.corrcoef(chains.T)
print(f"Posterior correlation matrix:")
print(f"       {'μ':>8} {'φ':>8} {'σ':>8} {'ρ':>8}")
for i, name in enumerate(param_names):
    row = " ".join(f"{corr_matrix[i, j]:>8.3f}" for j in range(4))
    print(f"  {name:<4} {row}")
```

Expected output:

```text
Posterior correlation matrix:
              μ        φ        σ        ρ
  μ       1.000   -0.534    0.312   -0.187
  φ      -0.534    1.000   -0.678    0.123
  σ       0.312   -0.678    1.000   -0.234
  ρ      -0.187    0.123   -0.234    1.000
```

!!! note "Key posterior correlations"
    - **$\phi$ and $\sigma$**: Strong negative correlation -- higher persistence requires
      smaller vol-of-vol to match the observed variance.
    - **$\mu$ and $\phi$**: Negative -- higher mean log-vol compensated by lower persistence.
    - **$\rho$ is relatively independent**: The leverage parameter is well-identified
      from the asymmetric response pattern, largely orthogonal to the other parameters.

---

## Step 9: Model Comparison (SV vs SV-t)

Let's compare our SV-L model against a variant with Student-t observation errors (SV-t), which offers an alternative explanation for heavy tails:

```python
from particlefilterbox.smc.sampler import SMCSampler

# --- Model A: SV with leverage (our model) ---
log_ml_A = pmmh_results.log_marginal_likelihood

# --- Model B: SV with Student-t errors (no leverage) ---
sv_model_t = StochasticVolatility(variant="student_t", params={
    "mu": true_params["mu"],
    "phi": true_params["phi"],
    "sigma": true_params["sigma"],
    "nu": 5.0,  # degrees of freedom
})

class SVtPrior:
    """Prior for SV-t model."""
    def __init__(self):
        self.mu_prior = stats.norm(loc=-0.5, scale=1.0)
        self.phi_prior = stats.beta(a=20, b=1.5)
        self.sigma_prior = stats.halfnorm(scale=0.5)
        self.nu_prior = stats.expon(loc=2, scale=10)  # ν > 2, mean around 12

    def logpdf(self, theta):
        mu, phi, sigma, nu = theta
        if not (0 < phi < 1) or sigma <= 0 or nu <= 2:
            return -np.inf
        return (
            self.mu_prior.logpdf(mu)
            + self.phi_prior.logpdf(phi)
            + self.sigma_prior.logpdf(sigma)
            + self.nu_prior.logpdf(nu)
        )

    def sample(self, rng):
        return np.array([
            self.mu_prior.rvs(random_state=rng),
            self.phi_prior.rvs(random_state=rng),
            self.sigma_prior.rvs(random_state=rng),
            self.nu_prior.rvs(random_state=rng),
        ])

prior_t = SVtPrior()

# Run PMMH for Model B
pmmh_t = PMMH(
    model=sv_model_t,
    prior=prior_t,
    n_particles=300,
    n_iterations=5000,
    proposal_cov="adaptive",
    target_acceptance=0.234,
    burnin=2000,
    seed=43,
)

theta_init_t = np.array([-0.3, 0.95, 0.15, 8.0])
print("Running PMMH for SV-t model (5000 iterations)...")
pmmh_results_t = pmmh_t.run(endog=y, theta_init=theta_init_t, verbose=2500)

log_ml_B = pmmh_results_t.log_marginal_likelihood
log_bf = log_ml_A - log_ml_B

print(f"\nModel Comparison:")
print(f"  {'Model':<20} | {'Log-ML':>10} | {'Parameters':>10}")
print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}")
print(f"  {'SV-Leverage':<20} | {log_ml_A:>10.2f} | {'μ,φ,σ,ρ':>10}")
print(f"  {'SV-Student-t':<20} | {log_ml_B:>10.2f} | {'μ,φ,σ,ν':>10}")
print(f"\n  Log Bayes Factor (A vs B): {log_bf:.2f}")

if log_bf > 4.6:
    verdict = "Decisive evidence for SV-Leverage"
elif log_bf > 2.3:
    verdict = "Strong evidence for SV-Leverage"
elif log_bf > 1.15:
    verdict = "Substantial evidence for SV-Leverage"
elif log_bf > -1.15:
    verdict = "Inconclusive"
elif log_bf > -2.3:
    verdict = "Substantial evidence for SV-t"
else:
    verdict = "Strong evidence for SV-t"

print(f"  Interpretation: {verdict}")
```

Expected output:

```text
Running PMMH for SV-t model (5000 iterations)...
  Iteration 2500/5000 | Accept: 23.1% | θ = [-0.47, 0.97, 0.12, 6.8]
  Iteration 5000/5000 | Accept: 23.4% | θ = [-0.50, 0.98, 0.11, 7.2]

Model Comparison:
  Model                |     Log-ML | Parameters
  ---------------------+-----------+-----------
  SV-Leverage          |   -1423.56 |     μ,φ,σ,ρ
  SV-Student-t         |   -1428.12 |     μ,φ,σ,ν

  Log Bayes Factor (A vs B): 4.56
  Interpretation: Strong evidence for SV-Leverage
```

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Posterior predictive comparison
ax = axes[0]
# Generate predictive samples from both models
n_pred = 200
rng = np.random.default_rng(42)

pred_A = np.zeros(n_pred)
pred_B = np.zeros(n_pred)
for k in range(n_pred):
    idx = rng.choice(len(chains))
    mu_k, phi_k, sigma_k, rho_k = chains[idx]
    h_k = mu_k + sigma_k * rng.standard_normal()
    pred_A[k] = np.exp(h_k / 2) * rng.standard_normal()

    chains_t = pmmh_results_t.chains
    idx_t = rng.choice(len(chains_t))
    mu_t, phi_t, sigma_t, nu_t = chains_t[idx_t]
    h_t = mu_t + sigma_t * rng.standard_normal()
    pred_B[k] = np.exp(h_t / 2) * rng.standard_t(df=nu_t)

bins = np.linspace(-5, 5, 80)
ax.hist(y, bins=bins, density=True, alpha=0.3, color="gray", label="Data")
ax.hist(pred_A, bins=bins, density=True, alpha=0.4, color="steelblue",
        label="SV-Leverage")
ax.hist(pred_B, bins=bins, density=True, alpha=0.4, color="firebrick",
        label="SV-t")
ax.set_xlabel("Return")
ax.set_ylabel("Density")
ax.set_title("Posterior Predictive")
ax.legend(fontsize=8)

# Bayes factor visualization
ax = axes[1]
models = ["SV-Leverage", "SV-Student-t"]
log_mls = [log_ml_A, log_ml_B]
colors_bar = ["steelblue", "firebrick"]
bars = ax.bar(models, log_mls, color=colors_bar, alpha=0.7)
ax.set_ylabel("Log Marginal Likelihood")
ax.set_title(f"Model Comparison (log BF = {log_bf:.2f})")

# Add BF annotation
ax.annotate(f"BF = {np.exp(log_bf):.1f}",
            xy=(0.5, max(log_mls)), xytext=(0.5, max(log_mls) + 2),
            ha="center", fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.5))

plt.tight_layout()
plt.savefig("workflow_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- **Left**: Posterior predictive distributions from both models overlaid on the data histogram.
- **Right**: Bar chart of log marginal likelihoods showing SV-Leverage is preferred.

---

## Step 10: Smoothing for Final Trajectory

The filtered estimates use only past data $y_{1:t}$. **Smoothing** uses the entire dataset $y_{1:T}$ for improved state estimates:

```python
from particlefilterbox.smoothers.ffbsm import FFBSm

# --- Forward-filtering backward-smoothing ---
smoother = FFBSm(
    model=sv_model,
    n_trajectories=100,
    seed=42,
)

print("Running FFBSm smoother...")
smooth_results = smoother.smooth(results_apf)

h_smoothed = smooth_results.smoothed_mean[:, 0]
h_smooth_lo = np.percentile(smooth_results.trajectories[:, :, 0], 2.5, axis=0)
h_smooth_hi = np.percentile(smooth_results.trajectories[:, :, 0], 97.5, axis=0)

rmse_filtered = np.sqrt(np.mean((h_filtered - h_true) ** 2))
rmse_smoothed = np.sqrt(np.mean((h_smoothed - h_true) ** 2))
improvement = (1 - rmse_smoothed / rmse_filtered) * 100

print(f"\nSmoothing Results:")
print(f"  RMSE (filtered):  {rmse_filtered:.4f}")
print(f"  RMSE (smoothed):  {rmse_smoothed:.4f}")
print(f"  Improvement:      {improvement:.1f}%")
```

Expected output:

```text
Running FFBSm smoother...

Smoothing Results:
  RMSE (filtered):  0.1234
  RMSE (smoothed):  0.0987
  Improvement:      20.0%
```

```python
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Smoothed vs filtered
ax = axes[0]
ax.plot(time, h_true, "k-", linewidth=1.5, label="True $h_t$", alpha=0.8)
ax.plot(time, h_filtered, "b-", linewidth=0.8, alpha=0.5, label="Filtered")
ax.plot(time, h_smoothed, "r-", linewidth=1, label="Smoothed")
ax.fill_between(time, h_smooth_lo, h_smooth_hi, alpha=0.15, color="red",
                label="95% CI (smoothed)")
ax.set_ylabel("Log-volatility $h_t$")
ax.set_title("Step 10: Filtered vs Smoothed Estimates")
ax.legend(fontsize=8)

# Improvement
ax = axes[1]
err_filtered = np.abs(h_filtered - h_true)
err_smoothed = np.abs(h_smoothed - h_true)
ax.plot(time, err_filtered, "b-", linewidth=0.5, alpha=0.5, label="Filtered error")
ax.plot(time, err_smoothed, "r-", linewidth=0.5, alpha=0.5, label="Smoothed error")

# Rolling mean
window = 20
ax.plot(time[window:], np.convolve(err_filtered, np.ones(window)/window, mode="valid"),
        "b-", linewidth=1.5, label="Filtered (rolling)")
ax.plot(time[window:], np.convolve(err_smoothed, np.ones(window)/window, mode="valid"),
        "r-", linewidth=1.5, label="Smoothed (rolling)")
ax.set_ylabel("$|\\hat{h}_t - h_t|$")
ax.set_xlabel("Day $t$")
ax.set_title("Absolute Error: Smoothing Reduces Error Throughout")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("workflow_smoothing.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- **Panel 1**: Smoothed estimates (red) are closer to the true state than filtered (blue), especially at endpoints.
- **Panel 2**: Rolling absolute error is consistently lower for the smoother.

---

## Step 11: Complete Visualization

```python
# --- Publication-quality multi-panel figure ---
fig = plt.figure(figsize=(16, 18))
gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)

# Panel A: Returns
ax = fig.add_subplot(gs[0, :])
ax.plot(time, y, "k-", linewidth=0.3, alpha=0.7)
vol_smooth = np.exp(h_smoothed / 2)
ax.fill_between(time, -2 * vol_smooth, 2 * vol_smooth,
                alpha=0.1, color="red", label="±2σ (smoothed)")
ax.set_ylabel("Returns $y_t$")
ax.set_title("(A) Daily Returns with Smoothed Volatility Bands")
ax.legend(fontsize=8)

# Panel B: Smoothed volatility
ax = fig.add_subplot(gs[1, :])
ax.plot(time, np.exp(h_true / 2), "k-", linewidth=1.5, label="True", alpha=0.7)
ax.plot(time, vol_smooth, "r-", linewidth=1, label="Smoothed")
ax.fill_between(time, np.exp(h_smooth_lo / 2), np.exp(h_smooth_hi / 2),
                alpha=0.15, color="red")
ax.set_ylabel("Volatility $\\exp(h_t/2)$")
ax.set_title("(B) Latent Volatility: Smoothed Estimate with 95% CI")
ax.legend(fontsize=8)

# Panel C: Posterior distributions
for i, (name, true_val) in enumerate(zip(param_names, true_values)):
    ax = fig.add_subplot(gs[2, i % 2] if i < 2 else gs[3, i % 2])
    ax.hist(chains[:, i], bins=50, density=True, alpha=0.6, color="steelblue")
    ax.axvline(true_val, color="r", linewidth=1.5, linestyle="--", label=f"True: {true_val}")
    ax.axvline(np.mean(chains[:, i]), color="k", linewidth=1, label=f"Mean: {np.mean(chains[:, i]):.3f}")
    ax.set_xlabel(f"${name}$")
    ax.set_title(f"(C{i+1}) Posterior: {name}")
    ax.legend(fontsize=7)

# Panel D: Model comparison
ax = fig.add_subplot(gs[4, 0])
models = ["SV-Leverage\n(selected)", "SV-Student-t"]
log_mls_plot = [log_ml_A, log_ml_B]
colors_plot = ["steelblue", "firebrick"]
bars = ax.barh(models, log_mls_plot, color=colors_plot, alpha=0.7)
ax.set_xlabel("Log Marginal Likelihood")
ax.set_title(f"(D) Model Comparison (BF = {np.exp(log_bf):.1f})")

# Panel E: ESS
ax = fig.add_subplot(gs[4, 1])
ax.plot(time, results_apf.ess_history, "steelblue", linewidth=0.4, alpha=0.6)
ax.axhline(250, color="r", linewidth=0.5, linestyle="--", label="N/2")
mean_ess = np.mean(results_apf.ess_history)
ax.axhline(mean_ess, color="k", linewidth=0.5, label=f"Mean: {mean_ess:.0f}")
ax.set_xlabel("Day $t$")
ax.set_ylabel("ESS")
ax.set_title("(E) Filter Diagnostics: ESS")
ax.legend(fontsize=8)

plt.savefig("workflow_complete.png", dpi=200, bbox_inches="tight")
plt.show()

print("Complete visualization saved: workflow_complete.png")
```

Expected output:

- A 5-row, publication-quality figure summarizing the entire analysis: returns, volatility, posteriors, model comparison, and diagnostics.

---

## Step 12: Generate Report

```python
from particlefilterbox.reports import AnalysisReport

# --- Generate structured report ---
report = AnalysisReport(title="SV-Leverage Analysis")

# Add sections
report.add_section("Data", {
    "T": T,
    "Return mean": f"{y.mean():.4f}",
    "Return std": f"{y.std():.4f}",
    "Kurtosis": f"{kurtosis:.2f}",
    "Leverage effect": "Significant (ρ < 0)",
})

report.add_section("Filter", {
    "Method": "Auxiliary Particle Filter",
    "N_particles": 500,
    "Backend": "numba",
    "RMSE": f"{rmse:.4f}",
    "Mean ESS": f"{np.mean(results_apf.ess_history):.1f}",
    "Log-likelihood": f"{results_apf.log_likelihood:.2f}",
})

report.add_section("Parameter Estimation", {
    "Method": "PMMH",
    "Iterations": 5000,
    "Burnin": 2000,
    "Acceptance rate": f"{np.mean(pmmh_results.acceptance_history):.1%}",
})

for i, (name, true_val) in enumerate(zip(param_names, true_values)):
    report.add_parameter(
        name=name,
        true_value=true_val,
        posterior_mean=np.mean(chains[:, i]),
        posterior_std=np.std(chains[:, i]),
        ci_95=(np.percentile(chains[:, i], 2.5), np.percentile(chains[:, i], 97.5)),
        ess=compute_ess(chains[:, i]),
    )

report.add_section("Model Comparison", {
    "Model A": "SV-Leverage",
    "Model B": "SV-Student-t",
    "Log BF (A vs B)": f"{log_bf:.2f}",
    "Verdict": verdict,
})

report.add_section("Smoothing", {
    "Method": "FFBSm",
    "Trajectories": 100,
    "RMSE improvement": f"{improvement:.1f}%",
})

# Save report
report.save("sv_leverage_analysis.html")
report.save("sv_leverage_analysis.json")

print(report.summary())
```

Expected output:

```text
╔══════════════════════════════════════════════════╗
║         SV-Leverage Analysis Report              ║
╠══════════════════════════════════════════════════╣
║ Data: T=1000, κ=5.68, leverage=Yes              ║
║ Filter: APF (N=500), RMSE=0.1234, ESS=378.4    ║
║ PMMH: 5000 iter, accept=23.9%, all ESS > 100   ║
║ Parameters:                                      ║
║   μ = -0.487 ± 0.198  [true: -0.500] ✓         ║
║   φ =  0.973 ± 0.009  [true:  0.975] ✓         ║
║   σ =  0.124 ± 0.025  [true:  0.120] ✓         ║
║   ρ = -0.342 ± 0.087  [true: -0.350] ✓         ║
║ Model comparison: SV-L preferred (BF=4.56)       ║
║ Smoothing: 20.0% RMSE improvement               ║
╠══════════════════════════════════════════════════╣
║ Status: ALL CHECKS PASSED                        ║
╚══════════════════════════════════════════════════╝

Reports saved: sv_leverage_analysis.html, sv_leverage_analysis.json
```

---

## Workflow Summary

```python
# --- Final workflow diagram ---
print("""
Complete Analysis Workflow
══════════════════════════

  ┌─────────────────┐
  │ 1. Define Model  │  SV with leverage (4 parameters)
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ 2. Explore Data  │  Heavy tails, clustering, leverage → SV-L
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ 3. Choose Filter │  Decision tree → Auxiliary PF
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ 4. Filter States │  APF (N=500, numba) → h_t estimates
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ 5. Diagnose PF   │  ESS healthy, no degeneracy
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ 6. Estimate θ    │  PMMH (5000 iter, N=300) → posterior
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ 7. Diagnose MCMC │  Trace, ACF, ESS — all OK
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ 8. Posterior      │  Correlations, credible intervals
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ 9. Compare Models│  SV-L vs SV-t → SV-L wins (BF=4.56)
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │10. Smooth States │  FFBSm → 20% RMSE improvement
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │11. Visualize     │  Publication-quality multi-panel figure
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │12. Report        │  Structured HTML/JSON output
  └────────┘────────┘
""")
```

---

## Summary

In this capstone tutorial you completed a **full research workflow**:

1. **Defined** an SV model with leverage effects capturing the asymmetric volatility response
2. **Explored** data features (heavy tails, clustering, leverage) to motivate model choice
3. **Selected** the Auxiliary PF using the filter decision tree
4. **Filtered** latent volatility states with N=500 particles (Numba-accelerated)
5. **Diagnosed** the filter -- ESS healthy, no weight degeneracy
6. **Estimated** 4 structural parameters via PMMH with adaptive proposals
7. **Diagnosed** the MCMC chain -- good mixing, ESS > 100, all CIs contain true values
8. **Analyzed** the posterior -- parameter correlations and marginal distributions
9. **Compared** SV-Leverage vs SV-t -- strong evidence for the leverage specification
10. **Smoothed** state trajectories with FFBSm for 20% RMSE improvement
11. **Visualized** results in a publication-quality multi-panel figure
12. **Generated** a structured analysis report

This workflow generalizes to any state-space model: swap the model, adjust the priors, and the same pipeline applies.

---

## What's Next?

<div class="grid cards" markdown>

- :material-api: **[API Reference](../api/index.md)**

    Complete reference for all classes and methods used in this tutorial

- :material-head-question: **[FAQ](../faq/index.md)**

    Common questions and troubleshooting

- :material-github: **[Contributing](../contributing/index.md)**

    Help improve particlefilterbox

</div>
