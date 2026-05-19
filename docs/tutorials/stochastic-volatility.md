---
title: "Tutorial: Stochastic Volatility Model"
description: Filter latent volatility with Bootstrap PF and SIR, run diagnostics, and apply to real financial data
---

# Tutorial: Stochastic Volatility Model

**Level**: :material-star-half-full:{.intermediate} Intermediate  
**Time**: ~45 minutes  
**Prerequisites**: [Fundamentals tutorial](fundamentals.md), basic knowledge of financial time series  

The **stochastic volatility (SV)** model is the classic particle filter application. The observation equation is nonlinear, so the Kalman filter cannot be used -- making this the simplest model that genuinely *requires* particle filtering.

---

## What You'll Learn

- Understand the SV model and its statistical properties
- Simulate data from the SV model
- Filter latent volatility with the Bootstrap PF
- Filter with SIR and compare performance
- Diagnose filter health with ESS and weight analysis
- Visualize filtered vs true volatility
- Apply particle filtering to real financial data

---

## Step 1: The Stochastic Volatility Model

The standard SV model describes asset returns $y_t$ with time-varying log-volatility $h_t$:

$$
h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta \eta_t, \qquad \eta_t \sim \mathcal{N}(0, 1)
$$

$$
y_t = \exp(h_t / 2) \, \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1)
$$

where:

- $h_t$ is the **log-volatility** (hidden state)
- $\mu$ is the **long-run mean** of log-volatility
- $\phi \in (-1, 1)$ is the **persistence** (typically $\phi \approx 0.95$--$0.99$)
- $\sigma_\eta$ is the **volatility of volatility**
- $y_t$ are observed **returns**

### Why particle filtering?

The observation equation $y_t = \exp(h_t/2)\varepsilon_t$ is **nonlinear** in the state $h_t$. Squaring and taking logs gives $\log y_t^2 = h_t + \log \varepsilon_t^2$, where $\log \varepsilon_t^2 \sim \log\chi^2_1$ -- a highly skewed, non-Gaussian distribution. Neither transformation yields a linear-Gaussian model, so the Kalman filter cannot provide exact inference.

!!! info "SV model parameters"

    | Parameter | Symbol | Typical range | Interpretation |
    |-----------|--------|---------------|----------------|
    | Long-run mean | $\mu$ | $[-1, 1]$ | Average log-volatility level |
    | Persistence | $\phi$ | $[0.90, 0.99]$ | How slowly volatility reverts to mean |
    | Vol-of-vol | $\sigma_\eta$ | $[0.05, 0.30]$ | How rapidly volatility fluctuates |

---

## Step 2: Simulate Data from the SV Model

Let's generate synthetic data where we know the true volatility:

```python
import numpy as np
from particlefilterbox.models.sv import SVModel

# --- Define model with known parameters ---
true_params = {"mu": 0.0, "phi": 0.97, "sigma_eta": 0.15}
model = SVModel(variant="basic", params=true_params)

print(f"Model: Stochastic Volatility ({model.variant})")
print(f"States: {model.k_states}, Observations: {model.k_obs}")
print(f"Parameters: {model.params}")

# --- Simulate T=500 observations ---
np.random.seed(123)
states, obs = model.simulate(n_obs=500)

h_true = states[:, 0]       # true log-volatility
vol_true = np.exp(h_true / 2)  # true volatility (std dev)
returns = obs[:, 0]          # observed returns

print(f"\nSimulated data:")
print(f"  Time steps:        {len(returns)}")
print(f"  Mean return:       {np.mean(returns):.4f}")
print(f"  Std return:        {np.std(returns):.4f}")
print(f"  Mean |return|:     {np.mean(np.abs(returns)):.4f}")
print(f"  Mean volatility:   {np.mean(vol_true):.4f}")
print(f"  Max volatility:    {np.max(vol_true):.4f}")
```

Expected output:

```text
Model: Stochastic Volatility (basic)
States: 1, Observations: 1
Parameters: {'mu': 0.0, 'phi': 0.97, 'sigma_eta': 0.15}

Simulated data:
  Time steps:        500
  Mean return:       0.0032
  Std return:        1.0842
  Mean |return|:     0.7965
  Mean volatility:   1.0523
  Max volatility:    2.1847
```

```python
import matplotlib.pyplot as plt
from particlefilterbox.visualization import set_theme

set_theme("nodesecon")

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
time = np.arange(len(returns))

# Returns
axes[0].plot(time, returns, "k-", linewidth=0.5)
axes[0].set_ylabel("Returns $y_t$")
axes[0].set_title("Simulated Stochastic Volatility Data")

# True volatility
axes[1].plot(time, vol_true, "r-", linewidth=1)
axes[1].set_ylabel("Volatility $\\exp(h_t/2)$")
axes[1].set_title("True Volatility (unobserved)")

# Log-volatility
axes[2].plot(time, h_true, "b-", linewidth=1)
axes[2].axhline(true_params["mu"], color="gray", linestyle="--", linewidth=0.5)
axes[2].set_ylabel("Log-volatility $h_t$")
axes[2].set_xlabel("Time step $t$")
axes[2].set_title("True Log-Volatility (unobserved)")

plt.tight_layout()
plt.savefig("sv_simulated_data.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- **Top**: Returns show volatility clustering -- periods of large returns clustered together.
- **Middle**: True volatility rises and falls slowly (high persistence $\phi = 0.97$).
- **Bottom**: Log-volatility mean-reverts around $\mu = 0$.

---

## Step 3: Filter Volatility with Bootstrap PF

```python
from particlefilterbox.filters.bootstrap import BootstrapFilter
from particlefilterbox.core import PFConfig

# --- Configure and run Bootstrap PF ---
config = PFConfig(
    n_particles=2000,
    resampling="systematic",
    ess_threshold=0.5,
    seed=42,
    store_particles=True,   # needed for diagnostics later
    store_weights=True,
)

pf_bootstrap = BootstrapFilter(model=model, config=config)
results_bpf = pf_bootstrap.filter(obs)

# --- Results ---
h_est = results_bpf.filtered_mean[:, 0]
rmse = np.sqrt(np.mean((h_est - h_true) ** 2))

print(f"Bootstrap PF results:")
print(f"  Particles:    {config.n_particles}")
print(f"  RMSE (h_t):   {rmse:.4f}")
print(f"  Mean ESS:     {np.mean(results_bpf.ess_history):.1f}")
print(f"  Min ESS:      {np.min(results_bpf.ess_history):.1f}")
print(f"  Log-lik:      {results_bpf.log_likelihood:.2f}")
print(f"  Time:         {results_bpf.computation_time:.3f}s")
```

Expected output:

```text
Bootstrap PF results:
  Particles:    2000
  RMSE (h_t):   0.2815
  Mean ESS:     1423.7
  Min ESS:      312.5
  Log-lik:      -742.31
  Time:         0.124s
```

---

## Step 4: Filter with SIR and Compare

The **SIR (Sequential Importance Resampling)** filter can use a better proposal distribution than the Bootstrap PF's transition prior. For the SV model, the SIR filter can incorporate observation information into the proposal, potentially improving efficiency.

```python
from particlefilterbox.filters.sir import SIRFilter

# --- Run SIR Filter ---
pf_sir = SIRFilter(model=model, config=config)
results_sir = pf_sir.filter(obs)

h_est_sir = results_sir.filtered_mean[:, 0]
rmse_sir = np.sqrt(np.mean((h_est_sir - h_true) ** 2))

print(f"SIR Filter results:")
print(f"  RMSE (h_t):   {rmse_sir:.4f}")
print(f"  Mean ESS:     {np.mean(results_sir.ess_history):.1f}")
print(f"  Min ESS:      {np.min(results_sir.ess_history):.1f}")
print(f"  Log-lik:      {results_sir.log_likelihood:.2f}")
print(f"  Time:         {results_sir.computation_time:.3f}s")

# --- Comparison ---
print(f"\nComparison:")
print(f"  {'Metric':<15} | {'Bootstrap':>10} | {'SIR':>10}")
print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}")
print(f"  {'RMSE':<15} | {rmse:>10.4f} | {rmse_sir:>10.4f}")
print(f"  {'Mean ESS':<15} | {np.mean(results_bpf.ess_history):>10.1f} | {np.mean(results_sir.ess_history):>10.1f}")
print(f"  {'Min ESS':<15} | {np.min(results_bpf.ess_history):>10.1f} | {np.min(results_sir.ess_history):>10.1f}")
print(f"  {'Log-lik':<15} | {results_bpf.log_likelihood:>10.2f} | {results_sir.log_likelihood:>10.2f}")
```

Expected output:

```text
SIR Filter results:
  RMSE (h_t):   0.2734
  Mean ESS:     1512.3
  Min ESS:      387.2
  Log-lik:      -741.89
  Time:         0.138s

Comparison:
  Metric          |  Bootstrap |        SIR
  ----------------+------------+-----------
  RMSE            |     0.2815 |     0.2734
  Mean ESS        |     1423.7 |     1512.3
  Min ESS         |       312.5 |      387.2
  Log-lik         |    -742.31 |    -741.89
```

!!! note "Bootstrap vs SIR for SV"
    The SIR filter achieves slightly better RMSE and higher ESS because its
    observation-informed proposal places particles closer to the posterior.
    The improvement is modest for the basic SV model but becomes more pronounced
    for models with highly informative observations.

---

## Step 5: Diagnostics -- ESS and Weight Degeneracy

Diagnosing filter health is crucial. Let's examine ESS trajectories and weight distributions:

```python
from particlefilterbox.visualization import plot_ess_timeline, plot_weight_histogram

# --- ESS Timeline ---
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax = axes[0]
ax.plot(time, results_bpf.ess_history, "b-", linewidth=0.8, alpha=0.7, label="Bootstrap PF")
ax.plot(time, results_sir.ess_history, "r-", linewidth=0.8, alpha=0.7, label="SIR")
ax.axhline(0.5 * config.n_particles, color="k", linestyle="--", linewidth=0.5, label="Threshold")
ax.set_ylabel("ESS")
ax.set_title("Effective Sample Size Over Time")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(time, np.abs(returns), "gray", linewidth=0.5, alpha=0.7)
ax.set_ylabel("$|y_t|$")
ax.set_xlabel("Time step $t$")
ax.set_title("Absolute Returns (observation magnitude)")

plt.tight_layout()
plt.savefig("sv_ess_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- ESS drops when absolute returns are large (the observation becomes very informative, concentrating weight on fewer particles).
- The SIR filter maintains consistently higher ESS than the Bootstrap PF.

```python
# --- Weight distribution at a high-volatility time point ---
# Find time with lowest ESS
t_worst = np.argmin(results_bpf.ess_history)
print(f"Worst ESS at t={t_worst}: {results_bpf.ess_history[t_worst]:.1f}")
print(f"Observation at t={t_worst}: y={returns[t_worst]:.4f}")
print(f"True h at t={t_worst}: h={h_true[t_worst]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bootstrap weights
w_bpf = results_bpf.weight_history[t_worst]
axes[0].hist(w_bpf, bins=50, color="blue", alpha=0.7, density=True)
axes[0].set_title(f"Bootstrap PF Weights at t={t_worst}\n(ESS={results_bpf.ess_history[t_worst]:.0f})")
axes[0].set_xlabel("Normalized weight")
axes[0].set_ylabel("Density")

# SIR weights
w_sir = results_sir.weight_history[t_worst]
axes[1].hist(w_sir, bins=50, color="red", alpha=0.7, density=True)
axes[1].set_title(f"SIR Weights at t={t_worst}\n(ESS={results_sir.ess_history[t_worst]:.0f})")
axes[1].set_xlabel("Normalized weight")

plt.tight_layout()
plt.savefig("sv_weight_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- Bootstrap PF weights are highly concentrated -- a few particles carry most weight (degeneracy).
- SIR weights are more uniform, indicating better particle diversity.

!!! warning "Weight degeneracy"
    When $\text{ESS} \ll N$, the filter relies on very few particles for state
    estimation. This increases variance and can lead to poor estimates. If you see
    persistent low ESS ($< 0.2N$), consider:

    1. Increasing $N$
    2. Using a better proposal (SIR, Auxiliary PF)
    3. Checking model specification

---

## Step 6: Visualize Filtered Volatility vs True Volatility

```python
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# --- Panel 1: Log-volatility ---
ax = axes[0]
h_bpf = results_bpf.filtered_mean[:, 0]
h_sir = results_sir.filtered_mean[:, 0]
std_bpf = np.sqrt(results_bpf.filtered_cov[:, 0, 0])

ax.plot(time, h_true, "k-", linewidth=1.5, label="True $h_t$", alpha=0.8)
ax.plot(time, h_bpf, "b-", linewidth=1, label="Bootstrap PF")
ax.plot(time, h_sir, "r--", linewidth=1, label="SIR")
ax.fill_between(
    time,
    h_bpf - 1.96 * std_bpf,
    h_bpf + 1.96 * std_bpf,
    alpha=0.15,
    color="blue",
    label="95% CI (Bootstrap)",
)
ax.set_ylabel("Log-volatility $h_t$")
ax.set_title("Filtered Log-Volatility")
ax.legend(fontsize=8)

# --- Panel 2: Volatility (exp scale) ---
ax = axes[1]
vol_bpf = np.exp(h_bpf / 2)
vol_sir = np.exp(h_sir / 2)

ax.plot(time, vol_true, "k-", linewidth=1.5, label="True vol", alpha=0.8)
ax.plot(time, vol_bpf, "b-", linewidth=1, label="Bootstrap PF")
ax.plot(time, vol_sir, "r--", linewidth=1, label="SIR")
ax.set_ylabel("Volatility $\\exp(h_t/2)$")
ax.set_title("Filtered Volatility")
ax.legend(fontsize=8)

# --- Panel 3: Returns with filtered vol bands ---
ax = axes[2]
ax.plot(time, returns, "gray", linewidth=0.5, alpha=0.7)
ax.fill_between(
    time, -1.96 * vol_bpf, 1.96 * vol_bpf,
    alpha=0.2, color="blue", label="$\\pm 1.96 \\hat{\\sigma}_t$ (Bootstrap)"
)
ax.set_ylabel("Returns $y_t$")
ax.set_xlabel("Time step $t$")
ax.set_title("Returns with Filtered Volatility Bands")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("sv_filtered_volatility.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- **Panel 1**: Both filters track the true log-volatility closely, with the 95% CI containing the truth.
- **Panel 2**: Filtered volatility on the natural scale captures the rise-and-fall dynamics.
- **Panel 3**: The volatility bands widen during high-volatility periods, capturing the heteroscedasticity in returns.

---

## Step 7: Apply to Real Financial Data

Now let's apply our SV filter to real financial returns. We'll use the S&P 500 data included with particlefilterbox:

```python
from particlefilterbox.datasets import load_dataset

# --- Load S&P 500 returns ---
data = load_dataset("sp500_returns")

print(f"Dataset: S&P 500 daily returns")
print(f"  Period:     {data['start_date']} to {data['end_date']}")
print(f"  N obs:      {len(data['returns'])}")
print(f"  Mean:       {np.mean(data['returns']):.6f}")
print(f"  Std:        {np.std(data['returns']):.6f}")
print(f"  Skewness:   {data.get('skewness', 'N/A')}")
print(f"  Kurtosis:   {data.get('kurtosis', 'N/A')}")
```

Expected output:

```text
Dataset: S&P 500 daily returns
  Period:     2018-01-02 to 2023-12-29
  N obs:      1508
  Mean:       0.000412
  Std:        0.012543
  Skewness:   -0.68
  Kurtosis:   14.23
```

```python
# --- Standardize returns (zero mean, unit variance) ---
returns_real = data["returns"]
returns_std = (returns_real - np.mean(returns_real)) / np.std(returns_real)
dates = data.get("dates", np.arange(len(returns_std)))

# --- Define SV model with typical parameters ---
# Initial parameter guesses for daily equity returns
model_real = SVModel(
    variant="basic",
    params={"mu": 0.0, "phi": 0.98, "sigma_eta": 0.10},
)

# --- Run Bootstrap PF ---
config = PFConfig(n_particles=3000, resampling="systematic", seed=42)
pf = BootstrapFilter(model=model_real, config=config)
results_real = pf.filter(returns_std.reshape(-1, 1))

h_filtered = results_real.filtered_mean[:, 0]
vol_filtered = np.exp(h_filtered / 2)

print(f"\nFiltering results:")
print(f"  Mean ESS:     {np.mean(results_real.ess_history):.0f}")
print(f"  Min ESS:      {np.min(results_real.ess_history):.0f}")
print(f"  Log-lik:      {results_real.log_likelihood:.2f}")
print(f"  Mean vol:     {np.mean(vol_filtered):.4f}")
print(f"  Max vol:      {np.max(vol_filtered):.4f}")
```

Expected output:

```text
Filtering results:
  Mean ESS:     2134
  Min ESS:      423
  Log-lik:      -2089.45
  Mean vol:     0.9821
  Max vol:      3.5672
```

```python
# --- Visualize real data results ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

t = np.arange(len(returns_std))

# Returns
ax = axes[0]
ax.plot(t, returns_std, "k-", linewidth=0.3)
ax.set_ylabel("Standardized returns")
ax.set_title("S&P 500 Standardized Returns")

# Filtered volatility
ax = axes[1]
ax.plot(t, vol_filtered, "b-", linewidth=0.8)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
ax.set_ylabel("Filtered $\\hat{\\sigma}_t$")
ax.set_title("Filtered Volatility (Bootstrap PF)")

# Returns with volatility bands
ax = axes[2]
ax.plot(t, returns_std, "gray", linewidth=0.3, alpha=0.7)
ax.fill_between(
    t, -1.96 * vol_filtered, 1.96 * vol_filtered,
    alpha=0.25, color="blue", label="$\\pm 1.96 \\hat{\\sigma}_t$"
)
ax.set_ylabel("Returns")
ax.set_xlabel("Observation index")
ax.set_title("Returns with Filtered Volatility Bands")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("sv_real_data.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- **Panel 1**: Standardized returns show clear volatility clustering with spikes during market stress.
- **Panel 2**: Filtered volatility rises sharply during crisis periods and slowly mean-reverts during calm periods.
- **Panel 3**: The $\pm 1.96\hat{\sigma}_t$ bands adapt to changing volatility, widening during turbulent periods.

!!! tip "From filtering to estimation"
    In this tutorial, we used **fixed parameters** for the SV model. In practice,
    you'll want to **estimate** $\mu$, $\phi$, and $\sigma_\eta$ from the data.
    See the [PMMH Tutorial](pmmh.md) for Bayesian parameter estimation using
    Particle MCMC.

!!! info "Interpreting the filtered volatility"
    - Volatility $> 1$ means the market is **more volatile** than average
    - Volatility $< 1$ means the market is **less volatile** than average
    - Sharp spikes correspond to market events (COVID crash, rate hikes, etc.)
    - The slow decay reflects the high persistence ($\phi = 0.98$) of the SV process

---

## Summary

In this tutorial you learned:

1. The **SV model** captures time-varying volatility with a nonlinear observation equation
2. The **Bootstrap PF** can filter latent log-volatility from observed returns
3. The **SIR filter** offers modest improvements through observation-informed proposals
4. **ESS diagnostics** reveal when the filter struggles (large observations concentrate weight)
5. **Weight histograms** visualize particle degeneracy
6. The **filtered volatility** tracks true volatility and captures clustering in real data
7. Real financial data shows the particle filter's practical value for **risk monitoring**

---

## What's Next?

<div class="grid cards" markdown>

- :material-flash: **[Auxiliary PF Tutorial](auxiliary-pf.md)**

    Learn how look-ahead resampling handles challenging models with jumps

- :material-tune-vertical: **[PMMH Tutorial](pmmh.md)**

    Estimate SV model parameters from data using Particle MCMC

- :material-math-integral: **[SV Theory](../theory/sv-theory.md)**

    Mathematical foundations of the stochastic volatility model

</div>
