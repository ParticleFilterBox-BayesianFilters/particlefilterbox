---
title: "Diagnostics API"
description: "API reference for particlefilterbox.diagnostics — filter diagnostics, MCMC convergence, Kalman validation, and predictive checks"
---

# Diagnostics API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.diagnostics import ESSDiagnostic, WeightDiagnostic, ConvergenceDiagnostic, DegeneracyDiagnostic, FilterComparison, KalmanValidation, MCMCConvergence, MixingDiagnostic, PredictiveCheck, MarginalLikelihood`
    **Source**: `particlefilterbox/diagnostics/`

## Overview

The diagnostics module provides **class-based** diagnostic tools organized into four families:

| Family | Classes | Purpose |
|--------|---------|---------|
| **Filter** | `ESSDiagnostic`, `WeightDiagnostic`, `ConvergenceDiagnostic`, `DegeneracyDiagnostic` | Particle-level health checks |
| **Comparison** | `FilterComparison`, `KalmanValidation` | Cross-filter validation |
| **MCMC** | `MCMCConvergence`, `MixingDiagnostic` | PMCMC chain quality |
| **Model** | `PredictiveCheck`, `MarginalLikelihood` | Model adequacy and comparison |

All diagnostics share a common pattern:

```python
diag = DiagnosticClass(**params)
report = diag.run(result)   # DiagnosticReport
print(report.summary())
```

Each `DiagnosticReport` exposes `passed`, `statistic`, `pvalue` (where applicable), and a `plot()` method.

---

## Filter Diagnostics

### ESSDiagnostic

Monitors Effective Sample Size (ESS) along the filter run. ESS is defined as:

$$
\text{ESS}_t = \frac{1}{\sum_{i=1}^{N} (W_t^{(i)})^2} \in [1, N]
$$

Low ESS (relative to $N$) indicates weight concentration and loss of effective particles.

#### Constructor

```python
ESSDiagnostic(
    threshold: float = 0.5,
    window: int | None = None,
    alert_fraction: float = 0.1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.5` | ESS threshold as fraction of $N$ |
| `window` | `int \| None` | `None` | Rolling window size for local ESS analysis |
| `alert_fraction` | `float` | `0.1` | Flag if fraction of timesteps below threshold exceeds this |

#### Methods

##### `run()`

```python
def run(self, result: ParticleFilterResults) -> ESSReport
```

**Returns**: `ESSReport` with attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `passed` | `bool` | True if no alert triggered |
| `ess_history` | `NDArray[np.float64]` | ESS at each time step |
| `ess_mean` | `float` | Mean ESS over run |
| `ess_min` | `float` | Minimum ESS |
| `frac_below_threshold` | `float` | Fraction of steps below threshold |

#### Example

```python
import particlefilterbox as pfb

diag = pfb.diagnostics.ESSDiagnostic(threshold=0.5)
report = diag.run(filter_results)

if not report.passed:
    print(f"ESS below threshold at {report.frac_below_threshold:.1%} of steps")
    print(f"Minimum ESS: {report.ess_min:.0f}")

report.plot()
```

---

### WeightDiagnostic

Analyzes particle-weight distribution at each time step. Detects weight skewness, max-weight domination, and weight entropy.

#### Constructor

```python
WeightDiagnostic(
    max_weight_threshold: float = 0.1,
    entropy_threshold: float | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_weight_threshold` | `float` | `0.1` | Flag if $\max_i W_t^{(i)}$ exceeds this fraction |
| `entropy_threshold` | `float \| None` | `None` | Minimum relative entropy $H / \log N$ |

#### Methods

##### `run()`

```python
def run(self, result: ParticleFilterResults) -> WeightReport
```

**Returns**: `WeightReport` with `max_weight_history`, `entropy_history`, `gini_history`, `passed`.

#### Example

```python
diag = pfb.diagnostics.WeightDiagnostic(max_weight_threshold=0.05)
report = diag.run(filter_results)
print(f"Peak max-weight: {report.max_weight_history.max():.3f}")
```

---

### ConvergenceDiagnostic

Checks convergence of particle estimates with respect to $N$ by running the filter at multiple particle counts and comparing log-likelihood / filtered-state estimates.

#### Constructor

```python
ConvergenceDiagnostic(
    n_particles_grid: list[int] | NDArray[np.int64],
    n_replicates: int = 10,
    tolerance: float = 0.05,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles_grid` | `list[int]` | *required* | Particle counts to test, e.g., `[500, 1000, 5000, 10000]` |
| `n_replicates` | `int` | `10` | Independent runs per particle count |
| `tolerance` | `float` | `0.05` | Relative tolerance for convergence |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

#### Methods

##### `run()`

```python
def run(
    self,
    model: ParticleFilterModel,
    observations: NDArray[np.float64],
    filter_class: type[BaseParticleFilter] = BootstrapPF,
) -> ConvergenceReport
```

**Returns**: `ConvergenceReport` with log-likelihood mean/std per $N$ and convergence verdict.

#### Example

```python
diag = pfb.diagnostics.ConvergenceDiagnostic(
    n_particles_grid=[500, 1000, 5000, 10000],
    n_replicates=20,
)
report = diag.run(model, observations)
report.plot()   # log-likelihood vs. N with error bars
```

---

### DegeneracyDiagnostic

Detects path-space degeneracy: the collapse of unique ancestral trajectories as $t$ grows. For smoothing and PMCMC, path degeneracy is often the primary failure mode.

$$
\text{UniqueAncestors}_t = \#\{i : \exists j \text{ with } a_{t \to T}^{(j)} = i\}
$$

#### Constructor

```python
DegeneracyDiagnostic(
    min_unique_fraction: float = 0.1,
    track_ancestors: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_unique_fraction` | `float` | `0.1` | Flag if unique ancestors at $t=0$ falls below this |
| `track_ancestors` | `bool` | `True` | Trace full ancestor tree (memory-heavy for long $T$) |

#### Methods

##### `run()`

```python
def run(self, result: ParticleFilterResults) -> DegeneracyReport
```

**Returns**: `DegeneracyReport` with `unique_ancestors_history`, `coalescence_time`, `passed`.

#### Example

```python
diag = pfb.diagnostics.DegeneracyDiagnostic(min_unique_fraction=0.1)
report = diag.run(filter_results)

if not report.passed:
    print(f"Severe path degeneracy: {report.unique_ancestors_history[0]} unique ancestors at t=0")
```

---

## Comparison Diagnostics

### FilterComparison

Compares the output of multiple filters on the same data. Reports agreement metrics (log-likelihood difference, Wasserstein distance between filtered distributions) and runtime.

#### Constructor

```python
FilterComparison(
    filters: list[BaseParticleFilter],
    metrics: list[str] = ("log_likelihood", "filtered_mean", "runtime"),
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filters` | `list[BaseParticleFilter]` | *required* | Instantiated filters to compare |
| `metrics` | `list[str]` | `("log_likelihood", "filtered_mean", "runtime")` | Metrics to evaluate |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

#### Methods

##### `run()`

```python
def run(
    self,
    observations: NDArray[np.float64],
    n_replicates: int = 5,
) -> ComparisonReport
```

**Returns**: `ComparisonReport` with per-filter statistics and pairwise comparison table.

#### Example

```python
model = pfb.models.StochasticVolatility()
config = pfb.PFConfig(n_particles=2000)

filters = [
    pfb.BootstrapPF(model, config),
    pfb.AuxiliaryPF(model, config),
    pfb.GuidedPF(model, config),
]

diag = pfb.diagnostics.FilterComparison(filters)
report = diag.run(observations, n_replicates=10)
print(report.to_dataframe())
```

---

### KalmanValidation

Validates a particle filter against a Kalman filter on a linear-Gaussian (or linearized) sub-model. Useful as a sanity check: for linear-Gaussian dynamics, the Kalman filter is exact, so the PF should converge to it.

Uses [kalmanbox](https://github.com/nodesecon/kalmanbox) internally.

#### Constructor

```python
KalmanValidation(
    linear_model: LinearGaussianModel,
    tolerance: float = 0.05,
    metric: Literal["log_lik", "kl", "mse"] = "log_lik",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `linear_model` | `LinearGaussianModel` | *required* | Linear-Gaussian model (state matrices for Kalman) |
| `tolerance` | `float` | `0.05` | Acceptable relative error |
| `metric` | `str` | `"log_lik"` | Comparison metric |

#### Methods

##### `run()`

```python
def run(
    self,
    pf: BaseParticleFilter,
    observations: NDArray[np.float64],
) -> KalmanReport
```

**Returns**: `KalmanReport` with `pf_log_lik`, `kalman_log_lik`, `passed`, per-step errors.

#### Example

```python
import particlefilterbox as pfb

linear_model = pfb.models.LinearGaussian(dim=2)
pf = pfb.BootstrapPF(linear_model, pfb.PFConfig(n_particles=5000))

diag = pfb.diagnostics.KalmanValidation(linear_model, tolerance=0.02)
report = diag.run(pf, observations)

print(f"PF log-lik:     {report.pf_log_lik:.3f}")
print(f"Kalman log-lik: {report.kalman_log_lik:.3f}")
print(f"Validation:     {'PASS' if report.passed else 'FAIL'}")
```

!!! tip
    Run `KalmanValidation` whenever you implement a new filter or model — matching a Kalman result on a linear-Gaussian problem is a strong correctness check.

---

## MCMC Diagnostics

### MCMCConvergence

Computes convergence diagnostics for one or more MCMC chains: $\hat{R}$ (Gelman-Rubin), effective sample size, Geweke z-score, and Heidelberger-Welch stationarity test.

#### Constructor

```python
MCMCConvergence(
    rhat_threshold: float = 1.05,
    ess_threshold: int = 400,
    geweke_window: float = 0.1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rhat_threshold` | `float` | `1.05` | Maximum acceptable $\hat{R}$ |
| `ess_threshold` | `int` | `400` | Minimum acceptable ESS per parameter |
| `geweke_window` | `float` | `0.1` | Fraction of chain for Geweke first/last windows |

#### Methods

##### `run()`

```python
def run(self, chains: MCMCChain | list[MCMCChain]) -> MCMCConvergenceReport
```

**Returns**: `MCMCConvergenceReport` with per-parameter $\hat{R}$, ESS, Geweke z, `passed`.

#### Example

```python
# Run multiple independent chains
chains = [
    pfb.PMMH(model, **kwargs).sample(observations)
    for seed in range(4)
]

diag = pfb.diagnostics.MCMCConvergence(rhat_threshold=1.01)
report = diag.run(chains)

print(report.summary())
# Parameter  Rhat  ESS   Geweke-z  Passed
# mu         1.00  1850  -0.21     True
# phi        1.02  920   0.45      True
# sigma_eta  1.04  610   1.12      True
```

---

### MixingDiagnostic

Analyzes autocorrelation structure and mixing speed of MCMC chains. Computes integrated autocorrelation time (IAT) and detects slow-mixing parameters.

$$
\tau_{\text{int}} = 1 + 2 \sum_{k=1}^{\infty} \rho(k), \qquad \text{ESS} = \frac{n}{\tau_{\text{int}}}
$$

#### Constructor

```python
MixingDiagnostic(
    max_lag: int | None = None,
    iat_threshold: float = 50.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_lag` | `int \| None` | `None` | Maximum lag for ACF computation (default: automatic) |
| `iat_threshold` | `float` | `50.0` | Flag parameters with IAT above this |

#### Methods

##### `run()`

```python
def run(self, chain: MCMCChain) -> MixingReport
```

**Returns**: `MixingReport` with `iat`, `acf`, `thinning_suggestion`, `passed`.

#### Example

```python
diag = pfb.diagnostics.MixingDiagnostic(iat_threshold=100.0)
report = diag.run(chain)

for name, iat in report.iat.items():
    print(f"{name}: IAT = {iat:.1f}, suggested thin = {int(iat)}")

report.plot_acf()
```

---

## Model Diagnostics

### PredictiveCheck

Posterior predictive check: draws from the posterior predictive distribution and compares to observed data via user-specified test statistics.

$$
T(y^{\text{rep}}) \sim p(y^{\text{rep}} \mid y), \qquad p_B = \mathbb{P}(T(y^{\text{rep}}) \geq T(y) \mid y)
$$

Bayesian p-values near 0 or 1 indicate model misfit.

#### Constructor

```python
PredictiveCheck(
    statistics: dict[str, Callable],
    n_replicates: int = 500,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `statistics` | `dict[str, Callable]` | *required* | Test statistics (name → function `y -> float`) |
| `n_replicates` | `int` | `500` | Number of replicated datasets |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

#### Methods

##### `run()`

```python
def run(
    self,
    model: ParticleFilterModel,
    chain: MCMCChain,
    observations: NDArray[np.float64],
) -> PredictiveReport
```

**Returns**: `PredictiveReport` with per-statistic Bayesian p-values and replicated distributions.

#### Example

```python
import numpy as np

statistics = {
    "mean":     lambda y: y.mean(),
    "std":      lambda y: y.std(),
    "kurtosis": lambda y: ((y - y.mean()) ** 4).mean() / y.std() ** 4,
    "max":      lambda y: y.max(),
}

diag = pfb.diagnostics.PredictiveCheck(statistics=statistics, n_replicates=1000)
report = diag.run(model, chain, observations)

print(report.bayesian_pvalues)
# {"mean": 0.48, "std": 0.52, "kurtosis": 0.03, "max": 0.91}
# kurtosis p-value of 0.03 suggests the model underestimates tail behavior
```

---

### MarginalLikelihood

Estimates the log-marginal-likelihood $\log p(y_{1:T})$ for model comparison (Bayes factors). Supports several estimators suited to different output types.

$$
\log p(y_{1:T}) = \log \int p(y_{1:T} \mid \theta) \, p(\theta) \, d\theta
$$

| Estimator | Required input | Notes |
|-----------|---------------|-------|
| `"smc"` | `SMCResult` | Unbiased via SMC identity |
| `"pf"` | `ParticleFilterResults` | Conditional on $\theta$ (plug-in) |
| `"harmonic"` | `MCMCChain` | Not recommended (high variance) |
| `"bridge"` | `MCMCChain` + proposal | Bridge sampling (robust) |

#### Constructor

```python
MarginalLikelihood(
    method: Literal["smc", "pf", "harmonic", "bridge"] = "smc",
    n_bridge_samples: int = 1000,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"smc"` | Estimator |
| `n_bridge_samples` | `int` | `1000` | Samples for bridge sampling |

#### Methods

##### `run()`

```python
def run(self, result: SMCResult | ParticleFilterResults | MCMCChain) -> MarginalReport
```

**Returns**: `MarginalReport` with `log_marginal`, `std_err`, `method`.

##### `compare()`

Compute log Bayes factors between two models.

```python
def compare(
    self,
    report_a: MarginalReport,
    report_b: MarginalReport,
) -> float
```

**Returns**: $\log B_{AB} = \log p(y \mid M_A) - \log p(y \mid M_B)$.

#### Example

```python
# Compare two SV variants
ml = pfb.diagnostics.MarginalLikelihood(method='smc')

sv_basic = pfb.SMCTempering(target=target_basic, n_particles=3000).sample()
sv_t     = pfb.SMCTempering(target=target_t,     n_particles=3000).sample()

report_basic = ml.run(sv_basic)
report_t     = ml.run(sv_t)

log_bf = ml.compare(report_t, report_basic)
print(f"log BF (t-SV vs basic): {log_bf:.2f}")
# log BF > 2.3 = strong evidence for t-SV (Kass & Raftery, 1995)
```

---

## Combined Workflow

A typical diagnostic workflow for a PMCMC estimate:

```python
import particlefilterbox as pfb

# Run PMMH
chain = pmmh.sample(observations)

# 1. Filter-level diagnostics (on the inner particle filter)
pf = pfb.BootstrapPF(model.with_params(**chain.posterior_mean()), config)
pf_result = pf.filter(observations)
ess_report = pfb.diagnostics.ESSDiagnostic().run(pf_result)

# 2. Validate against Kalman (if applicable)
kalman_report = pfb.diagnostics.KalmanValidation(linear_model).run(pf, observations)

# 3. MCMC convergence
mcmc_report = pfb.diagnostics.MCMCConvergence().run([chain_1, chain_2, chain_3, chain_4])

# 4. Mixing
mixing_report = pfb.diagnostics.MixingDiagnostic().run(chain)

# 5. Posterior predictive
pp_report = pfb.diagnostics.PredictiveCheck(statistics=my_stats).run(model, chain, observations)

# 6. Marginal likelihood for model comparison
ml = pfb.diagnostics.MarginalLikelihood(method='smc')
ml_report = ml.run(smc_result)

# Aggregate
all_passed = all([
    ess_report.passed,
    kalman_report.passed,
    mcmc_report.passed,
    mixing_report.passed,
])
print(f"All diagnostics passed: {all_passed}")
```

---

## See Also

- [User Guide: Diagnostics](../user-guide/diagnostics/index.md) — In-depth diagnostic workflows
- [User Guide: Filter Diagnostics](../user-guide/diagnostics/filter-diagnostics.md) — Filter-level diagnostics
- [User Guide: MCMC Diagnostics](../user-guide/diagnostics/mcmc-diagnostics.md) — PMCMC diagnostic guide
- [User Guide: Predictive Checks](../user-guide/diagnostics/predictive-checks.md) — Model adequacy
- [Tutorials: Model Comparison](../tutorials/model-comparison.md) — Full model-comparison tutorial
- [Theory: Convergence](../theory/convergence.md) — Convergence theory for PF and SMC
- [PMCMC API](pmcmc.md) — `MCMCChain` result type
- [SMC API](smc.md) — `SMCResult` result type
- [Visualization API](visualization.md) — Plotting helpers
