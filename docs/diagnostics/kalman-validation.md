---
title: Kalman Validation
description: "Cross-validation of particle filters against the exact Kalman filter using kalmanbox for linear-Gaussian models"
---

# Kalman Validation

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `KalmanValidation` |
    | **Import** | `from particlefilterbox.diagnostics import KalmanValidation` |
    | **Input** | Linear-Gaussian model, particle filter, and `kalmanbox.KalmanFilter` |
    | **Key method** | `.run(observations)` then `.passes(tolerance)` |
    | **Goal** | Verify that PF converges to exact Kalman solution on linear models |
    | **Dependency** | [kalmanbox](https://pypi.org/project/kalmanbox/) |

## Overview

For **linear-Gaussian** state-space models, the Kalman filter provides the **exact** filtering distribution. This is the gold standard --- no approximation, no Monte Carlo error. The particle filter, by contrast, is an approximation that should converge to the Kalman solution as $N \to \infty$.

This creates a powerful validation strategy:

$$
\text{If } \hat{\pi}_t^{N}(f) \xrightarrow{N \to \infty} \pi_t(f) = \text{Kalman solution}
$$

then the particle filter implementation is correct.

The `KalmanValidation` diagnostic automates this cross-check by running both filters on the same data and comparing their outputs.

!!! tip "When to use Kalman validation"
    - **After implementing a new filter**: Verify correctness on a linear test case before applying to nonlinear models.
    - **After refactoring**: Ensure that code changes haven't introduced bugs.
    - **Debugging unexpected results**: If a filter behaves oddly on a nonlinear model, first check if it works correctly on a linear one.
    - **Teaching and learning**: Build intuition for how particle filters approximate exact solutions.

---

## Basic Usage

```python
from kalmanbox import KalmanFilter
from particlefilterbox import BootstrapPF, PFConfig
from particlefilterbox.models import LinearStateSpace
from particlefilterbox.diagnostics import KalmanValidation
import numpy as np

# Define a linear-Gaussian state-space model
#   x_t = F x_{t-1} + G w_t,   w_t ~ N(0, Q)
#   y_t = H x_t + v_t,          v_t ~ N(0, R)
linear_model = LinearStateSpace(
    F=np.array([[0.95]]),       # state transition
    G=np.array([[1.0]]),        # noise input
    H=np.array([[1.0]]),        # observation matrix
    Q=np.array([[0.1]]),        # state noise covariance
    R=np.array([[1.0]]),        # observation noise covariance
    x0_mean=np.array([0.0]),    # prior mean
    x0_cov=np.array([[1.0]]),   # prior covariance
)

# Simulate data
rng = np.random.default_rng(42)
states, obs = linear_model.simulate(n_obs=200, rng=rng)

# Create particle filter
config = PFConfig(n_particles=5000, seed=42)
pf = BootstrapPF(linear_model, config)

# Create Kalman filter
kf = KalmanFilter()

# Run validation
val = KalmanValidation(
    linear_model=linear_model,
    particle_filter=pf,
    kalman_filter=kf,
)
val.run(obs)

# Quick pass/fail check
print(val.passes(tolerance=0.05))  # True
```

```text
True
```

---

## Comparing Filtered Means

The most direct comparison: does the PF posterior mean match the Kalman posterior mean?

```python
# Compare means
mean_report = val.compare_means()
print(mean_report)
```

```text
=== Filtered Mean Comparison ===
Time steps:        200
State dimension:   1

Max absolute error:   0.0312
Mean absolute error:  0.0089
Relative error (L2):  0.0041

Worst time step: t=87 (PF=0.842, Kalman=0.873, |diff|=0.031)

Verdict: PASS (max |error| < 0.05)
```

### Error Over Time

```python
# Plot PF mean vs Kalman mean
val.plot_means(
    figsize=(14, 6),
    show_error=True,    # bottom subplot with absolute error
)
```

The plot shows:

- **Top panel**: PF filtered mean (blue) overlaid on Kalman mean (orange dashed)
- **Bottom panel**: Absolute error $|\hat{x}_t^{\text{PF}} - \hat{x}_t^{\text{KF}}|$ over time
- **Shaded region**: Tolerance band

!!! note "Expected error magnitude"
    For a well-implemented Bootstrap PF with $N$ particles, the mean absolute error scales as:

    $$
    \mathbb{E}\!\left[|\hat{x}_t^{\text{PF}} - \hat{x}_t^{\text{KF}}|\right] = O\!\left(\frac{1}{\sqrt{N}}\right)
    $$

    At $N = 5000$, typical errors are $\approx 0.01\text{--}0.05$, depending on the model.

---

## Comparing Filtered Variances

The posterior variance (uncertainty) is harder to match than the mean. This is a stricter test of implementation correctness.

```python
# Compare variances
var_report = val.compare_variances()
print(var_report)
```

```text
=== Filtered Variance Comparison ===
Time steps:        200
State dimension:   1

Max absolute error:   0.0087
Mean absolute error:  0.0024
Relative error (L2):  0.0031

PF mean variance:     0.0912
Kalman variance:      0.0909

Verdict: PASS (relative error < 0.05)
```

```python
# Plot variance comparison
val.plot_variances(figsize=(14, 6))
```

!!! warning "Variance estimation in particle filters"
    The particle filter variance estimate includes **both** posterior uncertainty and Monte Carlo noise. As $N \to \infty$, the Monte Carlo component vanishes and the PF variance converges to the Kalman variance. At finite $N$, the PF variance is typically slightly **larger** than the Kalman variance --- this is expected and not a sign of a bug.

---

## Convergence Plot

The most informative diagnostic: how does the PF error decrease as $N$ grows?

```python
# Convergence study: PF -> Kalman as N -> infinity
val.convergence_plot(
    n_values=[100, 250, 500, 1000, 2000, 5000, 10000],
    n_runs=10,          # runs per N for confidence intervals
    metric="mean",      # or "variance", "log_likelihood"
    figsize=(12, 6),
)
```

The convergence plot shows:

- **x-axis**: Number of particles $N$ (log scale)
- **y-axis**: Mean absolute error vs. Kalman (log scale)
- **Points**: Mean error across runs, with error bars (std across runs)
- **Dashed line**: Theoretical $O(1/\sqrt{N})$ rate

```text
=== Convergence Study ===
       N |  MAE (mean) |  MAE (std) |  Rate vs prev
---------+-------------+------------+--------------
     100 |    0.0891   |   0.0124   |     ---
     250 |    0.0573   |   0.0078   |    -0.49
     500 |    0.0398   |   0.0051   |    -0.52
    1000 |    0.0284   |   0.0037   |    -0.49
    2000 |    0.0198   |   0.0024   |    -0.52
    5000 |    0.0126   |   0.0016   |    -0.50
   10000 |    0.0089   |   0.0011   |    -0.50

Empirical rate: O(N^{-0.50}) ✓ matches theory
```

!!! abstract "Key Takeaway"
    If the convergence rate is close to $O(N^{-0.5})$, the implementation is correct. Deviations indicate bugs:

    - **Rate much slower than $O(N^{-0.5})$**: Likely a resampling or weight computation error.
    - **Error plateaus (does not decrease with $N$)**: Systematic bias --- check state transition or observation model implementation.
    - **Error increases with $N$**: Something is fundamentally wrong --- review the entire filter loop.

---

## Detailed Metrics

### KL Divergence

The Kullback-Leibler divergence between the PF approximation and the Kalman posterior (both Gaussian in the linear case):

$$
D_{\text{KL}}\!\left(\mathcal{N}(\hat{\mu}_t^{\text{PF}}, \hat{\Sigma}_t^{\text{PF}}) \,\|\, \mathcal{N}(\mu_t^{\text{KF}}, \Sigma_t^{\text{KF}})\right)
$$

For univariate states, this simplifies to:

$$
D_{\text{KL}} = \log\frac{\sigma_{\text{KF}}}{\sigma_{\text{PF}}} + \frac{\sigma_{\text{PF}}^2 + (\mu_{\text{PF}} - \mu_{\text{KF}})^2}{2\sigma_{\text{KF}}^2} - \frac{1}{2}
$$

```python
# KL divergence over time
kl = val.kl_divergence()
print(f"Mean KL: {kl.mean():.6f}")
print(f"Max KL:  {kl.max():.6f} at t={kl.argmax()}")
```

### Log-Likelihood Comparison

```python
# Compare log-likelihood estimates
ll_report = val.compare_log_likelihood()
print(ll_report)
```

```text
=== Log-Likelihood Comparison ===
PF estimate:     -284.52 (std across runs: 0.087)
Kalman exact:    -284.49

Absolute error:   0.03
Relative error:   0.01%

Verdict: PASS
```

---

## Automated Pass/Fail

The `passes()` method provides a single True/False verdict:

```python
# Default tolerance: 0.05
val.passes(tolerance=0.05)

# Strict tolerance for high-N validation
val.passes(tolerance=0.01)

# Detailed report
val.passes(tolerance=0.05, verbose=True)
```

```text
=== Validation Report (tolerance=0.05) ===
✓ Mean absolute error:    0.0089 < 0.05
✓ Variance relative error: 0.0031 < 0.05
✓ Max KL divergence:      0.0042 < 0.05
✓ Log-likelihood error:   0.0001 < 0.05

Overall: PASS (4/4 checks passed)
```

!!! tip "Choosing the tolerance"
    The appropriate tolerance depends on $N$:

    | $N$ | Expected MAE | Suggested tolerance |
    |-----|-------------|-------------------|
    | 500 | $\approx 0.04$ | 0.10 |
    | 1000 | $\approx 0.03$ | 0.07 |
    | 2000 | $\approx 0.02$ | 0.05 |
    | 5000 | $\approx 0.01$ | 0.03 |
    | 10000 | $\approx 0.007$ | 0.02 |

---

## Example: Validate Bootstrap PF

```python
import numpy as np
from kalmanbox import KalmanFilter
from particlefilterbox import BootstrapPF, PFConfig
from particlefilterbox.models import LinearStateSpace
from particlefilterbox.diagnostics import KalmanValidation

# --- 1. Linear model ---
model = LinearStateSpace(
    F=np.array([[0.95]]),
    G=np.array([[1.0]]),
    H=np.array([[1.0]]),
    Q=np.array([[0.1]]),
    R=np.array([[1.0]]),
    x0_mean=np.array([0.0]),
    x0_cov=np.array([[1.0]]),
)

# --- 2. Simulate data ---
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=200, rng=rng)

# --- 3. Configure filters ---
config = PFConfig(n_particles=5000, seed=42)
pf = BootstrapPF(model, config)
kf = KalmanFilter()

# --- 4. Run validation ---
val = KalmanValidation(
    linear_model=model,
    particle_filter=pf,
    kalman_filter=kf,
)
val.run(obs)

# --- 5. Compare ---
val.compare_means()
val.compare_variances()

# --- 6. Convergence plot ---
val.convergence_plot(
    n_values=[100, 500, 1000, 5000, 10000],
    n_runs=10,
)

# --- 7. Pass/fail ---
assert val.passes(tolerance=0.05), "Bootstrap PF failed Kalman validation!"
print("Bootstrap PF: PASSED Kalman validation")
```

---

## Example: Validate RBPF (Linear Component)

The Rao-Blackwellized Particle Filter (RBPF) analytically marginalizes the **linear** component of the state. For a fully linear model, RBPF should match the Kalman filter **exactly** (up to numerical precision), regardless of $N$:

```python
import numpy as np
from kalmanbox import KalmanFilter
from particlefilterbox import RBPF, PFConfig
from particlefilterbox.models import LinearStateSpace
from particlefilterbox.diagnostics import KalmanValidation

# Linear model (same as above)
model = LinearStateSpace(
    F=np.array([[0.95]]),
    G=np.array([[1.0]]),
    H=np.array([[1.0]]),
    Q=np.array([[0.1]]),
    R=np.array([[1.0]]),
    x0_mean=np.array([0.0]),
    x0_cov=np.array([[1.0]]),
)

rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=200, rng=rng)

# RBPF with very few particles --- should still match Kalman
config = PFConfig(n_particles=100, seed=42)
rbpf = RBPF(model, config)
kf = KalmanFilter()

val = KalmanValidation(
    linear_model=model,
    particle_filter=rbpf,
    kalman_filter=kf,
)
val.run(obs)

# Compare means --- should be near-exact
mean_report = val.compare_means()
print(mean_report)

# For a fully linear model, RBPF should pass with very tight tolerance
assert val.passes(tolerance=0.001), "RBPF should match Kalman almost exactly!"
print("RBPF: PASSED with tight tolerance (0.001)")
```

```text
=== Filtered Mean Comparison ===
Time steps:        200
State dimension:   1

Max absolute error:   0.0002
Mean absolute error:  0.0001
Relative error (L2):  0.0000

Verdict: PASS (max |error| < 0.001)
```

!!! note "Why RBPF is special"
    For a fully linear model, the RBPF reduces to the Kalman filter itself --- the "nonlinear" part of the state is empty, so there is nothing to sample. The linear component is handled analytically. Errors are limited to floating-point precision ($\approx 10^{-15}$ in theory, $\approx 10^{-4}$ in practice due to resampling numerics).

    This makes RBPF validation a particularly powerful implementation test: if RBPF doesn't match Kalman on a linear model, the Rao-Blackwellization logic has a bug.

---

## Multivariate Models

Kalman validation extends naturally to multivariate state spaces:

```python
# 2D state-space model
model_2d = LinearStateSpace(
    F=np.array([[0.9, 0.1],
                [0.0, 0.95]]),
    G=np.eye(2),
    H=np.array([[1.0, 0.0]]),    # observe only first state
    Q=np.array([[0.1, 0.0],
                [0.0, 0.2]]),
    R=np.array([[1.0]]),
    x0_mean=np.zeros(2),
    x0_cov=np.eye(2),
)

rng = np.random.default_rng(42)
states, obs = model_2d.simulate(n_obs=200, rng=rng)

config = PFConfig(n_particles=10000, seed=42)
pf = BootstrapPF(model_2d, config)
kf = KalmanFilter()

val = KalmanValidation(
    linear_model=model_2d,
    particle_filter=pf,
    kalman_filter=kf,
)
val.run(obs)

# Per-dimension comparison
val.compare_means()
```

```text
=== Filtered Mean Comparison ===
Time steps:        200
State dimension:   2

Dimension 0 (observed):
  Max absolute error:   0.0284
  Mean absolute error:  0.0078

Dimension 1 (latent):
  Max absolute error:   0.0412
  Mean absolute error:  0.0125

Overall relative error (L2): 0.0098
Verdict: PASS
```

!!! warning "Curse of dimensionality"
    For higher-dimensional state spaces, particle filters need exponentially more particles to maintain accuracy. A PF that matches Kalman in 1D may fail in 10D. If validation fails in high dimensions, this is a fundamental limitation of the filter, not a bug --- consider RBPF or Ensemble PF for high-dimensional problems.

---

## Integration in CI/Testing

Use Kalman validation as an automated regression test:

```python
# tests/test_kalman_validation.py
import numpy as np
import pytest
from kalmanbox import KalmanFilter
from particlefilterbox import BootstrapPF, SIRPF, AuxiliaryPF, RBPF, PFConfig
from particlefilterbox.models import LinearStateSpace
from particlefilterbox.diagnostics import KalmanValidation


@pytest.fixture
def linear_setup():
    model = LinearStateSpace(
        F=np.array([[0.95]]),
        G=np.array([[1.0]]),
        H=np.array([[1.0]]),
        Q=np.array([[0.1]]),
        R=np.array([[1.0]]),
        x0_mean=np.array([0.0]),
        x0_cov=np.array([[1.0]]),
    )
    rng = np.random.default_rng(42)
    _, obs = model.simulate(n_obs=100, rng=rng)
    return model, obs


@pytest.mark.parametrize("FilterClass,n_particles,tol", [
    (BootstrapPF, 5000, 0.05),
    (SIRPF, 5000, 0.05),
    (AuxiliaryPF, 5000, 0.05),
    (RBPF, 100, 0.001),
])
def test_kalman_validation(linear_setup, FilterClass, n_particles, tol):
    model, obs = linear_setup
    config = PFConfig(n_particles=n_particles, seed=42)
    pf = FilterClass(model, config)
    kf = KalmanFilter()

    val = KalmanValidation(
        linear_model=model,
        particle_filter=pf,
        kalman_filter=kf,
    )
    val.run(obs)
    assert val.passes(tolerance=tol)
```

!!! tip "CI-friendly settings"
    For continuous integration, use moderate particle counts ($N = 2000\text{--}5000$) and generous tolerances ($0.05\text{--}0.10$) to avoid flaky tests while still catching real bugs. Reserve tight tolerances for nightly or release testing.

---

## API Summary

| Method | Description |
|--------|-------------|
| `KalmanValidation(linear_model, particle_filter, kalman_filter)` | Create validation object |
| `.run(observations)` | Run both filters on the observations |
| `.compare_means()` | Compare PF filtered mean vs Kalman mean |
| `.compare_variances()` | Compare PF variance vs Kalman variance |
| `.compare_log_likelihood()` | Compare log-likelihood estimates |
| `.kl_divergence()` | KL divergence at each time step |
| `.convergence_plot(n_values, n_runs, **kwargs)` | PF error vs $N$ convergence study |
| `.plot_means(**kwargs)` | Plot mean comparison with error band |
| `.plot_variances(**kwargs)` | Plot variance comparison |
| `.passes(tolerance, verbose)` | Automated pass/fail check |

---

## See Also

- [Convergence Diagnostic](convergence.md) --- general convergence analysis (not limited to linear models)
- [Filter Comparison](filter-comparison.md) --- compare filter accuracy on any model
- [Theory: Convergence](../theory/convergence-theory.md) --- CLT and convergence rates for particle filters
- [Theory: Particle Filters](../theory/particle-filter-theory.md) --- optimal proposals and weight computation
- [RBPF User Guide](../user-guide/filters/rbpf.md) --- Rao-Blackwellized PF details
- [Bootstrap PF](../user-guide/filters/bootstrap.md) --- the most common filter to validate
- [Filters Overview](../user-guide/filters/index.md) --- all available filter variants
