---
title: Degeneracy Diagnostic
description: "Particle degeneracy diagnostics: weight degeneracy, sample impoverishment, and path coalescence"
---

# Degeneracy Diagnostic

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `DegeneracyDiagnostic` |
    | **Import** | `from particlefilterbox.diagnostics import DegeneracyDiagnostic` |
    | **Input** | `FilterResult` or `SmootherResult` |
    | **Detects** | Weight degeneracy, sample impoverishment, path coalescence |
    | **Goal** | Identify and classify particle filter failure modes |

## Overview

**Degeneracy** is the fundamental failure mode of particle filters. It manifests in three distinct forms, each with different causes, detection methods, and remedies:

| Type | What happens | When it occurs | Key metric |
|------|-------------|----------------|------------|
| **Weight degeneracy** | One particle carries all the weight | Before resampling | $\text{ESS} \to 1$ |
| **Sample impoverishment** | All particles are copies of a few ancestors | After resampling | Unique particle count drops |
| **Path degeneracy** | All particle trajectories coalesce into one | Over time (smoothing) | Genealogical diversity collapses |

Understanding these three types is essential for diagnosing particle filter problems and choosing the right remedy.

---

## Basic Usage

```python
from particlefilterbox import BootstrapPF, PFConfig
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.diagnostics import DegeneracyDiagnostic
import numpy as np

# Run a filter
model = StochasticVolatility(variant="basic")
config = PFConfig(n_particles=1000, ess_threshold=0.5, seed=42)
pf = BootstrapPF(model, config)

rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=300, rng=rng)
result = pf.filter(obs)

# Create diagnostic
diag = DegeneracyDiagnostic(result)

# Full report
print(diag.degeneracy_report())
```

```text
=== Degeneracy Report ===

Weight Degeneracy:
  Status:           HEALTHY
  Mean ESS ratio:   0.692
  Min ESS ratio:    0.156
  ESS < 0.05 rate:  0.003

Sample Impoverishment:
  Status:           HEALTHY
  Mean unique ratio: 0.847
  Min unique ratio:  0.612
  Impoverished steps: 0/300 (0.0%)

Path Degeneracy:
  Status:           MILD
  Coalescence lag:  ~45 steps
  Effective paths at lag=10: 312.4
  Effective paths at lag=50: 23.1

Overall Verdict: HEALTHY (path degeneracy is mild and expected)
```

---

## Type 1: Weight Degeneracy

### What It Is

Weight degeneracy occurs when the importance weights become highly unequal --- a single particle (or very few particles) carries almost all the weight. The ESS approaches 1.

$$
\text{Weight degeneracy: } \text{ESS}_t \to 1 \iff \exists\, i : w_t^{(i)} \to 1
$$

### Why It Happens

The importance weights grow multiplicatively over time:

$$
w_t^{(i)} \propto \prod_{s=1}^{t} \frac{p(y_s \mid x_s^{(i)}) \, p(x_s^{(i)} \mid x_{s-1}^{(i)})}{q(x_s^{(i)} \mid x_{s-1}^{(i)}, y_s)}
$$

Without resampling, the variance of this product grows **exponentially** with $t$, guaranteeing that eventually one weight dominates.

### Detection

```python
# Weight degeneracy analysis
weight_report = diag.weight_degeneracy_report()

print(f"ESS < 0.05N rate: {weight_report['critical_rate']:.3f}")
print(f"Worst ESS: {weight_report['min_ess']:.1f} at t={weight_report['min_ess_time']}")
print(f"Near-degenerate steps: {weight_report['degenerate_steps']}")
```

### Remedies

!!! tip "Fixing weight degeneracy"
    Weight degeneracy is the **easiest** type to fix:
    
    1. **Resample more frequently** --- lower the ESS threshold ($\alpha$)
    2. **Use a better proposal** --- SIR, Guided, or Locally Optimal PF
    3. **Increase N** --- more particles provide better coverage
    4. **Temper the likelihood** --- spread the weight update across sub-steps

---

## Type 2: Sample Impoverishment

### What It Is

Sample impoverishment is the **flip side of resampling**. After resampling, particles with high weights are duplicated and particles with low weights are eliminated. If this happens too aggressively or too often, the particle population loses diversity:

$$
\text{Sample impoverishment: } \frac{|\{x_t^{(i)} : \text{unique}\}|}{N} \ll 1
$$

### Why It Happens

Resampling is a **variance reduction** technique that introduces **bias** in exchange. Each resampling step reduces the number of unique particles. In the worst case, after aggressive resampling at multiple consecutive time steps, the entire particle cloud may consist of copies of a single ancestor.

### Detection

```python
# Unique particles over time
diag.unique_particles_over_time()

# Customized
diag.unique_particles_over_time(
    show_resampling=True,   # mark resampling events
    show_threshold=0.5,     # warning line at 50% unique
    figsize=(14, 5),
)
```

The plot shows:

- **Blue line**: Fraction of unique particles at each time step
- **Red markers**: Time steps where resampling occurred (unique count drops)
- **Green markers**: Time steps with no resampling (unique count = 1.0)

### Interpreting the Plot

| Unique Ratio | Interpretation |
|-------------|---------------|
| $> 0.8$ | Excellent particle diversity |
| $0.5\text{--}0.8$ | Good --- normal after resampling |
| $0.2\text{--}0.5$ | Moderate impoverishment |
| $< 0.2$ | Severe --- particles are heavily duplicated |
| $< 0.05$ | Critical --- effectively a few particles |

### Remedies

!!! tip "Fixing sample impoverishment"
    Sample impoverishment requires fundamentally different remedies than weight degeneracy:
    
    1. **Resample less frequently** --- raise the ESS threshold ($\alpha$)
    2. **Use regularized resampling** --- the Regularized PF adds a kernel move after resampling to diversify particles
    3. **Use MCMC moves** --- Particle Gibbs and resample-move algorithms rejuvenate particles
    4. **Stratified/systematic resampling** --- produces less variance than multinomial resampling
    5. **Increase N** --- more particles means more unique survivors

---

## Type 3: Path Degeneracy

### What It Is

Path degeneracy is specific to **smoothing** applications where you need the entire trajectory $x_{0:t}$, not just the current state $x_t$. Even when the filtering distribution is well-approximated, the particle **genealogies** (ancestry trees) coalesce: looking back in time, all current particles share a single ancestor.

$$
\text{Path degeneracy: } |\{x_{t-L}^{(a_t^{(i)})} : i = 1, \ldots, N\}| \to 1 \text{ as } L \to \infty
$$

### Why It Happens

Each resampling step eliminates some ancestral lineages. Over time, the surviving lineages merge into a single common ancestor. The **coalescence rate** depends on the ESS and the resampling scheme.

For a standard particle filter with $N$ particles, the expected coalescence time is $O(N)$ steps. This means that for smoothing at lag $L$, you need $N \gg L$ to maintain trajectory diversity.

### Detection

```python
# Path coalescence analysis
diag.path_coalescence(lag=10)

# Multiple lags
for lag in [5, 10, 20, 50, 100]:
    effective_paths = diag.effective_paths(lag=lag)
    print(f"Lag {lag:3d}: {effective_paths:.1f} effective paths "
          f"({effective_paths/1000:.1%} of N)")
```

```text
Lag   5: 687.3 effective paths (68.7% of N)
Lag  10: 312.4 effective paths (31.2% of N)
Lag  20:  98.7 effective paths (9.9% of N)
Lag  50:  23.1 effective paths (2.3% of N)
Lag 100:   4.2 effective paths (0.4% of N)
```

### Coalescence Plot

```python
# Effective paths as a function of lag
diag.plot_coalescence(
    max_lag=100,
    show_theoretical=True,  # theoretical coalescence rate
    figsize=(10, 6),
)
```

### Interpreting Coalescence

| Effective Paths at Lag $L$ | Interpretation |
|---------------------------|---------------|
| $> 0.5N$ | Good trajectory diversity |
| $0.1N\text{--}0.5N$ | Moderate --- smoothing estimates have variance |
| $0.01N\text{--}0.1N$ | Poor --- smoothing estimates are unreliable |
| $< 0.01N$ | Degenerate --- all trajectories have coalesced |

!!! warning "Path degeneracy is unavoidable in standard particle filters"
    No amount of increasing $N$ can eliminate path degeneracy for fixed-lag smoothing with large lags. The coalescence time is $O(N)$, so doubling $N$ only doubles the lag at which coalescence occurs. For smoothing applications, you need **fundamentally different algorithms**:
    
    - **Fixed-lag smoother** with lag $\ll N$
    - **FFBSm/FFBSi** (backward smoothers) that reconstruct trajectories
    - **PG-AS** (Particle Gibbs with Ancestor Sampling) that maintains path diversity
    - **Two-filter smoother** that avoids forward-only genealogies

### Remedies

!!! tip "Fixing path degeneracy"
    Path degeneracy is the **hardest** type to fix within the standard particle filter framework:
    
    1. **Use a dedicated smoother** --- FFBSm, FFBSi, or Two-Filter smoother
    2. **PG-AS** --- ancestor sampling maintains path diversity across MCMC iterations
    3. **Fixed-lag approximation** --- only smooth up to a manageable lag
    4. **Increase N dramatically** --- coalescence time is $O(N)$, so $N$ must be much larger than the smoothing lag
    5. **Waste-Free SMC** --- reduces the rate of genealogical depletion

---

## Complete Example: Three-Type Analysis

```python
import numpy as np
from particlefilterbox import BootstrapPF, PFConfig
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.diagnostics import DegeneracyDiagnostic

# Setup with deliberately small N to illustrate degeneracy
model = StochasticVolatility(variant="basic")
config = PFConfig(n_particles=500, ess_threshold=0.5, seed=42)
pf = BootstrapPF(model, config)

rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=200, rng=rng)
result = pf.filter(obs)

# Full degeneracy analysis
diag = DegeneracyDiagnostic(result)

# 1. Full report
print(diag.degeneracy_report())

# 2. Weight degeneracy (before resampling)
weight_report = diag.weight_degeneracy_report()
print(f"\nWeight degeneracy rate: {weight_report['critical_rate']:.3f}")

# 3. Sample impoverishment (after resampling)
diag.unique_particles_over_time(show_resampling=True)

# 4. Path degeneracy (genealogical diversity)
diag.plot_coalescence(max_lag=100, show_theoretical=True)

# 5. Summary: which type is the bottleneck?
bottleneck = diag.identify_bottleneck()
print(f"\nPrimary bottleneck: {bottleneck['type']}")
print(f"Recommendation: {bottleneck['recommendation']}")
```

```text
Primary bottleneck: path_degeneracy
Recommendation: For smoothing at lag > 20, switch to FFBSm or PG-AS.
                For filtering only, current setup is adequate.
```

---

## Comparison of the Three Types

| Aspect | Weight Degeneracy | Sample Impoverishment | Path Degeneracy |
|--------|------------------|-----------------------|-----------------|
| **When** | Before resampling | After resampling | Over time |
| **Metric** | ESS | Unique particle count | Effective paths |
| **Cause** | Poor proposal | Aggressive resampling | Genealogical coalescence |
| **Affects** | Current-time estimates | Current-time diversity | Historical trajectory estimates |
| **Fix** | Better proposal, tempering | Regularization, MCMC moves | Dedicated smoothers, PG-AS |
| **Scales with N** | Linear improvement | Linear improvement | Only shifts coalescence time |
| **Severity** | Easy to detect and fix | Moderate | Fundamental limitation |

---

## API Summary

| Method | Description |
|--------|-------------|
| `DegeneracyDiagnostic(result)` | Create diagnostic from filter/smoother result |
| `.degeneracy_report()` | Full report covering all three types |
| `.weight_degeneracy_report()` | Detailed weight degeneracy analysis |
| `.unique_particles_over_time(**kwargs)` | Plot unique particle fraction over time |
| `.path_coalescence(lag)` | Analyze genealogical coalescence at a given lag |
| `.effective_paths(lag)` | Number of effective ancestral paths at lag $L$ |
| `.plot_coalescence(max_lag, **kwargs)` | Effective paths vs. lag plot |
| `.identify_bottleneck()` | Identify the primary degeneracy bottleneck |

---

## See Also

- [ESS Diagnostic](ess-diagnostic.md) --- detailed ESS analysis (weight degeneracy detection)
- [Weight Diagnostic](weight-diagnostic.md) --- weight distribution analysis
- [Convergence Diagnostic](convergence.md) --- how $N$ affects degeneracy
- [Core: Resampling](../user-guide/core/resampling.md) --- resampling algorithms and their trade-offs
- [Smoothers](../user-guide/smoothers/index.md) --- algorithms that address path degeneracy
- [PG-AS](../user-guide/pmcmc/pgas.md) --- ancestor sampling for path diversity
- [Regularized PF](../user-guide/filters/regularized.md) --- kernel smoothing to combat sample impoverishment
- [Ensemble PF](../user-guide/filters/ensemble.md) --- handles high-dimensional states where degeneracy is worst
- [Theory: Convergence](../theory/convergence-theory.md) --- theoretical bounds on degeneracy rates
- [Acceleration: GPU](../acceleration/gpu.md) --- scale up $N$ to reduce degeneracy when model complexity demands it
