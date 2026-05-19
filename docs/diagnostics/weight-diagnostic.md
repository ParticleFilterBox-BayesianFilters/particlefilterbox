---
title: Weight Diagnostic
description: "Particle weight diagnostics: distribution analysis, concentration, entropy, and log-weight stability"
---

# Weight Diagnostic

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `WeightDiagnostic` |
    | **Import** | `from particlefilterbox.diagnostics import WeightDiagnostic` |
    | **Input** | `FilterResult` from any particle filter |
    | **Key metrics** | Max weight, entropy, concentration ratio |
    | **Healthy range** | Max weight $< 0.1$, entropy $> 0.8 \cdot \log N$ |

## Overview

While the [ESS diagnostic](ess-diagnostic.md) provides a scalar summary of weight quality, the Weight Diagnostic examines the **full distribution of particle weights** at each time step. This reveals problems that ESS alone can miss:

- A few particles dominating the approximation (high max weight)
- Systematic log-weight drift leading to numerical overflow or underflow
- Weight entropy that degrades faster than ESS suggests

The weight distribution is the most direct indicator of how well the proposal distribution matches the filtering distribution.

---

## Basic Usage

```python
from particlefilterbox import BootstrapPF, PFConfig
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.diagnostics import WeightDiagnostic
import numpy as np

# Run a filter
model = StochasticVolatility(variant="basic")
config = PFConfig(n_particles=2000, ess_threshold=0.5, seed=42)
pf = BootstrapPF(model, config)

rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=300, rng=rng)
result = pf.filter(obs)

# Create diagnostic
diag = WeightDiagnostic(result)

# Summary
print(diag.summary())
```

```text
=== Weight Diagnostic Summary ===
Particles (N):     2000
Time steps (T):    300

Weight Statistics (averaged over time):
  Max weight:      0.034  (ideal: 1/N = 0.0005)
  Weight entropy:  7.21   (max: ln(2000) = 7.60)
  Entropy ratio:   0.949
  Gini coefficient: 0.412

Concentration:
  50% of mass in:  312 particles (15.6% of N)
  90% of mass in:  1124 particles (56.2% of N)
  99% of mass in:  1876 particles (93.8% of N)

Log-weight Stability:
  Max log-weight range: 42.3
  Underflow events:     0
  Overflow events:      0

Verdict: HEALTHY
```

---

## Weight Histogram

The weight histogram at a specific time step reveals how concentrated the particle approximation is:

```python
# Histogram at a specific time step
diag.weight_histogram(t=50)

# Histogram at the worst time step (lowest ESS)
diag.weight_histogram(t="worst")

# Customized
diag.weight_histogram(
    t=50,
    bins=50,
    log_scale=True,     # log-scale x-axis (recommended)
    show_uniform=True,  # reference line for uniform weights
    figsize=(10, 5),
)
```

### Interpreting the Histogram

| Shape | Interpretation |
|-------|---------------|
| Concentrated near $1/N$ | Excellent --- weights are nearly uniform |
| Right-skewed with thin tail | Good --- moderate weight variation |
| Heavy right tail | Concerning --- a few particles dominate |
| One spike + mass at zero | Degenerate --- one particle carries most weight |

!!! tip "Use log-scale"
    Always use `log_scale=True` for weight histograms. The natural scale compresses most weights near zero, hiding the structure. The log-scale reveals the full distribution shape.

---

## Weight Concentration

The **concentration curve** shows how many particles are needed to capture a given fraction of the total weight:

$$
C(\alpha) = \min \left\{ k : \sum_{i=1}^{k} w_{(i)} \geq \alpha \right\}
$$

where $w_{(1)} \geq w_{(2)} \geq \cdots \geq w_{(N)}$ are the sorted weights.

```python
# Concentration curve
diag.concentration_curve()

# At a specific time step
diag.concentration_curve(t=100)

# Numerical: how many particles carry 50%, 90%, 99% of the weight?
for alpha in [0.5, 0.9, 0.99]:
    k = diag.concentration_count(alpha=alpha, t=100)
    print(f"{alpha:.0%} of weight in {k} particles ({k/2000:.1%} of N)")
```

```text
50% of weight in 287 particles (14.4% of N)
90% of weight in 1098 particles (54.9% of N)
99% of weight in 1834 particles (91.7% of N)
```

### Interpreting Concentration

| Concentration | Quality |
|---------------|---------|
| 50% of weight in $> 25\%$ of particles | Excellent diversity |
| 50% of weight in $10\text{--}25\%$ of particles | Good |
| 50% of weight in $1\text{--}10\%$ of particles | Moderate --- monitor closely |
| 50% of weight in $< 1\%$ of particles | Severe concentration --- action needed |

---

## Weight Entropy

The **Shannon entropy** of the normalized weights measures how evenly distributed they are:

$$
H_t = -\sum_{i=1}^{N} w_t^{(i)} \log w_t^{(i)}
$$

The maximum entropy is $\log N$ (uniform weights). The **entropy ratio** $H_t / \log N$ gives a normalized measure in $[0, 1]$.

```python
# Entropy over time
diag.plot_entropy()

# Entropy ratio (normalized to [0, 1])
diag.plot_entropy(normalized=True)
```

| Entropy Ratio | Interpretation |
|---------------|---------------|
| $> 0.95$ | Weights are nearly uniform |
| $0.80\text{--}0.95$ | Healthy weight distribution |
| $0.60\text{--}0.80$ | Moderate concentration |
| $< 0.60$ | Severe concentration |

!!! note "Entropy vs ESS"
    Entropy and ESS are related but not identical. ESS is dominated by the largest weights (it depends on $\sum w_i^2$), while entropy is more sensitive to the overall distribution shape. A distribution with a few very large weights and many small ones can have moderate entropy but very low ESS. Use both for a complete picture.

---

## Max Weight Over Time

The **maximum weight** at each time step is the simplest indicator of weight concentration:

$$
w_{\max}(t) = \max_{i=1,\ldots,N} w_t^{(i)}
$$

```python
# Max weight time series
diag.max_weight_over_time()

# With customization
diag.max_weight_over_time(
    show_threshold=0.1,  # horizontal warning line
    figsize=(14, 4),
)
```

### Interpreting Max Weight

| Max Weight | Quality |
|-----------|---------|
| $w_{\max} \approx 1/N$ | Perfect --- all weights equal |
| $w_{\max} < 0.05$ | Healthy |
| $w_{\max} \in [0.05, 0.20]$ | Moderate concentration |
| $w_{\max} > 0.20$ | One particle dominates --- poor approximation |
| $w_{\max} \to 1.0$ | Complete degeneracy |

---

## Log-Weight Stability

Particle filters work internally with **log-weights** to avoid numerical overflow. However, even in log-space, extreme values can cause problems:

- **Underflow**: All log-weights are so negative that $\exp(w)$ rounds to zero
- **Overflow**: Log-weight differences are so large that normalization fails
- **Range explosion**: The gap between max and min log-weights grows without bound

```python
# Log-weight range over time
diag.plot_log_weight_range()

# Check for numerical issues
stability = diag.log_weight_stability()
print(f"Max log-weight range: {stability['max_range']:.1f}")
print(f"Underflow events:     {stability['underflow_count']}")
print(f"Overflow events:      {stability['overflow_count']}")
```

!!! warning "Log-weight range > 50"
    If the log-weight range exceeds approximately 50, you are at risk of numerical issues even with log-space computation. This typically indicates that the observation model assigns near-zero likelihood to most particles. Consider:
    
    1. **Tempering** the likelihood
    2. **Increasing** the observation noise (if uncertain)
    3. Using a **heavier-tailed** observation model

---

## Complete Example

```python
import numpy as np
from particlefilterbox import BootstrapPF, PFConfig
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.diagnostics import WeightDiagnostic

# Setup
model = StochasticVolatility(variant="leverage")
config = PFConfig(n_particles=3000, ess_threshold=0.5, seed=42)
pf = BootstrapPF(model, config)

rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=500, rng=rng)
result = pf.filter(obs)

# Full weight diagnostic
diag = WeightDiagnostic(result)

# 1. Overview
print(diag.summary())

# 2. Weight histogram at the worst time step
diag.weight_histogram(t="worst", log_scale=True)

# 3. Concentration curve (averaged over time)
diag.concentration_curve()

# 4. Max weight and entropy over time
diag.max_weight_over_time()
diag.plot_entropy(normalized=True)

# 5. Log-weight stability check
stability = diag.log_weight_stability()
if stability["overflow_count"] > 0 or stability["underflow_count"] > 0:
    print("WARNING: Numerical issues detected in log-weights")
```

---

## Corrective Actions

| Issue | Diagnostic Signal | Remedy |
|-------|------------------|--------|
| High max weight | $w_{\max} > 0.1$ consistently | Better proposal (SIR, Guided PF) |
| Low entropy | Entropy ratio $< 0.7$ | Increase $N$ or improve proposal |
| Extreme concentration | 50% of weight in $< 5\%$ of particles | Regularized PF or tempering |
| Log-weight instability | Range $> 50$ or underflow/overflow | Temper likelihood; heavier tails |
| Asymmetric histogram | Long right tail | The proposal is too diffuse; tighten it |

---

## API Summary

| Method | Description |
|--------|-------------|
| `WeightDiagnostic(result)` | Create diagnostic from a `FilterResult` |
| `.summary()` | Print comprehensive weight summary |
| `.weight_histogram(t, **kwargs)` | Weight histogram at time step $t$ |
| `.concentration_curve(t)` | Cumulative weight concentration curve |
| `.concentration_count(alpha, t)` | Particles needed for $\alpha$ fraction of weight |
| `.max_weight_over_time(**kwargs)` | Plot max weight time series |
| `.plot_entropy(normalized)` | Plot weight entropy over time |
| `.log_weight_stability()` | Check for numerical log-weight issues |
| `.plot_log_weight_range()` | Plot log-weight range over time |

---

## See Also

- [ESS Diagnostic](ess-diagnostic.md) --- scalar summary of weight quality
- [Degeneracy Diagnostic](degeneracy.md) --- consequences of persistent weight problems
- [Core: Resampling](../user-guide/core/resampling.md) --- how resampling resets the weights
- [Regularized PF](../user-guide/filters/regularized.md) --- a filter that maintains weight diversity
- [Guided PF](../user-guide/filters/guided.md) --- observation-driven proposal that reduces weight variance
- [Auxiliary PF](../user-guide/filters/auxiliary.md) --- look-ahead weighting for better particle efficiency
- [Theory: Convergence](../theory/convergence-theory.md) --- how weight variance relates to asymptotic error
- [Acceleration Overview](../acceleration/index.md) --- when weight diagnostics suggest you need more particles, acceleration helps
