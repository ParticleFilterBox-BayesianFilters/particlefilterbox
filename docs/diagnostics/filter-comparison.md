---
title: Filter Comparison
description: "Systematic comparison of particle filters: RMSE, log-likelihood, ESS, runtime, statistical tests, and automated ranking"
---

# Filter Comparison

!!! info "Quick Reference"
    | | |
    |---|---|
    | **Class** | `FilterComparison` |
    | **Import** | `from particlefilterbox.diagnostics import FilterComparison` |
    | **Input** | Model, list of filters, and observations |
    | **Key method** | `.run()` then `.summary_table()` |
    | **Goal** | Determine the best filter for a given model and dataset |

## Overview

Choosing the right particle filter is one of the most impactful decisions in Sequential Monte Carlo. Different filters offer different trade-offs between accuracy, computational cost, and robustness. The `FilterComparison` diagnostic provides a **systematic, reproducible** framework for comparing multiple filters on the same model and data.

Rather than eyeballing a few runs, this tool:

1. Runs each filter multiple times to account for Monte Carlo variability
2. Computes a comprehensive set of metrics (accuracy, efficiency, cost)
3. Performs statistical tests to determine if differences are significant
4. Produces publication-ready plots and tables
5. Generates an automated recommendation

---

## Basic Usage

```python
from particlefilterbox import BootstrapPF, SIRPF, AuxiliaryPF, GuidedPF, RBPF
from particlefilterbox import PFConfig
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.diagnostics import FilterComparison
import numpy as np

# Setup model and data
model = StochasticVolatility(variant="basic")
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=300, rng=rng)

# Configure filters
config = PFConfig(n_particles=2000, seed=42)
filters = [
    BootstrapPF(model, config),
    SIRPF(model, config),
    AuxiliaryPF(model, config),
    GuidedPF(model, config),
    RBPF(model, config),
]

# Run comparison
comp = FilterComparison(
    model=model,
    filters=filters,
    observations=obs,
    true_states=states,   # optional, for RMSE computation
    n_repeats=50,         # independent runs per filter
)
comp.run()

# View results
comp.summary_table()
```

```text
=== Filter Comparison Summary (50 repeats, N=2000, T=300) ===

Filter       |  RMSE (mean) |  Log-lik (mean) |  ESS ratio |  Runtime (s) | Memory (MB) | Rank
-------------+--------------+-----------------+------------+--------------+-------------+-----
Bootstrap    |     0.284    |    -412.32      |    0.692   |     0.82     |    48.2     |   4
SIR          |     0.231    |    -411.87      |    0.806   |     1.34     |    52.1     |   2
Auxiliary    |     0.218    |    -411.65      |    0.871   |     1.41     |    54.8     |   1
Guided       |     0.226    |    -411.78      |    0.834   |     1.89     |    58.3     |   3
RBPF         |     0.312    |    -412.81      |    0.654   |     0.71     |    42.6     |   5

Best filter: Auxiliary (lowest RMSE, highest log-likelihood, highest ESS)
```

---

## Comparison Metrics

### Accuracy Metrics

The comparison computes the following accuracy metrics for each filter:

#### RMSE (Root Mean Squared Error)

When true states are available (simulation studies), the RMSE measures the average distance between the filtered state estimate and the truth:

$$
\text{RMSE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \left(\hat{x}_t - x_t^{\text{true}}\right)^2}
$$

```python
# RMSE requires true states
comp = FilterComparison(
    model=model,
    filters=filters,
    observations=obs,
    true_states=states,  # pass simulated ground truth
    n_repeats=50,
)
```

!!! note "When true states are unavailable"
    For real data, true states are unknown. In this case, the comparison uses **inter-run variance** as a proxy for accuracy --- a filter whose estimates vary less across runs is more reliable.

#### Log-Likelihood

The estimated log-marginal-likelihood $\log \hat{p}(y_{1:T})$ is an unbiased measure of model fit (on the probability scale). Higher is better:

$$
\log \hat{p}(y_{1:T}) = \sum_{t=1}^{T} \log \left(\frac{1}{N} \sum_{i=1}^{N} w_t^{(i)}\right)
$$

The comparison reports both the mean and standard deviation of the log-likelihood across repeats. Low standard deviation indicates a reliable estimate.

### Efficiency Metrics

#### Mean ESS Ratio

The average $\text{ESS}_t / N$ across all time steps measures how efficiently the filter uses its particles:

$$
\overline{\text{ESS}} = \frac{1}{T} \sum_{t=1}^{T} \frac{\text{ESS}_t}{N}
$$

Higher is better --- it means less particle waste and more effective samples per step.

#### Runtime

Wall-clock time averaged across repeats, in seconds. Measures computational cost.

#### Peak Memory

Maximum memory usage during a single filter run, in megabytes.

---

## Statistical Tests

Raw differences in metrics can be misleading due to Monte Carlo noise. The comparison includes formal statistical tests to determine whether differences are significant.

### Diebold-Mariano Test

The **Diebold-Mariano (DM) test** compares the predictive accuracy of two filters based on their loss differentials:

$$
d_t = L(e_{t}^{A}) - L(e_{t}^{B})
$$

where $L(\cdot)$ is a loss function (e.g., squared error) and $e_t^A$, $e_t^B$ are the forecast errors from filters $A$ and $B$.

Under the null hypothesis $H_0: \mathbb{E}[d_t] = 0$ (equal predictive ability), the test statistic is:

$$
\text{DM} = \frac{\bar{d}}{\sqrt{\hat{\text{Var}}(\bar{d})}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

```python
# Run all pairwise statistical tests
test_results = comp.statistical_tests()
print(test_results)
```

```text
=== Diebold-Mariano Pairwise Tests (loss = squared error) ===

Filter A     vs  Filter B     |   DM stat  |  p-value  | Significant (5%)
--------------+---------------+------------+-----------+-----------------
Bootstrap     vs  SIR          |    4.21    |   0.000   |       Yes
Bootstrap     vs  Auxiliary    |    5.87    |   0.000   |       Yes
Bootstrap     vs  Guided       |    3.92    |   0.000   |       Yes
Bootstrap     vs  RBPF         |   -1.34    |   0.180   |       No
SIR           vs  Auxiliary    |    2.01    |   0.044   |       Yes
SIR           vs  Guided       |   -0.42    |   0.674   |       No
SIR           vs  RBPF         |   -4.89    |   0.000   |       Yes
Auxiliary     vs  Guided       |   -1.78    |   0.075   |       No
Auxiliary     vs  RBPF         |   -5.62    |   0.000   |       Yes
Guided        vs  RBPF         |   -4.13    |   0.000   |       Yes
```

### Log-Likelihood Ratio Test

For comparing filters via their log-likelihood estimates:

```python
# Compare log-likelihood distributions
comp.log_likelihood_test(filter_a="Bootstrap", filter_b="Auxiliary")
```

```text
=== Log-Likelihood Comparison ===
Filter A (Bootstrap):  mean = -412.32, std = 0.158
Filter B (Auxiliary):  mean = -411.65, std = 0.091

Difference: 0.67 (Auxiliary higher)
Paired t-test: t = 8.42, p < 0.001
Conclusion: Auxiliary produces significantly higher log-likelihood
```

!!! warning "Multiple comparisons"
    When testing all pairs of $K$ filters, you perform $\binom{K}{2}$ tests. Apply a Bonferroni correction or use the Holm method to control the family-wise error rate:

    ```python
    # Automatic Holm-Bonferroni correction
    comp.statistical_tests(correction="holm")
    ```

---

## Visualization

### Comparison Plots

```python
# All-in-one comparison plot
comp.plot_comparison(figsize=(16, 10))
```

This generates a four-panel figure:

1. **Boxplot of RMSE**: Distribution of RMSE across repeats for each filter
2. **Barplot of runtime**: Mean runtime with error bars
3. **ESS comparison**: Boxplot of mean ESS ratio per filter
4. **Log-likelihood**: Boxplot of $\log \hat{p}(y_{1:T})$ across repeats

### Individual Plots

```python
# RMSE boxplot
comp.plot_rmse(figsize=(10, 6))

# Runtime barplot
comp.plot_runtime(figsize=(10, 6))

# ESS comparison
comp.plot_ess(figsize=(10, 6))

# Log-likelihood comparison
comp.plot_log_likelihood(figsize=(10, 6))
```

### State Trajectory Comparison

Compare filtered state estimates from all filters at once:

```python
# Overlay filtered states from each filter
comp.plot_state_comparison(
    time_range=(100, 200),    # zoom into a region
    show_truth=True,          # plot true states
    show_ci=True,             # 95% credible intervals
    figsize=(14, 6),
)
```

---

## Automated Ranking

The comparison produces a weighted ranking across all metrics:

```python
# Default weights: RMSE=0.3, log-lik=0.3, ESS=0.2, runtime=0.1, memory=0.1
comp.ranking()

# Custom weights (e.g., prioritize speed)
comp.ranking(weights={
    "rmse": 0.2,
    "log_likelihood": 0.2,
    "ess": 0.1,
    "runtime": 0.4,
    "memory": 0.1,
})
```

```text
=== Filter Ranking (custom weights) ===

Rank | Filter       | Weighted Score | Best At
-----+--------------+----------------+--------------------
  1  | Bootstrap    |     0.82       | Runtime, Memory
  2  | SIR          |     0.78       | Balance
  3  | RBPF         |     0.76       | Runtime
  4  | Auxiliary    |     0.71       | RMSE, ESS, Log-lik
  5  | Guided       |     0.63       | ---

Note: Weights favor runtime (0.4). For accuracy, use default weights.
```

### Automated Recommendation

```python
# Get a recommendation based on use case
rec = comp.recommend(use_case="pmcmc")
print(rec)
```

```text
=== Recommendation for PMCMC ===
Recommended filter: SIR

Rationale:
- Log-likelihood variance is low (std = 0.091), ensuring good MCMC mixing
- ESS ratio (0.806) is well above the 0.3 threshold
- Runtime (1.34s) is moderate --- important since PMCMC runs the filter thousands of times
- Auxiliary is slightly more accurate, but 5% slower per iteration

Alternative: Auxiliary (if accuracy is paramount)
Avoid: RBPF (high log-likelihood variance will cause sticky chains)
```

!!! tip "Available use cases"
    The `recommend()` method accepts the following use cases, each with tailored metric weighting:

    | Use case | Priority |
    |----------|----------|
    | `"filtering"` | RMSE and ESS (default) |
    | `"pmcmc"` | Log-likelihood variance and runtime |
    | `"online"` | Runtime and memory |
    | `"smoothing"` | RMSE and ESS at long lags |
    | `"model_selection"` | Log-likelihood accuracy |

---

## Complete Example: Comparing 5 Filters on SV Model

```python
import numpy as np
from particlefilterbox import (
    BootstrapPF, SIRPF, AuxiliaryPF, GuidedPF, RBPF, PFConfig
)
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.diagnostics import FilterComparison

# --- 1. Simulate data ---
model = StochasticVolatility(variant="basic")
rng = np.random.default_rng(42)
states, obs = model.simulate(n_obs=300, rng=rng)

# --- 2. Configure filters ---
config = PFConfig(n_particles=2000, seed=42)
filters = [
    BootstrapPF(model, config),
    SIRPF(model, config),
    AuxiliaryPF(model, config),
    GuidedPF(model, config),
    RBPF(model, config),
]

# --- 3. Run comparison ---
comp = FilterComparison(
    model=model,
    filters=filters,
    observations=obs,
    true_states=states,
    n_repeats=50,
)
comp.run()

# --- 4. Summary table ---
comp.summary_table()

# --- 5. Statistical tests ---
comp.statistical_tests(correction="holm")

# --- 6. Visualize ---
comp.plot_comparison(figsize=(16, 10))

# --- 7. Ranking and recommendation ---
comp.ranking()
rec = comp.recommend(use_case="filtering")
print(rec)

# --- 8. Detailed state comparison ---
comp.plot_state_comparison(
    time_range=(100, 200),
    show_truth=True,
    show_ci=True,
)
```

!!! abstract "Key Takeaway"
    No single filter dominates in all scenarios. The **Auxiliary PF** often wins on accuracy for univariate models, while the **Bootstrap PF** is hard to beat on speed. For PMCMC applications, the **SIR PF** provides a good balance between log-likelihood precision and computational cost. Always validate your choice with a formal comparison rather than relying on rules of thumb.

---

## Best Practices

!!! tip "Designing a fair comparison"
    - **Same particle count**: Compare all filters at the same $N$ for a fair accuracy comparison.
    - **Same random seed**: Use a fixed base seed so that Monte Carlo differences reflect filter quality, not random chance.
    - **Enough repeats**: Use $n_{\text{repeats}} \geq 30$ for reliable standard errors and valid statistical tests.
    - **Same resampling**: Use the same resampling scheme (e.g., systematic) across all filters unless resampling is the variable under study.

!!! tip "Interpreting results"
    - A filter with **lower RMSE but higher runtime** may or may not be preferable --- it depends on your budget.
    - Use **accuracy per second** (RMSE / runtime) for a cost-adjusted comparison.
    - If no filter is significantly better than another (DM test $p > 0.05$), prefer the simpler or faster one.
    - Look at the **worst case** (max RMSE across repeats), not just the mean --- robustness matters.

!!! warning "Comparison pitfalls"
    - **Overfitting to one dataset**: Run comparisons on multiple simulated datasets if possible.
    - **Ignoring initialization**: Make sure all filters start from the same prior.
    - **Comparing apples to oranges**: RBPF exploits model structure that other filters ignore --- it is only fair to compare filters that use the same information.

---

## API Summary

| Method | Description |
|--------|-------------|
| `FilterComparison(model, filters, observations, ...)` | Create comparison object |
| `.run()` | Execute all filter runs |
| `.summary_table()` | Print comprehensive comparison table |
| `.statistical_tests(correction)` | Pairwise Diebold-Mariano tests |
| `.log_likelihood_test(filter_a, filter_b)` | Compare log-likelihoods of two filters |
| `.plot_comparison(**kwargs)` | Four-panel comparison figure |
| `.plot_rmse(**kwargs)` | RMSE boxplot |
| `.plot_runtime(**kwargs)` | Runtime barplot |
| `.plot_ess(**kwargs)` | ESS comparison |
| `.plot_log_likelihood(**kwargs)` | Log-likelihood boxplot |
| `.plot_state_comparison(**kwargs)` | Overlay state trajectories |
| `.ranking(weights)` | Weighted multi-metric ranking |
| `.recommend(use_case)` | Automated filter recommendation |

---

## See Also

- [ESS Diagnostic](ess-diagnostic.md) --- per-filter ESS analysis
- [Convergence Diagnostic](convergence.md) --- how many particles does each filter need?
- [Weight Diagnostic](weight-diagnostic.md) --- understand why some filters have lower ESS
- [Kalman Validation](kalman-validation.md) --- validate implementations against exact solutions
- [Filters Overview](../user-guide/filters/index.md) --- detailed documentation for each filter variant
- [Experiment Framework](../user-guide/experiment.md) --- run large-scale filter comparisons with multiple configurations
- [PMCMC Overview](../user-guide/pmcmc/index.md) --- when filter comparison informs PMCMC filter choice
- [Acceleration Overview](../acceleration/index.md) --- speed up comparisons with backend acceleration
