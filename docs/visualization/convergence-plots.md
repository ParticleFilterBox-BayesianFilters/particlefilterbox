---
title: Convergence & Diagnostic Plots
description: Visualization tools for convergence diagnostics, model comparison, and posterior predictive checks
---

# Convergence & Diagnostic Plots

Convergence and diagnostic plots help answer the critical question: **can we trust our results?** These functions visualize how estimates change with the number of particles, compare filter performance, and assess MCMC chain convergence.

```python
from particlefilterbox.viz import (
    plot_n_convergence,
    plot_filter_comparison_boxplot,
    plot_filter_comparison_time,
    plot_gelman_rubin,
    plot_geweke,
    plot_predictive_check,
    plot_marginal_likelihood,
)
```

---

## `plot_n_convergence` { #plot_n_convergence }

Shows how filter estimates converge as the number of particles $N$ increases. Essential for selecting an appropriate particle count.

### API

```python
plot_n_convergence(
    results,                  # Dict or list of FilterResults keyed by N
    metric="rmse",            # Metric to plot: "rmse", "log_likelihood", "ess_mean"
    true_state=None,          # True state for RMSE computation
    log_x=True,               # Log scale on x-axis (particle count)
    ci=True,                  # Show confidence intervals from repeated runs
    color="#4051B5",
    ax=None,
    figsize=(8, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `dict[int, FilterResult] \| list[tuple]` | required | Mapping from $N$ to filter results (or list of `(N, result)` tuples) |
| `metric` | `str` | `"rmse"` | Convergence metric: `"rmse"`, `"log_likelihood"`, or `"ess_mean"` |
| `true_state` | `np.ndarray \| None` | `None` | Ground-truth state (required for `"rmse"`) |
| `log_x` | `bool` | `True` | Use log scale for the particle count axis |
| `ci` | `bool` | `True` | Display confidence intervals (requires multiple runs per $N$) |

### Example

```python
import numpy as np
from particlefilterbox import BootstrapFilter
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.viz import plot_n_convergence

model = StochasticVolatility(phi=0.97, sigma=0.15, beta=0.65)
true_states, observations = model.simulate(T=200, seed=42)

# Run filter with increasing N
results = {}
for N in [50, 100, 250, 500, 1000, 2500, 5000]:
    pf = BootstrapFilter(model, n_particles=N)
    results[N] = pf.filter(observations)

fig, ax = plot_n_convergence(
    results,
    metric="rmse",
    true_state=true_states,
    log_x=True,
)
```

!!! note "Output"
    A line plot with $N$ on the x-axis (log scale) and the chosen metric on the y-axis. RMSE decreases as $N$ grows, typically following the $\mathcal{O}(1/\sqrt{N})$ Monte Carlo rate. The curve flattens at some $N^*$ beyond which additional particles yield diminishing returns — this is the recommended particle count.

!!! tip "Interpretation"
    The theoretical convergence rate for particle filters is $\mathcal{O}(1/\sqrt{N})$. If the observed rate is significantly slower, the model may have particle degeneracy issues. Consider a better proposal distribution.

---

## `plot_filter_comparison_boxplot` { #plot_filter_comparison_boxplot }

Box plot comparing performance metrics across different filter algorithms.

### API

```python
plot_filter_comparison_boxplot(
    comparison,               # ComparisonResult or dict of results
    metric="rmse",            # Metric to compare
    filter_names=None,        # Custom names for filters
    colors=None,              # Custom colors per filter
    show_points=True,         # Overlay individual data points
    ax=None,
    figsize=(8, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comparison` | `ComparisonResult \| dict` | required | Comparison results from `compare_filters()` or a dict mapping filter names to metric arrays |
| `metric` | `str` | `"rmse"` | Metric to compare: `"rmse"`, `"log_likelihood"`, `"ess_mean"`, `"runtime"` |
| `filter_names` | `list[str] \| None` | `None` | Display names for each filter |
| `colors` | `list[str] \| None` | `None` | Colors for each box |
| `show_points` | `bool` | `True` | Overlay jittered individual run points |

### Example

```python
from particlefilterbox import BootstrapFilter, GuidedFilter, AuxiliaryFilter
from particlefilterbox.diagnostics import compare_filters
from particlefilterbox.viz import plot_filter_comparison_boxplot

comparison = compare_filters(
    model=model,
    observations=observations,
    filters={
        "Bootstrap": BootstrapFilter(model, n_particles=500),
        "Guided": GuidedFilter(model, n_particles=500),
        "Auxiliary": AuxiliaryFilter(model, n_particles=500),
    },
    n_runs=20,
    true_state=true_states,
)

fig, ax = plot_filter_comparison_boxplot(
    comparison,
    metric="rmse",
)
```

!!! note "Output"
    A box-and-whisker plot with one box per filter algorithm. Each box shows the median, interquartile range, and outliers of the chosen metric across repeated runs. Jittered dots overlay individual runs for transparency. Lower RMSE boxes indicate better accuracy; tighter boxes indicate more consistent performance.

---

## `plot_filter_comparison_time` { #plot_filter_comparison_time }

Scatter plot of runtime vs. accuracy for different filter configurations, revealing the efficiency frontier.

### API

```python
plot_filter_comparison_time(
    comparison,               # ComparisonResult or dict
    accuracy_metric="rmse",   # Metric for y-axis
    time_metric="runtime",    # Metric for x-axis
    filter_names=None,        # Labels
    marker_size=80,           # Marker size
    annotate=True,            # Label each point with filter name
    pareto=True,              # Highlight the Pareto frontier
    ax=None,
    figsize=(8, 6),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comparison` | `ComparisonResult \| dict` | required | Comparison results |
| `accuracy_metric` | `str` | `"rmse"` | Metric for the y-axis (lower = better) |
| `time_metric` | `str` | `"runtime"` | Metric for the x-axis (seconds) |
| `annotate` | `bool` | `True` | Add text labels to each point |
| `pareto` | `bool` | `True` | Draw the Pareto frontier (best accuracy for a given runtime) |

### Example

```python
from particlefilterbox.viz import plot_filter_comparison_time

fig, ax = plot_filter_comparison_time(
    comparison,
    accuracy_metric="rmse",
    pareto=True,
)
```

!!! note "Output"
    A scatter plot where each point represents a filter configuration. The x-axis is runtime (seconds), the y-axis is RMSE (lower is better). The Pareto frontier connects points that are not dominated by any other — these are the optimal choices. Points below and to the left are preferable. Annotations label each filter for easy identification.

!!! tip "Choosing a filter"
    Points on the Pareto frontier represent the best accuracy-speed tradeoff. If your application is latency-sensitive, pick the leftmost Pareto point. If accuracy is paramount, pick the lowest.

---

## `plot_gelman_rubin` { #plot_gelman_rubin }

Tracks the evolution of the Gelman-Rubin $\hat{R}$ statistic across MCMC iterations for multiple chains.

### API

```python
plot_gelman_rubin(
    chains,                   # List of MCMC chains or PMCMCResult
    param_names=None,         # Parameter names
    threshold=1.1,            # Convergence threshold
    window="expanding",       # "expanding" or int for rolling window
    colors=None,              # Colors per parameter
    ax=None,
    figsize=(10, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chains` | `list[np.ndarray] \| PMCMCResult` | required | Multiple chains, each of shape `(n_iter, n_params)` |
| `param_names` | `list[str] \| None` | `None` | Parameter labels |
| `threshold` | `float` | `1.1` | Horizontal line marking the convergence criterion |
| `window` | `str \| int` | `"expanding"` | `"expanding"` computes $\hat{R}$ from iteration 1 to $t$; an `int` uses a rolling window |

### Example

```python
from particlefilterbox.viz import plot_gelman_rubin

# Run multiple PMCMC chains
chains = [pmcmc.run(observations, n_iter=5000) for _ in range(4)]

fig, ax = plot_gelman_rubin(
    chains=[c.chain for c in chains],
    param_names=["φ", "σ_η", "β"],
    threshold=1.1,
)
```

!!! note "Output"
    One line per parameter showing $\hat{R}$ vs. iteration. All lines should approach 1.0 from above. A horizontal dashed line at $\hat{R} = 1.1$ marks the conventional threshold. Parameters that remain above 1.1 have not converged — the chain needs more iterations or better tuning.

!!! warning "Interpretation"
    $\hat{R} < 1.1$ is necessary but not sufficient for convergence. Always combine with trace plots and effective sample size checks. The PSRF requires at least 2 chains.

---

## `plot_geweke` { #plot_geweke }

Plots Geweke's z-scores for a single MCMC chain, testing whether the first and last portions of the chain have the same mean.

### API

```python
plot_geweke(
    chain,                    # Single MCMC chain, shape (n_iter,) or (n_iter, n_params)
    param_names=None,         # Parameter names
    first_frac=0.1,           # Fraction of chain for the "first" segment
    last_frac=0.5,            # Fraction of chain for the "last" segment
    n_segments=20,            # Number of segments for sliding Geweke test
    significance=0.05,        # Significance level for z-score bands
    ax=None,
    figsize=(10, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain` | `np.ndarray` | required | MCMC chain, shape `(n_iter,)` or `(n_iter, n_params)` |
| `param_names` | `list[str] \| None` | `None` | Parameter labels |
| `first_frac` | `float` | `0.1` | Fraction of chain for early segment |
| `last_frac` | `float` | `0.5` | Fraction of chain for late segment |
| `n_segments` | `int` | `20` | Number of starting points for the sliding test |
| `significance` | `float` | `0.05` | Two-sided significance level for the critical bands |

### Example

```python
from particlefilterbox.viz import plot_geweke

fig, ax = plot_geweke(
    chain=pmcmc_result.chain,
    param_names=["φ", "σ_η", "β"],
    n_segments=25,
    significance=0.05,
)
```

!!! note "Output"
    A plot with the chain starting index on the x-axis and the Geweke z-score on the y-axis. Horizontal bands at $\pm 1.96$ (for $\alpha = 0.05$) define the acceptance region. Points falling outside the bands indicate that the early portion of the chain has not converged to the stationary distribution — those initial iterations should be discarded as burn-in.

!!! tip "Determining burn-in"
    Slide the first segment forward until all z-scores fall within the bands. The starting index at which this occurs is a data-driven burn-in recommendation.

---

## `plot_predictive_check` { #plot_predictive_check }

Posterior predictive check: overlays simulated data from the posterior on the observed data to assess model fit.

### API

```python
plot_predictive_check(
    ppc,                      # PosteriorPredictiveResult or dict
    observations=None,        # Observed data
    n_draws=100,              # Number of predictive draws to overlay
    obs_color="black",        # Color for observed data
    draw_color="#4051B5",     # Color for predictive draws
    draw_alpha=0.1,           # Transparency per draw
    summary="median",         # Summary statistic: "median", "mean", or None
    quantiles=(0.05, 0.95),   # Credible band for predictive distribution
    ax=None,
    figsize=(12, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ppc` | `PosteriorPredictiveResult \| dict` | required | Predictive simulation output |
| `observations` | `np.ndarray \| None` | `None` | Observed data to overlay |
| `n_draws` | `int` | `100` | Number of simulated datasets to draw |
| `summary` | `str \| None` | `"median"` | Summary line: `"median"`, `"mean"`, or `None` |
| `quantiles` | `tuple[float, float]` | `(0.05, 0.95)` | Predictive credible interval |

### Example

```python
from particlefilterbox.diagnostics import posterior_predictive_check
from particlefilterbox.viz import plot_predictive_check

ppc = posterior_predictive_check(
    model=model,
    chain=pmcmc_result.chain,
    n_draws=500,
    T=200,
)

fig, ax = plot_predictive_check(
    ppc,
    observations=observations,
    n_draws=200,
    quantiles=(0.05, 0.95),
)
```

!!! note "Output"
    A spaghetti plot of simulated data series in faint blue, overlaid on the observed data in solid black. A shaded 90% predictive band summarizes the ensemble. The observed data should lie mostly within the band. Systematic departures indicate model misspecification — e.g., if observed tails exceed the predictive band, the model underestimates tail risk.

---

## `plot_marginal_likelihood` { #plot_marginal_likelihood }

Bar chart or table comparing marginal likelihoods (or log Bayes factors) across competing models.

### API

```python
plot_marginal_likelihood(
    results,                  # Dict mapping model names to log marginal likelihoods
    reference=None,           # Reference model for Bayes factor computation
    style="bar",              # "bar" or "table"
    sort=True,                # Sort by marginal likelihood (descending)
    colors=None,              # Custom colors
    annotate=True,            # Show values on bars
    fmt=".1f",                # Number format
    ax=None,
    figsize=(8, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `dict[str, float]` | required | Mapping from model name to log marginal likelihood $\log \hat{p}(y \mid \mathcal{M})$ |
| `reference` | `str \| None` | `None` | Reference model for computing log Bayes factors |
| `style` | `str` | `"bar"` | `"bar"` chart or `"table"` display |
| `sort` | `bool` | `True` | Sort models by marginal likelihood |
| `annotate` | `bool` | `True` | Display numeric values on bars |

### Example

```python
from particlefilterbox.viz import plot_marginal_likelihood

# Marginal likelihoods from particle filter runs
ml_results = {
    "SV": -342.5,
    "SV-Leverage": -338.1,
    "SV-Jump": -335.7,
    "SV-2Factor": -340.2,
}

fig, ax = plot_marginal_likelihood(
    ml_results,
    reference="SV",
    sort=True,
)
```

!!! note "Output"
    A horizontal bar chart with model names on the y-axis and log marginal likelihood on the x-axis. Bars are sorted from highest (best) to lowest. Annotations show exact values and, if a reference model is specified, log Bayes factors relative to the reference. The preferred model has the longest bar.

The standard interpretation scale for Bayes factors:

| $\log_{10} BF$ | Evidence |
|-----------------|----------|
| $< 0$ | Supports reference model |
| $0$ -- $0.5$ | Barely worth mentioning |
| $0.5$ -- $1$ | Substantial |
| $1$ -- $2$ | Strong |
| $> 2$ | Decisive |

---

## Composing Diagnostic Panels

Build comprehensive diagnostic dashboards by composing individual plots:

```python
import matplotlib.pyplot as plt
from particlefilterbox.viz import (
    plot_gelman_rubin,
    plot_geweke,
    plot_predictive_check,
    plot_marginal_likelihood,
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_gelman_rubin(chains, ax=axes[0, 0], show=False)
plot_geweke(chain=chains[0], ax=axes[0, 1], show=False)
plot_predictive_check(ppc, observations=obs, ax=axes[1, 0], show=False)
plot_marginal_likelihood(ml_results, ax=axes[1, 1], show=False)

axes[0, 0].set_title("Gelman-Rubin $\\hat{R}$")
axes[0, 1].set_title("Geweke Z-scores")
axes[1, 0].set_title("Posterior Predictive Check")
axes[1, 1].set_title("Model Comparison")

fig.suptitle("PMCMC Convergence Diagnostics", fontsize=14)
fig.tight_layout()
plt.savefig("convergence_panel.pdf", dpi=300, bbox_inches="tight")
plt.show()
```

---

## Export and Resolution

All convergence plots support publication-quality export:

```python
# High-resolution PNG for reports
fig, ax = plot_gelman_rubin(chains, show=False)
fig.savefig("gelman_rubin.png", dpi=600, bbox_inches="tight")

# Vector PDF for journal submission
fig.savefig("gelman_rubin.pdf", bbox_inches="tight")

# SVG for web dashboards
fig.savefig("gelman_rubin.svg", bbox_inches="tight")
```

!!! tip "Theme integration"
    All diagnostic functions accept the `theme` parameter:

    ```python
    plot_gelman_rubin(chains, theme="paper")       # Journal-ready
    plot_geweke(chain, theme="presentation")        # Slides
    plot_predictive_check(ppc, theme="dark")        # Dashboards
    ```
