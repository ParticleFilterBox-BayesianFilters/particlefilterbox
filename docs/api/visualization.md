---
title: "Visualization API"
description: "API reference for particlefilterbox.visualization — particle, weight, state, PMCMC, model, and diagnostic plots, plus theming"
---

# Visualization API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.visualization import ...`
    **Source**: `particlefilterbox/visualization/`

## Overview

The visualization module exposes 30+ plotting functions organized by concern: particle clouds, weight dynamics, filtered/smoothed states, PMCMC chain diagnostics, model-specific visualizations (volatility, regimes, jumps), and filter diagnostics. All functions share a consistent signature:

```python
fig = plot_*(..., ax=None, theme=None, **kwargs)
```

Every plot accepts an optional `matplotlib.axes.Axes` (for composition) and a `theme` (string or `Theme` instance). The return value is a `matplotlib.figure.Figure` — or a Plotly figure when `backend="plotly"` is passed.

| Category | Functions |
|----------|-----------|
| Particle | `plot_particles`, `plot_particle_evolution`, `plot_particle_histogram`, `plot_particle_cloud_2d`, `plot_particle_trajectories` |
| Weight | `plot_weights`, `plot_weight_evolution`, `plot_weight_histogram`, `plot_ess_over_time`, `plot_max_weight`, `plot_weight_entropy` |
| State | `plot_filtered_state`, `plot_smoothed_state`, `plot_filtered_vs_true`, `plot_filtering_vs_smoothing`, `plot_quantiles`, `plot_state_density` |
| PMCMC | `plot_trace`, `plot_posterior`, `plot_posterior_2d`, `plot_acf`, `plot_running_mean`, `plot_acceptance_rate`, `plot_prior_vs_posterior` |
| Model | `plot_volatility`, `plot_impulse_response`, `plot_regime_probabilities`, `plot_jump_detection` |
| Diagnostic | `plot_n_convergence`, `plot_filter_comparison_boxplot`, `plot_gelman_rubin`, `plot_predictive_check` |
| Theming | `set_theme`, `get_theme`, `Theme` |

---

## Particle Plots

### `plot_particles()`

Scatter of particle positions at a single time step.

```python
def plot_particles(
    cloud: ParticleCloud,
    dims: tuple[int, ...] = (0,),
    ax: Axes | None = None,
    theme: str | Theme | None = None,
    size_by_weight: bool = True,
    alpha: float = 0.3,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cloud` | `ParticleCloud` | — | Single-time particle cloud |
| `dims` | `tuple[int, ...]` | `(0,)` | State dimensions to plot |
| `size_by_weight` | `bool` | `True` | Scale marker size by normalized weight |

### `plot_particle_evolution()`

Particle positions vs. time with trajectories.

```python
def plot_particle_evolution(
    result: ParticleFilterResults,
    dim: int = 0,
    n_samples: int = 100,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_particle_histogram()`

Weighted histogram of the particle cloud at a given time step.

```python
def plot_particle_histogram(
    cloud: ParticleCloud,
    dim: int = 0,
    bins: int = 50,
    overlay_kde: bool = True,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_particle_cloud_2d()`

2-D scatter of the cloud in a chosen pair of state dimensions.

```python
def plot_particle_cloud_2d(
    cloud: ParticleCloud,
    dims: tuple[int, int] = (0, 1),
    ax: Axes | None = None,
    theme: str | Theme | None = None,
    contour: bool = False,
) -> Figure
```

### `plot_particle_trajectories()`

Sample of full ancestral trajectories $x_{1:T}^{(i)}$.

```python
def plot_particle_trajectories(
    result: ParticleFilterResults,
    dim: int = 0,
    n_samples: int = 50,
    alpha: float = 0.2,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

---

## Weight Plots

### `plot_weights()`

Normalized weights at a single time step.

```python
def plot_weights(
    cloud: ParticleCloud,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
    sort: bool = True,
) -> Figure
```

### `plot_weight_evolution()`

Heatmap of normalized weights $(N \times T)$.

```python
def plot_weight_evolution(
    result: ParticleFilterResults,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
    cmap: str = "viridis",
) -> Figure
```

### `plot_weight_histogram()`

Histogram of weights at a given time.

```python
def plot_weight_histogram(
    cloud: ParticleCloud,
    bins: int = 30,
    log_scale: bool = True,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_ess_over_time()`

Effective sample size trajectory, with optional resampling threshold.

```python
def plot_ess_over_time(
    result: ParticleFilterResults,
    threshold: float | None = None,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_max_weight()`

Maximum normalized weight over time — early warning for degeneracy.

```python
def plot_max_weight(
    result: ParticleFilterResults,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_weight_entropy()`

Entropy of the weight distribution: $H_t = -\sum_i W_t^{(i)} \log W_t^{(i)}$.

```python
def plot_weight_entropy(
    result: ParticleFilterResults,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

---

## State Plots

### `plot_filtered_state()`

Filtered mean with credible bands.

```python
def plot_filtered_state(
    result: ParticleFilterResults,
    dim: int = 0,
    band: float = 0.95,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_smoothed_state()`

Smoothed mean with credible bands.

```python
def plot_smoothed_state(
    result: ParticleSmootherResults,
    dim: int = 0,
    band: float = 0.95,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_filtered_vs_true()`

Filtered estimate overlaid on true latent states (for simulated data).

```python
def plot_filtered_vs_true(
    result: ParticleFilterResults,
    true_state: NDArray[np.float64],
    dim: int = 0,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_filtering_vs_smoothing()`

Side-by-side comparison of filtered and smoothed estimates.

```python
def plot_filtering_vs_smoothing(
    filter_result: ParticleFilterResults,
    smoother_result: ParticleSmootherResults,
    dim: int = 0,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_quantiles()`

Posterior quantiles $q_\alpha, q_{1-\alpha}$ as a fan chart.

```python
def plot_quantiles(
    result: ParticleFilterResults,
    dim: int = 0,
    levels: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_state_density()`

KDE of the posterior marginal $p(x_t \mid y_{1:t})$ at one time.

```python
def plot_state_density(
    cloud: ParticleCloud,
    dim: int = 0,
    kernel: str = "gaussian",
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

---

## PMCMC Plots

### `plot_trace()`

Traceplot of one or more chains.

```python
def plot_trace(
    chain: PMCMCResults,
    param: str | list[str] | None = None,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
    thin: int = 1,
) -> Figure
```

### `plot_posterior()`

Marginal posterior density with mean/CI overlay.

```python
def plot_posterior(
    chain: PMCMCResults,
    param: str,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
    hdi: float = 0.95,
) -> Figure
```

### `plot_posterior_2d()`

Joint posterior contour / hexbin for two parameters.

```python
def plot_posterior_2d(
    chain: PMCMCResults,
    params: tuple[str, str],
    kind: str = "kde",
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

`kind` ∈ {`"kde"`, `"hexbin"`, `"scatter"`}.

### `plot_acf()`

Autocorrelation function of a scalar parameter.

```python
def plot_acf(
    chain: PMCMCResults,
    param: str,
    max_lag: int = 50,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_running_mean()`

Running posterior mean across iterations.

```python
def plot_running_mean(
    chain: PMCMCResults,
    param: str,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_acceptance_rate()`

Running Metropolis acceptance rate.

```python
def plot_acceptance_rate(
    chain: PMCMCResults,
    window: int = 200,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_prior_vs_posterior()`

Prior and posterior densities side-by-side.

```python
def plot_prior_vs_posterior(
    chain: PMCMCResults,
    param: str,
    prior: Callable[[NDArray], NDArray],
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

---

## Model Plots

### `plot_volatility()`

Filtered log-volatility path (SV models).

```python
def plot_volatility(
    result: ParticleFilterResults,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
    scale: str = "log",
) -> Figure
```

### `plot_impulse_response()`

Impulse-response function from a DSGE or state-space model.

```python
def plot_impulse_response(
    model: ParticleFilterModel,
    shock: str,
    horizon: int = 40,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_regime_probabilities()`

Filtered probabilities of each regime (regime-switching models).

```python
def plot_regime_probabilities(
    result: ParticleFilterResults,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
    stacked: bool = True,
) -> Figure
```

### `plot_jump_detection()`

Posterior jump probability over time (jump-diffusion models).

```python
def plot_jump_detection(
    result: ParticleFilterResults,
    threshold: float = 0.5,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

---

## Diagnostic Plots

### `plot_n_convergence()`

Convergence of an estimator as $N \to \infty$.

```python
def plot_n_convergence(
    study: ConvergenceStudy,
    metric: str = "rmse",
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_filter_comparison_boxplot()`

Boxplot of a metric across filters and replications.

```python
def plot_filter_comparison_boxplot(
    comparison: ComparisonResult,
    metric: str = "log_likelihood",
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_gelman_rubin()`

Gelman–Rubin $\hat{R}$ statistics across PMCMC chains.

```python
def plot_gelman_rubin(
    chains: list[PMCMCResults],
    params: list[str] | None = None,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

### `plot_predictive_check()`

Posterior predictive check: observed vs. replicated data.

```python
def plot_predictive_check(
    result: ParticleFilterResults,
    observed: NDArray[np.float64],
    statistic: Callable[[NDArray], float] | None = None,
    ax: Axes | None = None,
    theme: str | Theme | None = None,
) -> Figure
```

---

## Theming

### `Theme`

Dataclass holding colors, fonts, and grid style used across all plots.

```python
@dataclass
class Theme:
    name: str
    palette: list[str]
    font_family: str = "sans-serif"
    font_size: int = 11
    grid: bool = True
    background: str = "white"
    foreground: str = "black"
    accent: str = "#1f77b4"
```

### `set_theme()`

```python
def set_theme(name: str | Theme) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str \| Theme` | Theme name (`"default"`, `"academic"`, `"dark"`, `"publication"`) or a custom `Theme` instance |

### `get_theme()`

```python
def get_theme(name: str | None = None) -> Theme
```

Returns the current (or named) theme.

### Example

```python
from particlefilterbox.visualization import set_theme, plot_filtered_state

set_theme("publication")
fig = plot_filtered_state(result, dim=0, band=0.95)
fig.savefig("state.pdf", bbox_inches="tight")
```

---

## See Also

- [Reports API](reports.md) — embeds visualizations in HTML/LaTeX reports
- [Diagnostics API](diagnostics.md) — underlying metrics for diagnostic plots
- [PMCMC API](pmcmc.md) — chain results consumed by PMCMC plots
