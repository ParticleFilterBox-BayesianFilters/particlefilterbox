---
title: Visualization
description: Publication-ready plots for particle filters, SMC, and PMCMC diagnostics
---

# Visualization

The `particlefilterbox.viz` module provides **publication-ready plots** for every stage of the particle filtering and PMCMC workflow. All plots are designed with sensible defaults while remaining fully customizable.

## Philosophy

1. **Ready out of the box** -- every function produces a complete, labeled figure
2. **Customizable** -- colors, sizes, labels, and axes are exposed as parameters
3. **Consistent** -- shared style across all plot types for cohesive manuscripts
4. **Dual backend** -- matplotlib for static/publication, plotly for interactive exploration

## Quick Start

```python
from particlefilterbox.viz import (
    plot_filtered_state,
    plot_ess_over_time,
    plot_particles,
    plot_trace,
)

# After running a particle filter
result = bootstrap_filter.filter(observations)

# One-liner plots
plot_filtered_state(result)
plot_ess_over_time(result)
plot_particles(result.clouds[0], t=0)
```

## Backends

=== "matplotlib (default)"

    ```python
    from particlefilterbox.viz import plot_filtered_state

    fig, ax = plot_filtered_state(result)
    fig.savefig("filtered_state.pdf", dpi=300, bbox_inches="tight")
    ```

    Best for: journal figures, PDF reports, batch generation.

=== "plotly (interactive)"

    ```python
    from particlefilterbox.viz import plot_filtered_state

    fig = plot_filtered_state(result, backend="plotly")
    fig.show()
    ```

    Best for: exploratory analysis, dashboards, presentations.

## Themes

All plot functions accept a `theme` parameter to switch between predefined styles:

| Theme | Description | Best For |
|-------|-------------|----------|
| `"default"` | Clean, minimal style with indigo palette | General use |
| `"paper"` | Black/white with markers, LaTeX-ready | Journal submissions |
| `"presentation"` | Larger fonts, high contrast | Slides and talks |
| `"dark"` | Dark background, bright accents | Dashboards |

```python
plot_filtered_state(result, theme="paper")
```

!!! tip "Custom themes"
    Create a custom theme by passing a dictionary of matplotlib `rcParams`:

    ```python
    my_theme = {
        "font.size": 14,
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.0,
    }
    plot_filtered_state(result, theme=my_theme)
    ```

## Quick Reference

### Particle Plots

| Function | Description |
|----------|-------------|
| [`plot_particles`](particle-plots.md#plot_particles) | Scatter plot of particles at time $t$ |
| [`plot_particle_evolution`](particle-plots.md#plot_particle_evolution) | Particles over time |
| [`plot_particle_histogram`](particle-plots.md#plot_particle_histogram) | Weighted histogram of a particle cloud |
| [`plot_particle_cloud_2d`](particle-plots.md#plot_particle_cloud_2d) | 2D scatter with weight-proportional size |
| [`plot_particle_trajectories`](particle-plots.md#plot_particle_trajectories) | Sampled particle trajectories |

### Weight Plots

| Function | Description |
|----------|-------------|
| [`plot_weights`](weight-plots.md#plot_weights) | Bar plot of weights at time $t$ |
| [`plot_weight_evolution`](weight-plots.md#plot_weight_evolution) | Heatmap of weights over time |
| [`plot_weight_histogram`](weight-plots.md#plot_weight_histogram) | Distribution of weights |
| [`plot_ess_over_time`](weight-plots.md#plot_ess_over_time) | ESS trajectory with threshold |
| [`plot_max_weight`](weight-plots.md#plot_max_weight) | Maximum weight over time |
| [`plot_weight_entropy`](weight-plots.md#plot_weight_entropy) | Weight entropy over time |

### State Estimate Plots

| Function | Description |
|----------|-------------|
| [`plot_filtered_state`](state-plots.md#plot_filtered_state) | Filtered mean with confidence bands |
| [`plot_smoothed_state`](state-plots.md#plot_smoothed_state) | Smoothed mean with confidence bands |
| [`plot_filtered_vs_true`](state-plots.md#plot_filtered_vs_true) | Filtered estimate vs. true state |
| [`plot_filtering_vs_smoothing`](state-plots.md#plot_filtering_vs_smoothing) | Side-by-side comparison |
| [`plot_quantiles`](state-plots.md#plot_quantiles) | Fan chart with quantile bands |
| [`plot_state_density`](state-plots.md#plot_state_density) | KDE of the filtered state at time $t$ |

### PMCMC Plots

| Function | Description |
|----------|-------------|
| [`plot_trace`](pmcmc-plots.md#plot_trace) | Trace plots for chain parameters |
| [`plot_posterior`](pmcmc-plots.md#plot_posterior) | Posterior histogram/KDE |
| [`plot_posterior_2d`](pmcmc-plots.md#plot_posterior_2d) | Bivariate posterior (contour + scatter) |
| [`plot_acf`](pmcmc-plots.md#plot_acf) | Autocorrelation function |
| [`plot_running_mean`](pmcmc-plots.md#plot_running_mean) | Running mean for convergence |
| [`plot_acceptance_rate`](pmcmc-plots.md#plot_acceptance_rate) | Acceptance rate over iterations |
| [`plot_prior_vs_posterior`](pmcmc-plots.md#plot_prior_vs_posterior) | Prior vs. posterior overlay |

## Common Parameters

All plot functions share a set of common keyword arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `matplotlib.axes.Axes` | `None` | Axes to plot on; creates new figure if `None` |
| `figsize` | `tuple[float, float]` | `(10, 6)` | Figure size in inches |
| `title` | `str` | Auto | Plot title |
| `xlabel` | `str` | Auto | X-axis label |
| `ylabel` | `str` | Auto | Y-axis label |
| `theme` | `str \| dict` | `"default"` | Plot theme or custom rcParams |
| `backend` | `str` | `"matplotlib"` | `"matplotlib"` or `"plotly"` |
| `show` | `bool` | `True` | Whether to display the plot |

## Composing Plots

Use the `ax` parameter to compose multiple plots into a single figure:

```python
import matplotlib.pyplot as plt
from particlefilterbox.viz import plot_filtered_state, plot_ess_over_time

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

plot_filtered_state(result, ax=axes[0], show=False)
plot_ess_over_time(result, ax=axes[1], show=False)

axes[0].set_title("Filtered State Estimate")
axes[1].set_title("Effective Sample Size")
fig.tight_layout()
plt.savefig("diagnostic_panel.pdf", dpi=300, bbox_inches="tight")
plt.show()
```

!!! info "Return values"
    All matplotlib-based functions return `(fig, ax)` so you can further customize after the call:

    ```python
    fig, ax = plot_filtered_state(result, show=False)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    fig.savefig("annotated.pdf")
    ```
