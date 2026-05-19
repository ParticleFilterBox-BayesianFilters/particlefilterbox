---
title: State Estimate Plots
description: Visualize filtered and smoothed state estimates with confidence bands, fan charts, and density plots
---

# State Estimate Plots

State estimate plots display the output of particle filters and smoothers: posterior means, credible intervals, quantile bands, and density estimates. These are the primary plots for communicating filtering and smoothing results.

```python
from particlefilterbox.viz import (
    plot_filtered_state,
    plot_smoothed_state,
    plot_filtered_vs_true,
    plot_filtering_vs_smoothing,
    plot_quantiles,
    plot_state_density,
)
```

---

## `plot_filtered_state` { #plot_filtered_state }

Filtered posterior mean with confidence bands, computed from the weighted particle cloud at each time step.

The filtered mean and variance are:

$$
\hat{x}_t = \sum_{i=1}^{N} w_t^{(i)} x_t^{(i)}, \qquad
\hat{\sigma}_t^2 = \sum_{i=1}^{N} w_t^{(i)} \left(x_t^{(i)} - \hat{x}_t\right)^2
$$

### API

```python
plot_filtered_state(
    result,                   # FilterResult
    state_idx=0,              # State dimension
    confidence=0.95,          # Confidence band level
    color="#4051B5",
    band_alpha=0.2,           # Confidence band transparency
    observations=None,        # Overlay raw observations
    obs_marker="o",
    obs_color="#9E9E9E",
    obs_size=15,
    ax=None,
    figsize=(12, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output |
| `state_idx` | `int` | `0` | State dimension index |
| `confidence` | `float` | `0.95` | Confidence level for bands (e.g., $0.95$ for $95\%$) |
| `observations` | `np.ndarray \| None` | `None` | Observation array to overlay as scatter points |
| `band_alpha` | `float` | `0.2` | Transparency of the confidence band |

### Example

```python
import numpy as np
from particlefilterbox import BootstrapFilter
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.viz import plot_filtered_state

model = StochasticVolatility(phi=0.97, sigma=0.15, beta=0.65)
true_states, observations = model.simulate(T=300, seed=42)

pf = BootstrapFilter(model, n_particles=1000)
result = pf.filter(observations)

fig, ax = plot_filtered_state(
    result,
    confidence=0.95,
    observations=observations,
    color="#1565C0",
)
```

!!! note "Output"
    A line plot of the filtered mean $\hat{x}_t$ over time, surrounded by a shaded band representing the $95\%$ credible interval. Gray dots show raw observations. The band width reflects filtering uncertainty -- it typically narrows when observations are informative and widens during volatile periods.

---

## `plot_smoothed_state` { #plot_smoothed_state }

Smoothed posterior mean with confidence bands. Smoothed estimates use information from the entire observation sequence $y_{1:T}$, producing tighter intervals than filtering.

### API

```python
plot_smoothed_state(
    smoothed,                 # SmootherResult
    state_idx=0,
    confidence=0.95,
    color="#E91E63",
    band_alpha=0.2,
    observations=None,
    ax=None,
    figsize=(12, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smoothed` | `SmootherResult` | required | Smoother output |
| `state_idx` | `int` | `0` | State dimension index |
| `confidence` | `float` | `0.95` | Confidence level for bands |

### Example

```python
from particlefilterbox.smoothers import FFBSm
from particlefilterbox.viz import plot_smoothed_state

smoother = FFBSm(model)
smoothed = smoother.smooth(result, n_trajectories=100)

fig, ax = plot_smoothed_state(
    smoothed,
    confidence=0.95,
    observations=observations,
    color="#C62828",
)
```

!!! note "Output"
    Similar to the filtered state plot but with narrower confidence bands. Smoothed estimates condition on the full dataset $y_{1:T}$ rather than only $y_{1:t}$, so the posterior variance is always less than or equal to the filtering variance.

---

## `plot_filtered_vs_true` { #plot_filtered_vs_true }

Overlay of the filtered estimate and the true (simulated) latent state, with error bands.

### API

```python
plot_filtered_vs_true(
    result,                   # FilterResult
    true_state,               # True state array
    state_idx=0,
    confidence=0.95,
    filtered_color="#4051B5",
    true_color="#2E7D32",
    true_linestyle="--",
    band_alpha=0.15,
    show_error=False,         # Show error subplot
    ax=None,
    figsize=(12, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output |
| `true_state` | `np.ndarray` | required | Ground truth state array of shape `(T,)` or `(T, d)` |
| `show_error` | `bool` | `False` | Add a subplot with $\hat{x}_t - x_t$ |
| `true_linestyle` | `str` | `"--"` | Line style for the true state |

### Example

```python
from particlefilterbox.viz import plot_filtered_vs_true

fig, ax = plot_filtered_vs_true(
    result,
    true_state=true_states,
    confidence=0.95,
    show_error=True,
    filtered_color="#1565C0",
    true_color="#2E7D32",
)
```

!!! note "Output"
    Two overlaid lines: the filtered mean (solid, blue) and the true state (dashed, green), with a shaded $95\%$ credible band around the filtered mean. When `show_error=True`, a second subplot displays the error $\hat{x}_t - x_t$, making systematic bias visible.

!!! tip "Validation workflow"
    This plot is essential for simulation studies. The true state should fall within the credible band approximately $95\%$ of the time. Systematic departures indicate model misspecification or filter misconfiguration.

---

## `plot_filtering_vs_smoothing` { #plot_filtering_vs_smoothing }

Side-by-side comparison of filtered and smoothed estimates, highlighting the variance reduction from smoothing.

### API

```python
plot_filtering_vs_smoothing(
    result,                   # FilterResult
    smoothed,                 # SmootherResult
    state_idx=0,
    confidence=0.95,
    true_state=None,
    filtered_color="#4051B5",
    smoothed_color="#E91E63",
    true_color="#2E7D32",
    layout="overlay",         # "overlay" or "side_by_side"
    ax=None,
    figsize=(12, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output |
| `smoothed` | `SmootherResult` | required | Smoother output |
| `layout` | `str` | `"overlay"` | `"overlay"` for same axes, `"side_by_side"` for subplots |
| `true_state` | `np.ndarray \| None` | `None` | True state to overlay |

### Example

```python
from particlefilterbox.viz import plot_filtering_vs_smoothing

fig, ax = plot_filtering_vs_smoothing(
    result,
    smoothed,
    true_state=true_states,
    layout="overlay",
    confidence=0.90,
)
```

!!! note "Output"
    With `layout="overlay"`: both estimates on the same axes, filtered in blue and smoothed in pink, each with its own confidence band. The smoothed band is visibly narrower. With `layout="side_by_side"`: two subplots stacked vertically sharing the x-axis, making it easy to compare band widths at any time step.

---

## `plot_quantiles` { #plot_quantiles }

Fan chart showing multiple quantile bands of the filtering distribution, creating a layered visualization of uncertainty.

### API

```python
plot_quantiles(
    result,                   # FilterResult
    state_idx=0,
    quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
    cmap="Blues",             # Colormap for quantile bands
    median_color="#0D47A1",   # Color for median line
    true_state=None,
    alpha_range=(0.15, 0.4), # Alpha range from outer to inner band
    ax=None,
    figsize=(12, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output |
| `quantiles` | `tuple[float, ...]` | `(0.05, 0.25, 0.5, 0.75, 0.95)` | Quantile levels (symmetric pairs recommended) |
| `cmap` | `str` | `"Blues"` | Colormap for nested bands |
| `alpha_range` | `tuple[float, float]` | `(0.15, 0.4)` | Transparency range (outer, inner) |

### Example

```python
from particlefilterbox.viz import plot_quantiles

fig, ax = plot_quantiles(
    result,
    quantiles=(0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95),
    cmap="Blues",
    true_state=true_states,
    median_color="#0D47A1",
)
```

!!! note "Output"
    A fan chart with nested shaded bands. The darkest inner band spans the $25$th to $75$th percentiles (interquartile range). Progressively lighter outer bands cover the $10$th--$90$th and $5$th--$95$th percentile ranges. The median line ($50$th percentile) runs through the center. This provides a richer view of uncertainty than a single confidence band.

---

## `plot_state_density` { #plot_state_density }

Kernel Density Estimate (KDE) of the filtering distribution at a single time step $t$, computed from the weighted particle cloud.

### API

```python
plot_state_density(
    result,                   # FilterResult
    t,                        # Time step
    state_idx=0,
    bw="silverman",           # Bandwidth method
    n_points=500,             # Number of evaluation points
    fill=True,                # Fill under the KDE curve
    color="#4051B5",
    fill_alpha=0.3,
    true_value=None,          # Mark true state value
    ax=None,
    figsize=(8, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output |
| `t` | `int` | required | Time step index |
| `bw` | `str \| float` | `"silverman"` | KDE bandwidth selection method or fixed value |
| `n_points` | `int` | `500` | Number of points for KDE evaluation |
| `true_value` | `float \| None` | `None` | True state value to mark with a vertical line |

### Example

```python
from particlefilterbox.viz import plot_state_density

fig, ax = plot_state_density(
    result,
    t=150,
    fill=True,
    color="#7B1FA2",
    true_value=true_states[150],
)
ax.set_title(r"$p(x_{150} \mid y_{1:150})$")
```

!!! note "Output"
    A smooth curve showing the estimated filtering density $p(x_t \mid y_{1:t})$ at time $t$. The area under the curve integrates to $1$. A vertical dashed line marks the true state if provided. Multi-modal densities indicate that the filter maintains multiple hypotheses about the state.

!!! tip "Animated density evolution"
    Loop over time steps to create an animation of the evolving filtering density:

    ```python
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from particlefilterbox.viz import plot_state_density

    fig, ax = plt.subplots(figsize=(8, 5))

    def update(t):
        ax.clear()
        plot_state_density(result, t=t, ax=ax, show=False,
                           true_value=true_states[t])
        ax.set_title(f"$p(x_{{{t}}} \\mid y_{{1:{t}}})$")
        ax.set_xlim(-3, 3)
        ax.set_ylim(0, 2)

    anim = FuncAnimation(fig, update, frames=range(0, 200, 2), interval=100)
    anim.save("density_evolution.gif", writer="pillow", dpi=100)
    ```

---

## Customization

### Confidence Band Styles

```python
# Adjust confidence level
plot_filtered_state(result, confidence=0.99)  # 99% band

# Multiple confidence bands via quantiles
plot_quantiles(result, quantiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
```

### Color Schemes

```python
# Custom colors for comparison plots
plot_filtering_vs_smoothing(
    result, smoothed,
    filtered_color="#1565C0",
    smoothed_color="#C62828",
    true_color="#2E7D32",
)

# Fan chart with different colormaps
plot_quantiles(result, cmap="Oranges")
plot_quantiles(result, cmap="Greens")
```

### Multi-Dimensional States

```python
import matplotlib.pyplot as plt

# Plot each state dimension
n_dims = result.state_dim
fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims), sharex=True)

state_names = ["Position", "Velocity", "Acceleration"]
for d in range(n_dims):
    plot_filtered_state(
        result,
        state_idx=d,
        ax=axes[d],
        show=False,
    )
    axes[d].set_title(state_names[d])

fig.tight_layout()
plt.savefig("multivariate_state.pdf", dpi=300, bbox_inches="tight")
plt.show()
```

### Publication-Ready Settings

```python
# Journal-quality filtered state
fig, ax = plot_filtered_state(
    result,
    confidence=0.95,
    theme="paper",
    figsize=(6.5, 3.5),
    band_alpha=0.15,
    observations=observations,
    obs_size=8,
    obs_color="#BDBDBD",
)
ax.set_xlabel("Time")
ax.set_ylabel(r"Log-Volatility $h_t$")
fig.savefig("filtered_sv.pdf", dpi=300, bbox_inches="tight")
```
