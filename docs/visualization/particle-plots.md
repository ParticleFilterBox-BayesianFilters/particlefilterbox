---
title: Particle Plots
description: Visualize particle clouds, evolution, histograms, 2D scatters, and trajectories
---

# Particle Plots

Particle plots visualize the discrete approximation that particle filters use to represent posterior distributions. They are essential for understanding how particles explore the state space, where they concentrate, and how degeneracy manifests visually.

```python
from particlefilterbox.viz import (
    plot_particles,
    plot_particle_evolution,
    plot_particle_histogram,
    plot_particle_cloud_2d,
    plot_particle_trajectories,
)
```

---

## `plot_particles` { #plot_particles }

Scatter plot of particle positions at a single time step, with optional color or size encoding of weights.

### API

```python
plot_particles(
    cloud,                    # ParticleCloud at time t
    t=None,                   # Time index (for title annotation)
    color_by_weight=True,     # Color particles by normalized weight
    size_by_weight=False,     # Scale marker size by weight
    cmap="viridis",           # Colormap for weight encoding
    marker_size=20,           # Base marker size
    alpha=0.7,                # Marker transparency
    ax=None,
    figsize=(10, 4),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cloud` | `ParticleCloud` | required | Particle cloud to visualize |
| `t` | `int \| None` | `None` | Time index for title annotation |
| `color_by_weight` | `bool` | `True` | Map particle color to normalized weight |
| `size_by_weight` | `bool` | `False` | Scale marker area proportional to weight |
| `cmap` | `str` | `"viridis"` | Matplotlib colormap name |
| `marker_size` | `float` | `20` | Base marker size in points$^2$ |
| `alpha` | `float` | `0.7` | Marker transparency ($0$--$1$) |

### Example

```python
import numpy as np
from particlefilterbox import BootstrapFilter
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.viz import plot_particles

# Setup model and run filter
model = StochasticVolatility(phi=0.97, sigma=0.15, beta=0.65)
observations = model.simulate(T=200, seed=42)[1]

pf = BootstrapFilter(model, n_particles=500)
result = pf.filter(observations)

# Plot particles at t=50
fig, ax = plot_particles(
    result.clouds[50],
    t=50,
    color_by_weight=True,
    cmap="plasma",
    marker_size=30,
)
```

!!! note "Output"
    A horizontal scatter where the x-axis is the particle index and the y-axis is the state value. Particles with higher weights appear in warmer colors (with `"plasma"`). Clusters indicate regions of high posterior probability.

---

## `plot_particle_evolution` { #plot_particle_evolution }

Displays all particles across time as a scatter plot, revealing how the particle cloud tracks the latent state.

### API

```python
plot_particle_evolution(
    result,                   # FilterResult object
    state_idx=0,              # State dimension to plot
    subsample=None,           # Max particles to display (None = all)
    color_by_weight=True,     # Color by weight at each time step
    cmap="viridis",
    alpha=0.3,
    marker_size=2,
    true_state=None,          # Overlay true state if available
    ax=None,
    figsize=(12, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Output from a particle filter |
| `state_idx` | `int` | `0` | State dimension index for multivariate models |
| `subsample` | `int \| None` | `None` | Randomly subsample particles for faster rendering |
| `true_state` | `np.ndarray \| None` | `None` | True state array to overlay as a line |

### Example

```python
from particlefilterbox.viz import plot_particle_evolution

true_states, observations = model.simulate(T=200, seed=42)
result = pf.filter(observations)

fig, ax = plot_particle_evolution(
    result,
    color_by_weight=True,
    alpha=0.2,
    true_state=true_states,
)
```

!!! note "Output"
    A dense scatter plot spanning the full time axis. Each dot is one particle at one time step. The true state appears as a solid line cutting through the particle cloud. Well-performing filters show tight clustering around the true state; degeneracy appears as sparse regions or sudden collapses.

---

## `plot_particle_histogram` { #plot_particle_histogram }

Weighted histogram of particle positions, approximating the filtering density at a single time step.

### API

```python
plot_particle_histogram(
    cloud,                    # ParticleCloud
    state_idx=0,              # State dimension
    bins=50,                  # Number of histogram bins
    density=True,             # Normalize to density
    kde=True,                 # Overlay KDE curve
    kde_bw="silverman",       # KDE bandwidth method
    color="#4051B5",          # Bar color
    kde_color="#E91E63",      # KDE line color
    alpha=0.6,
    ax=None,
    figsize=(8, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cloud` | `ParticleCloud` | required | Particle cloud to histogram |
| `bins` | `int` | `50` | Number of histogram bins |
| `density` | `bool` | `True` | Normalize histogram area to 1 |
| `kde` | `bool` | `True` | Overlay a weighted KDE curve |
| `kde_bw` | `str \| float` | `"silverman"` | Bandwidth selection method or fixed value |

### Example

```python
from particlefilterbox.viz import plot_particle_histogram

fig, ax = plot_particle_histogram(
    result.clouds[100],
    bins=40,
    kde=True,
    color="#3F51B5",
    kde_color="#FF5722",
)
ax.axvline(true_states[100], color="black", linestyle="--", label="True state")
ax.legend()
```

!!! note "Output"
    A histogram with the x-axis representing state values and the y-axis representing density. The KDE overlay provides a smooth approximation of the filtering distribution $p(x_t \mid y_{1:t})$. A dashed vertical line marks the true state.

---

## `plot_particle_cloud_2d` { #plot_particle_cloud_2d }

Two-dimensional scatter plot for multivariate state spaces, where marker size is proportional to particle weight.

### API

```python
plot_particle_cloud_2d(
    cloud,                    # ParticleCloud
    dims=(0, 1),              # State dimensions to plot
    size_scale=500,           # Scaling factor for marker sizes
    color="#4051B5",
    alpha=0.5,
    edgecolors="white",       # Marker edge color
    linewidths=0.5,           # Marker edge width
    ax=None,
    figsize=(8, 8),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cloud` | `ParticleCloud` | required | Particle cloud |
| `dims` | `tuple[int, int]` | `(0, 1)` | Pair of state dimension indices |
| `size_scale` | `float` | `500` | Multiplier: marker area $= w_i \times$ `size_scale` |
| `edgecolors` | `str` | `"white"` | Marker edge color for visibility |

### Example

```python
from particlefilterbox.viz import plot_particle_cloud_2d

# For a 2D state-space model (e.g., position + velocity)
fig, ax = plot_particle_cloud_2d(
    result.clouds[100],
    dims=(0, 1),
    size_scale=800,
    color="#7C4DFF",
    alpha=0.6,
)
ax.set_xlabel("Position")
ax.set_ylabel("Velocity")
```

!!! note "Output"
    A 2D scatter where each particle is a circle positioned at its state values for the two chosen dimensions. Larger circles indicate higher weights, making the effective posterior mass immediately visible. Clusters of large circles indicate modes of the bivariate posterior.

---

## `plot_particle_trajectories` { #plot_particle_trajectories }

Draws trajectories of selected particles over time, useful for understanding how individual particles evolve through resampling.

### API

```python
plot_particle_trajectories(
    result,                   # FilterResult
    n=50,                     # Number of trajectories to draw
    state_idx=0,              # State dimension
    selection="random",       # "random", "highest_weight", or list of indices
    true_state=None,          # Overlay true state
    cmap="tab20",             # Colormap for trajectories
    alpha=0.4,
    linewidth=0.8,
    ax=None,
    figsize=(12, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | `50` | Number of trajectories to draw |
| `selection` | `str \| list[int]` | `"random"` | Selection strategy or explicit particle indices |
| `true_state` | `np.ndarray \| None` | `None` | True state to overlay |
| `linewidth` | `float` | `0.8` | Trajectory line width |

### Example

```python
from particlefilterbox.viz import plot_particle_trajectories

fig, ax = plot_particle_trajectories(
    result,
    n=30,
    selection="random",
    true_state=true_states,
    alpha=0.3,
    linewidth=0.6,
)
```

!!! note "Output"
    Multiple thin colored lines, each tracing a single particle's path through time. When resampling occurs, trajectories merge (many particles copy the same ancestor). The true state appears as a thick black line. Trajectory collapse after resampling is a direct visual indicator of sample impoverishment.

!!! tip "Diagnosing degeneracy"
    If most trajectories merge into a few paths early on, particle degeneracy is severe. Consider increasing `n_particles` or switching to a better proposal (e.g., auxiliary or guided PF).

---

## Customization

### Color Palettes

```python
# Use any matplotlib colormap
plot_particles(cloud, cmap="coolwarm")
plot_particle_evolution(result, cmap="inferno")

# Custom discrete colors for trajectories
plot_particle_trajectories(result, cmap="Set2", n=8)
```

### Marker Styles

```python
# Adjust marker size and transparency
plot_particles(cloud, marker_size=50, alpha=0.9)

# Use size encoding instead of color
plot_particles(cloud, color_by_weight=False, size_by_weight=True, marker_size=100)
```

### Combining with Matplotlib

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_particle_histogram(result.clouds[0], ax=axes[0], show=False)
plot_particle_histogram(result.clouds[100], ax=axes[1], show=False)
plot_particle_histogram(result.clouds[199], ax=axes[2], show=False)

for i, t in enumerate([0, 100, 199]):
    axes[i].set_title(f"$t = {t}$")
    axes[i].axvline(true_states[t], color="red", linestyle="--")

fig.suptitle("Evolution of the Filtering Distribution", fontsize=14)
fig.tight_layout()
plt.savefig("filtering_evolution.pdf", dpi=300, bbox_inches="tight")
plt.show()
```

### Publication-Ready Settings

```python
# Use the paper theme for journal-quality output
plot_particle_evolution(
    result,
    true_state=true_states,
    theme="paper",
    figsize=(6.5, 3.5),  # Single-column width
    marker_size=1,
    alpha=0.15,
)
```
