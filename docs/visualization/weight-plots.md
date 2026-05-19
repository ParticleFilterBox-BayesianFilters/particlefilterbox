---
title: Weight Plots
description: Visualize particle weights, ESS, entropy, and weight degeneracy diagnostics
---

# Weight Plots

Weight plots provide insight into the quality of the particle approximation. Healthy weights are roughly uniform; concentrated weights signal degeneracy. These diagnostics are critical for tuning filters and validating results.

```python
from particlefilterbox.viz import (
    plot_weights,
    plot_weight_evolution,
    plot_weight_histogram,
    plot_ess_over_time,
    plot_max_weight,
    plot_weight_entropy,
)
```

---

## `plot_weights` { #plot_weights }

Bar plot of normalized particle weights at a single time step.

### API

```python
plot_weights(
    result,                   # FilterResult
    t,                        # Time step index
    sort=True,                # Sort bars by weight (descending)
    color="#4051B5",          # Bar color
    highlight_top=5,          # Highlight top-k particles
    highlight_color="#E91E63",
    alpha=0.8,
    ax=None,
    figsize=(10, 4),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output |
| `t` | `int` | required | Time step index |
| `sort` | `bool` | `True` | Sort particles by weight in descending order |
| `highlight_top` | `int` | `5` | Number of highest-weight particles to highlight |
| `highlight_color` | `str` | `"#E91E63"` | Color for highlighted particles |

### Example

```python
from particlefilterbox.viz import plot_weights

fig, ax = plot_weights(result, t=100, sort=True, highlight_top=10)
ax.axhline(y=1/result.n_particles, color="gray", linestyle="--",
           label=f"Uniform = {1/result.n_particles:.4f}")
ax.legend()
```

!!! note "Output"
    A bar chart where each bar represents one particle's normalized weight $w_t^{(i)}$. When sorted, a sharp drop-off indicates weight concentration. The dashed horizontal line at $1/N$ marks the uniform (ideal) weight. Bars highlighted in pink are the top-$k$ heaviest particles.

---

## `plot_weight_evolution` { #plot_weight_evolution }

Heatmap of particle weights across time, revealing temporal patterns in weight concentration.

### API

```python
plot_weight_evolution(
    result,                   # FilterResult
    cmap="hot",               # Colormap
    log_scale=False,          # Use log-scale for colors
    sort_particles=True,      # Sort by weight at each time step
    ax=None,
    figsize=(12, 6),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output |
| `cmap` | `str` | `"hot"` | Colormap for the heatmap |
| `log_scale` | `bool` | `False` | Apply $\log$ transform to weights for visualization |
| `sort_particles` | `bool` | `True` | Sort particles by weight at each time step |

### Example

```python
from particlefilterbox.viz import plot_weight_evolution

fig, ax = plot_weight_evolution(
    result,
    cmap="hot",
    log_scale=True,
    sort_particles=True,
)
```

!!! note "Output"
    A 2D heatmap with time on the x-axis and particle index on the y-axis. Bright regions indicate high-weight particles. With `sort_particles=True`, the top row always holds the highest weight. Horizontal bright streaks suggest persistent dominance by a few particles; uniform color bands indicate healthy weight distributions.

---

## `plot_weight_histogram` { #plot_weight_histogram }

Distribution of weights across all particles, useful for assessing overall degeneracy.

### API

```python
plot_weight_histogram(
    result,                   # FilterResult
    t=None,                   # Time step (None = aggregate over all t)
    bins=50,
    log_scale=True,           # Log-scale x-axis
    density=True,
    color="#4051B5",
    alpha=0.7,
    ax=None,
    figsize=(8, 5),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output |
| `t` | `int \| None` | `None` | Time step; `None` aggregates across all steps |
| `log_scale` | `bool` | `True` | Logarithmic x-axis for weight values |
| `density` | `bool` | `True` | Normalize to density |

### Example

```python
from particlefilterbox.viz import plot_weight_histogram

# Compare weight distributions at two time points
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_weight_histogram(result, t=10, ax=axes[0], show=False)
axes[0].set_title("$t = 10$ (after resampling)")

plot_weight_histogram(result, t=50, ax=axes[1], show=False)
axes[1].set_title("$t = 50$ (before resampling)")

fig.tight_layout()
plt.show()
```

!!! note "Output"
    A histogram of weight values. With `log_scale=True`, the x-axis is $\log w_t^{(i)}$. A distribution concentrated near $\log(1/N)$ is healthy. A long left tail indicates many near-zero weights (degeneracy).

---

## `plot_ess_over_time` { #plot_ess_over_time }

Effective Sample Size trajectory over time with an optional resampling threshold.

The ESS is defined as:

$$
\text{ESS}_t = \frac{1}{\sum_{i=1}^{N} \left(w_t^{(i)}\right)^2}
$$

### API

```python
plot_ess_over_time(
    result,                   # FilterResult
    threshold=None,           # Resampling threshold (float or "auto")
    normalize=True,           # Show ESS/N instead of raw ESS
    color="#4051B5",
    threshold_color="#E91E63",
    fill=True,                # Fill area under ESS curve
    fill_alpha=0.15,
    ax=None,
    figsize=(10, 4),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output |
| `threshold` | `float \| str \| None` | `None` | Resampling threshold; `"auto"` reads from result metadata |
| `normalize` | `bool` | `True` | Display ESS$/N$ ratio ($0$--$1$) instead of raw ESS |
| `fill` | `bool` | `True` | Fill area under the ESS curve |

### Example

```python
from particlefilterbox.viz import plot_ess_over_time

fig, ax = plot_ess_over_time(
    result,
    threshold=0.5,
    normalize=True,
    fill=True,
    color="#1976D2",
    threshold_color="#D32F2F",
)
```

!!! note "Output"
    A line plot of ESS$/N$ over time. The dashed red line marks the resampling threshold. Time steps where ESS drops below the threshold trigger resampling. Sustained low ESS indicates the proposal distribution is a poor match for the target.

!!! warning "Interpreting ESS"
    ESS$/N > 0.5$ generally indicates a healthy filter. Persistent ESS$/N < 0.2$ signals severe weight degeneracy -- consider using a better proposal (auxiliary PF, guided PF) or increasing $N$.

---

## `plot_max_weight` { #plot_max_weight }

Maximum normalized weight over time. A complementary diagnostic to ESS.

### API

```python
plot_max_weight(
    result,                   # FilterResult
    color="#4051B5",
    threshold=None,           # Optional threshold line
    ax=None,
    figsize=(10, 4),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output |
| `threshold` | `float \| None` | `None` | Optional horizontal threshold line |

### Example

```python
from particlefilterbox.viz import plot_max_weight

fig, ax = plot_max_weight(
    result,
    color="#FF6F00",
    threshold=0.1,
)
ax.set_ylabel(r"$\max_i \, w_t^{(i)}$")
```

!!! note "Output"
    A line plot of $\max_i w_t^{(i)}$ over time. Values near $1/N$ indicate uniform weights. A maximum weight approaching $1.0$ means a single particle dominates the entire approximation.

---

## `plot_weight_entropy` { #plot_weight_entropy }

Entropy of the normalized weight distribution over time, defined as:

$$
H_t = -\sum_{i=1}^{N} w_t^{(i)} \log w_t^{(i)}
$$

Maximum entropy ($\log N$) corresponds to uniform weights.

### API

```python
plot_weight_entropy(
    result,                   # FilterResult
    normalize=True,           # Show H_t / log(N)
    color="#4051B5",
    fill=True,
    fill_alpha=0.15,
    ax=None,
    figsize=(10, 4),
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output |
| `normalize` | `bool` | `True` | Display $H_t / \log N$ (range $0$--$1$) |
| `fill` | `bool` | `True` | Fill area under the entropy curve |

### Example

```python
from particlefilterbox.viz import plot_weight_entropy

fig, ax = plot_weight_entropy(
    result,
    normalize=True,
    color="#00897B",
    fill=True,
)
```

!!! note "Output"
    A line plot of normalized weight entropy $H_t / \log N$ over time. Values near $1.0$ indicate uniform weights (healthy). Values near $0.0$ indicate extreme concentration (degenerate). Entropy provides a smoother diagnostic than ESS and is particularly useful for detecting gradual weight collapse.

---

## Customization

### Diagnostic Dashboard

```python
import matplotlib.pyplot as plt
from particlefilterbox.viz import (
    plot_ess_over_time,
    plot_max_weight,
    plot_weight_entropy,
    plot_weight_evolution,
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_ess_over_time(result, threshold=0.5, ax=axes[0, 0], show=False)
plot_max_weight(result, ax=axes[0, 1], show=False)
plot_weight_entropy(result, ax=axes[1, 0], show=False)
plot_weight_evolution(result, cmap="hot", ax=axes[1, 1], show=False)

axes[0, 0].set_title("ESS / N")
axes[0, 1].set_title("Max Weight")
axes[1, 0].set_title("Weight Entropy")
axes[1, 1].set_title("Weight Heatmap")

fig.suptitle("Weight Diagnostics Dashboard", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig("weight_diagnostics.pdf", dpi=300, bbox_inches="tight")
plt.show()
```

### Comparing Filters

```python
import matplotlib.pyplot as plt
from particlefilterbox.viz import plot_ess_over_time

fig, ax = plt.subplots(figsize=(12, 5))

plot_ess_over_time(result_bootstrap, ax=ax, color="#1976D2",
                   label="Bootstrap", show=False)
plot_ess_over_time(result_auxiliary, ax=ax, color="#E91E63",
                   label="Auxiliary", show=False)
plot_ess_over_time(result_guided, ax=ax, color="#4CAF50",
                   label="Guided", show=False)

ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
ax.legend()
ax.set_title("ESS Comparison Across Filters")
plt.tight_layout()
plt.show()
```

### Publication-Ready Settings

```python
plot_ess_over_time(
    result,
    threshold=0.5,
    theme="paper",
    figsize=(6.5, 3.0),
    fill_alpha=0.1,
)
```
