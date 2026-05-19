---
title: Model-Specific Plots
description: Specialized visualization functions for Stochastic Volatility, DSGE, Regime-Switching, and Jump-Diffusion models
---

# Model-Specific Plots

Beyond general-purpose particle and state plots, `particlefilterbox.viz` provides **model-specific visualization functions** tailored to the unique structure of each supported model class. These functions extract and display the quantities that practitioners actually inspect for each model type.

```python
from particlefilterbox.viz import (
    # Stochastic Volatility
    plot_volatility,
    plot_volatility_surface,
    plot_leverage,
    # DSGE
    plot_impulse_response,
    plot_shock_decomposition,
    # Regime-Switching
    plot_regime_probabilities,
    plot_regime_transitions,
    # Jump-Diffusion
    plot_jump_detection,
    plot_jump_intensity,
)
```

---

## Stochastic Volatility Plots

### `plot_volatility` { #plot_volatility }

Plots the filtered log-volatility (or exponentiated volatility) alongside the observed returns series.

#### API

```python
plot_volatility(
    result,                   # FilterResult from an SV model
    returns=None,             # Original return series (overlaid as bars)
    exponentiate=True,        # Plot exp(h_t/2) instead of h_t
    quantiles=(0.05, 0.95),   # Credible interval
    vol_color="#4051B5",      # Volatility line color
    return_color="#90A4AE",   # Return bar color
    ax=None,
    figsize=(12, 5),
    **kwargs,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output from an SV model |
| `returns` | `np.ndarray \| None` | `None` | Observed returns to overlay as a bar chart |
| `exponentiate` | `bool` | `True` | Show $\sigma_t = \exp(h_t / 2)$ instead of raw $h_t$ |
| `quantiles` | `tuple[float, float]` | `(0.05, 0.95)` | Lower and upper quantiles for the credible band |
| `vol_color` | `str` | `"#4051B5"` | Color for the volatility line |
| `return_color` | `str` | `"#90A4AE"` | Color for return bars |

#### Example

```python
from particlefilterbox.models import StochasticVolatility
from particlefilterbox import BootstrapFilter
from particlefilterbox.viz import plot_volatility

model = StochasticVolatility(phi=0.97, sigma=0.15, beta=0.65)
true_states, observations = model.simulate(T=500, seed=42)

pf = BootstrapFilter(model, n_particles=1000)
result = pf.filter(observations)

fig, ax = plot_volatility(
    result,
    returns=observations,
    exponentiate=True,
    quantiles=(0.05, 0.95),
)
```

!!! note "Output"
    **Top layer**: a solid line showing the filtered volatility $\sigma_t = \exp(h_t / 2)$ with a shaded 90% credible band. **Bottom layer**: faint gray bars representing observed returns $y_t$. Volatility spikes align with periods of large return magnitude, confirming the model captures clustering.

---

### `plot_volatility_surface` { #plot_volatility_surface }

Displays the full posterior distribution of volatility across time as a heatmap or fan chart across quantiles.

#### API

```python
plot_volatility_surface(
    result,                   # FilterResult from an SV model
    quantile_levels=None,     # List of quantiles; default: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    cmap="Blues",             # Colormap for the surface
    exponentiate=True,        # Plot exp(h_t/2) scale
    ax=None,
    figsize=(12, 5),
    **kwargs,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output from an SV model |
| `quantile_levels` | `list[float] \| None` | `None` | Quantile levels for the fan chart (symmetric pairs recommended) |
| `cmap` | `str` | `"Blues"` | Colormap for shading between quantile bands |
| `exponentiate` | `bool` | `True` | Show $\exp(h_t / 2)$ scale |

#### Example

```python
from particlefilterbox.viz import plot_volatility_surface

fig, ax = plot_volatility_surface(
    result,
    quantile_levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    cmap="Purples",
)
```

!!! note "Output"
    A fan chart where each band corresponds to a quantile interval. The median ($q = 0.5$) appears as the darkest central line. Wider bands indicate greater posterior uncertainty about the current volatility level. Useful for identifying periods where the filter is confident vs. uncertain.

---

### `plot_leverage` { #plot_leverage }

Visualizes the **leverage effect** — the correlation between returns and future volatility changes — from the filtered posterior.

#### API

```python
plot_leverage(
    result,                   # FilterResult from an SV model
    returns=None,             # Observed returns
    lag=1,                    # Lag for correlation computation
    window=50,                # Rolling window size
    scatter=True,             # Show scatter of (return, Δvol) pairs
    regression=True,          # Overlay OLS regression line
    ax=None,
    figsize=(8, 6),
    **kwargs,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output from an SV model |
| `returns` | `np.ndarray \| None` | `None` | Observed returns; extracted from result if `None` |
| `lag` | `int` | `1` | Number of periods to lag volatility changes |
| `window` | `int` | `50` | Rolling window for time-varying correlation |
| `scatter` | `bool` | `True` | Show scatter plot of return vs. $\Delta h_{t+1}$ |
| `regression` | `bool` | `True` | Overlay least-squares regression line |

#### Example

```python
from particlefilterbox.viz import plot_leverage

fig, ax = plot_leverage(
    result,
    returns=observations,
    window=60,
)
```

!!! note "Output"
    **Left panel** (if `scatter=True`): scatter of $(y_t, \Delta h_{t+1})$ pairs with a regression line. A negative slope confirms the leverage effect — negative returns precede volatility increases. **Right panel**: rolling correlation over time, showing how the leverage relationship varies across the sample.

---

## DSGE Plots

### `plot_impulse_response` { #plot_impulse_response }

Plots impulse response functions (IRFs) from the posterior distribution of a DSGE model estimated via PMCMC.

#### API

```python
plot_impulse_response(
    model,                    # DSGE model specification
    chain,                    # PMCMC chain (or array of parameter draws)
    shock="all",              # Shock name or "all"
    variables=None,           # Variables to plot; None = all observed
    horizon=40,               # IRF horizon (periods)
    n_draws=200,              # Number of posterior draws to compute IRFs
    quantiles=(0.05, 0.5, 0.95),  # Quantiles for the posterior IRF band
    subplot_layout="grid",    # "grid" or "vertical"
    ax=None,
    figsize=(14, 8),
    **kwargs,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `DSGEModel` | required | DSGE model with `compute_irf` method |
| `chain` | `np.ndarray` | required | Parameter draws from PMCMC, shape `(n_iter, n_params)` |
| `shock` | `str` | `"all"` | Name of the structural shock, or `"all"` for all shocks |
| `variables` | `list[str] \| None` | `None` | Subset of model variables to display |
| `horizon` | `int` | `40` | Number of periods for the IRF |
| `n_draws` | `int` | `200` | Posterior draws used to compute IRF bands |
| `quantiles` | `tuple` | `(0.05, 0.5, 0.95)` | Quantile levels for the IRF envelope |

#### Example

```python
from particlefilterbox.viz import plot_impulse_response

# After PMCMC estimation of a DSGE model
fig, axes = plot_impulse_response(
    model=dsge_model,
    chain=pmcmc_result.chain,
    shock="monetary",
    variables=["output", "inflation", "interest_rate"],
    horizon=40,
    n_draws=500,
    quantiles=(0.1, 0.5, 0.9),
)
```

!!! note "Output"
    A grid of subplots, one per variable. Each subplot shows the median IRF as a solid line with shaded credible bands. The zero line is marked as a dashed reference. For a monetary shock, output typically shows a hump-shaped decline, inflation falls, and the interest rate rises on impact before reverting.

!!! tip "Comparing shocks"
    Use `shock="all"` to generate a full IRF matrix (shocks $\times$ variables), which is the standard presentation in DSGE papers.

---

### `plot_shock_decomposition` { #plot_shock_decomposition }

Displays the historical contribution of each structural shock to the observed variables over time.

#### API

```python
plot_shock_decomposition(
    result,                   # FilterResult or SmootherResult from DSGE
    variables=None,           # Variables to decompose; None = all
    stacked=True,             # Stacked area chart vs. line chart
    cmap="tab10",             # Colormap for shocks
    ax=None,
    figsize=(14, 6),
    **kwargs,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Result containing smoothed shocks |
| `variables` | `list[str] \| None` | `None` | Variables to display |
| `stacked` | `bool` | `True` | Use stacked area plot |
| `cmap` | `str` | `"tab10"` | Colormap for shock categories |

#### Example

```python
from particlefilterbox.viz import plot_shock_decomposition

fig, ax = plot_shock_decomposition(
    result=smoother_result,
    variables=["output"],
    stacked=True,
)
```

!!! note "Output"
    A stacked area chart where each color represents a structural shock (technology, monetary, demand, etc.). The sum of all areas equals the observed deviation of the variable from steady state. This plot answers "which shocks drove output during the recession?" Dominant colors during specific periods reveal the main drivers.

---

## Regime-Switching Plots

### `plot_regime_probabilities` { #plot_regime_probabilities }

Plots the filtered (or smoothed) probabilities of each regime over time.

#### API

```python
plot_regime_probabilities(
    result,                   # FilterResult from a regime-switching model
    smoothed=False,           # Use smoothed probabilities if available
    regime_names=None,        # Custom names for regimes
    colors=None,              # Custom colors per regime
    threshold=0.5,            # Threshold line for binary regime identification
    shade_regimes=True,       # Shade background by most-probable regime
    ax=None,
    figsize=(12, 4),
    **kwargs,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output from a regime-switching model |
| `smoothed` | `bool` | `False` | Use Kim-smoother probabilities instead of filtered |
| `regime_names` | `list[str] \| None` | `None` | Labels for each regime (e.g., `["Expansion", "Recession"]`) |
| `colors` | `list[str] \| None` | `None` | Colors for each regime probability line |
| `threshold` | `float` | `0.5` | Horizontal reference line |
| `shade_regimes` | `bool` | `True` | Color the background by the most likely regime |

#### Example

```python
from particlefilterbox.viz import plot_regime_probabilities

fig, ax = plot_regime_probabilities(
    result,
    regime_names=["Expansion", "Recession"],
    colors=["#4CAF50", "#F44336"],
    shade_regimes=True,
)
```

!!! note "Output"
    Two probability lines (summing to 1.0 at each $t$) with the background shaded green for expansion and red for recession. The 0.5 threshold line helps identify regime switches. Sharp transitions indicate clear regime changes; gradual crossings suggest ambiguity in the data.

---

### `plot_regime_transitions` { #plot_regime_transitions }

Visualizes the estimated transition probability matrix as a heatmap or directed graph.

#### API

```python
plot_regime_transitions(
    result,                   # FilterResult or transition matrix directly
    regime_names=None,        # Custom regime labels
    style="heatmap",          # "heatmap" or "graph"
    annotate=True,            # Show probability values in cells
    cmap="YlOrRd",            # Colormap for heatmap
    fmt=".3f",                # Number format for annotations
    ax=None,
    figsize=(6, 5),
    **kwargs,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult \| np.ndarray` | required | Filter result or a $K \times K$ transition matrix |
| `regime_names` | `list[str] \| None` | `None` | Labels for rows/columns |
| `style` | `str` | `"heatmap"` | `"heatmap"` for matrix view, `"graph"` for directed graph |
| `annotate` | `bool` | `True` | Show probability values |
| `cmap` | `str` | `"YlOrRd"` | Colormap for the heatmap |
| `fmt` | `str` | `".3f"` | Format string for probability annotations |

#### Example

```python
from particlefilterbox.viz import plot_regime_transitions

# Heatmap view
fig, ax = plot_regime_transitions(
    result,
    regime_names=["Low Vol", "High Vol", "Crisis"],
    style="heatmap",
    cmap="Blues",
)
```

!!! note "Output"
    A $K \times K$ heatmap where rows represent the current regime and columns represent the next regime. Diagonal entries (persistence probabilities) are typically the largest. Off-diagonal entries reveal how likely transitions are between regimes. For a 3-regime model, the matrix is $3 \times 3$ with each row summing to 1.0.

!!! tip "Graph view"
    Use `style="graph"` to render a directed graph where nodes are regimes and edge widths are proportional to transition probabilities. This is particularly intuitive for presentations.

---

## Jump-Diffusion Plots

### `plot_jump_detection` { #plot_jump_detection }

Identifies and highlights detected jumps in the posterior, overlaid on the observation series.

#### API

```python
plot_jump_detection(
    result,                   # FilterResult from a jump-diffusion model
    observations=None,        # Observed series for overlay
    jump_threshold=0.5,       # Posterior probability threshold for jump detection
    marker="v",               # Marker style for detected jumps
    marker_size=100,          # Marker size for jump markers
    jump_color="#F44336",     # Color for jump markers
    obs_color="#90A4AE",      # Color for observation line
    ax=None,
    figsize=(12, 4),
    **kwargs,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output from a jump-diffusion model |
| `observations` | `np.ndarray \| None` | `None` | Observed data series |
| `jump_threshold` | `float` | `0.5` | Posterior $P(\text{jump}_t) >$ threshold to flag |
| `marker` | `str` | `"v"` | Matplotlib marker style for detected jumps |
| `marker_size` | `float` | `100` | Size of jump markers |
| `jump_color` | `str` | `"#F44336"` | Color for jump markers |

#### Example

```python
from particlefilterbox.viz import plot_jump_detection

fig, ax = plot_jump_detection(
    result,
    observations=observations,
    jump_threshold=0.5,
    marker="v",
    jump_color="red",
)
```

!!! note "Output"
    A line plot of observations with red downward-pointing triangles ($\blacktriangledown$) marking each time step where the posterior jump probability exceeds the threshold. Jump markers cluster around large return events (e.g., flash crashes, earnings surprises). The density of markers reveals whether the model attributes a move to diffusion or jump.

---

### `plot_jump_intensity` { #plot_jump_intensity }

Plots the time-varying jump intensity $\lambda_t$ (or posterior jump probability) over the sample.

#### API

```python
plot_jump_intensity(
    result,                   # FilterResult from a jump-diffusion model
    observations=None,        # Observed series for dual-axis overlay
    smoothed=False,           # Use smoothed intensity if available
    intensity_color="#7C4DFF", # Color for intensity line
    quantiles=(0.1, 0.9),    # Credible band
    ax=None,
    figsize=(12, 4),
    **kwargs,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FilterResult` | required | Filter output from a jump-diffusion model |
| `observations` | `np.ndarray \| None` | `None` | Observed series on secondary y-axis |
| `smoothed` | `bool` | `False` | Use smoothed jump intensity |
| `intensity_color` | `str` | `"#7C4DFF"` | Color for the intensity line |
| `quantiles` | `tuple[float, float]` | `(0.1, 0.9)` | Credible interval for jump intensity |

#### Example

```python
from particlefilterbox.viz import plot_jump_intensity

fig, ax = plot_jump_intensity(
    result,
    observations=observations,
    quantiles=(0.05, 0.95),
)
```

!!! note "Output"
    **Primary axis**: the filtered jump intensity $\lambda_t$ as a solid purple line with a shaded credible band. **Secondary axis** (right): the observed series in gray. The intensity rises during turbulent periods and falls during calm markets. This plot is key for understanding whether jump risk is time-varying or constant.

!!! warning "Model requirement"
    `plot_jump_intensity` requires a model with time-varying jump intensity (e.g., Hawkes-driven or self-exciting jump processes). For constant-intensity models, the plot displays a flat line at $\hat{\lambda}$ with its posterior uncertainty band.

---

## Composing Model Plots

All model-specific functions follow the same `(fig, ax)` return convention, making them easy to compose:

```python
import matplotlib.pyplot as plt
from particlefilterbox.viz import (
    plot_volatility,
    plot_leverage,
    plot_volatility_surface,
)

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

plot_volatility(result, returns=observations, ax=axes[0], show=False)
plot_volatility_surface(result, ax=axes[1], show=False)
plot_leverage(result, returns=observations, ax=axes[2], show=False)

axes[0].set_title("Filtered Volatility with Returns")
axes[1].set_title("Volatility Quantile Surface")
axes[2].set_title("Leverage Effect")

fig.tight_layout()
plt.savefig("sv_diagnostic_panel.pdf", dpi=300, bbox_inches="tight")
plt.show()
```

---

## Export and Resolution

All model plots support high-resolution export for publication:

```python
# Save as PDF (vector graphics, ideal for journals)
fig, ax = plot_volatility(result, returns=observations, show=False)
fig.savefig("volatility.pdf", dpi=300, bbox_inches="tight")

# Save as SVG (vector, ideal for web)
fig.savefig("volatility.svg", bbox_inches="tight")

# Save as PNG (raster, high resolution)
fig.savefig("volatility.png", dpi=600, bbox_inches="tight")
```

!!! tip "Journal-ready figures"
    Use `theme="paper"` with model plots for publication-quality output:

    ```python
    plot_volatility(result, returns=observations, theme="paper", figsize=(6.5, 3.5))
    ```

    The `"paper"` theme uses black/white with markers and LaTeX-compatible fonts, matching typical journal requirements.
