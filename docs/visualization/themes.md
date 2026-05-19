---
title: Themes & Customization
description: Built-in themes, custom styles, color palettes, accessibility, and export settings for particlefilterbox plots
---

# Themes & Customization

The `particlefilterbox.viz` module includes a **theming system** that provides consistent, publication-ready styling across all plot functions. Themes control colors, fonts, line widths, and layout — so you can switch between journal, presentation, and dashboard contexts with a single parameter.

```python
from particlefilterbox.viz import set_theme, get_theme, reset_theme
```

---

## Quick Start

```python
from particlefilterbox.viz import set_theme, plot_filtered_state

# Apply a theme globally
set_theme("paper")

# All subsequent plots use the paper theme
plot_filtered_state(result)

# Or apply per-plot
plot_filtered_state(result, theme="presentation")
```

---

## Built-in Themes

### Overview

| Theme | Font Size | Line Width | Background | Best For |
|-------|-----------|------------|------------|----------|
| `"default"` | 12 | 1.5 | White | General use, exploration |
| `"paper"` | 10 | 1.0 | White | Journal submissions, LaTeX |
| `"presentation"` | 16 | 2.5 | White | Slides, talks, posters |
| `"dark"` | 12 | 1.5 | `#1a1a2e` | Dashboards, dark-mode apps |

---

### `"default"` Theme

Clean, minimal style with the indigo-based particlefilterbox palette. Good for everyday exploration and reports.

```python
set_theme("default")
```

**Key properties:**

| Property | Value |
|----------|-------|
| Font family | sans-serif |
| Base font size | 12 |
| Axes line width | 1.0 |
| Line width | 1.5 |
| Primary color | `#4051B5` (indigo) |
| Grid | Light gray, dashed |
| Legend | Frame on, top-right |

---

### `"paper"` Theme

Optimized for **journal submissions** and LaTeX documents. Uses serif fonts, thinner lines, and a black-and-white-friendly palette.

```python
set_theme("paper")
```

**Key properties:**

| Property | Value |
|----------|-------|
| Font family | serif (Computer Modern if available) |
| Base font size | 10 |
| Axes line width | 0.8 |
| Line width | 1.0 |
| Primary colors | Black, dark gray, medium gray |
| Grid | Off by default |
| Legend | No frame, best location |
| Math text | `"cm"` (Computer Modern) |

!!! tip "LaTeX integration"
    The `"paper"` theme uses fonts compatible with LaTeX documents. For exact font matching, enable matplotlib's LaTeX renderer:

    ```python
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = True
    set_theme("paper")
    ```

---

### `"presentation"` Theme

**Larger fonts**, bolder lines, and high-contrast colors for readability at a distance.

```python
set_theme("presentation")
```

**Key properties:**

| Property | Value |
|----------|-------|
| Font family | sans-serif |
| Base font size | 16 |
| Axes line width | 1.5 |
| Line width | 2.5 |
| Primary color | `#4051B5` |
| Grid | Medium gray, solid |
| Legend | Large font, frame on |
| Title size | 20 |

!!! info "Beamer and PowerPoint"
    Use `figsize=(10, 6)` for 16:9 slides or `figsize=(8, 6)` for 4:3. These sizes fill the slide without excessive white space.

---

### `"dark"` Theme

Dark background with bright accent colors for **dashboards** and dark-mode applications.

```python
set_theme("dark")
```

**Key properties:**

| Property | Value |
|----------|-------|
| Font family | sans-serif |
| Base font size | 12 |
| Background | `#1a1a2e` |
| Axes background | `#16213e` |
| Text color | `#e0e0e0` |
| Primary color | `#00d2ff` (cyan) |
| Grid | `#333366`, dashed |
| Line width | 1.5 |

---

## Using Themes

### Global Theme

Set a theme that applies to **all subsequent** plot calls:

```python
from particlefilterbox.viz import set_theme

set_theme("paper")

# All plots now use the paper theme
plot_filtered_state(result)
plot_ess_over_time(result)
plot_trace(chain)
```

### Per-Plot Theme

Override the global theme for a single plot:

```python
# Global theme is "default"
set_theme("default")

# This one plot uses "presentation"
plot_filtered_state(result, theme="presentation")

# Back to "default" for subsequent calls
plot_ess_over_time(result)
```

### Get Current Theme

```python
from particlefilterbox.viz import get_theme

current = get_theme()
print(current)  # "default"
```

### Reset to Default

```python
from particlefilterbox.viz import reset_theme

reset_theme()  # Restores "default" and resets matplotlib rcParams
```

---

## Custom Themes

### From a Dictionary

Pass a dictionary of matplotlib `rcParams` to use a fully custom theme:

```python
my_theme = {
    "font.family": "monospace",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#333333",
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.prop_cycle": plt.cycler(color=[
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#264653"
    ]),
}

# Apply globally
set_theme(my_theme)

# Or per-plot
plot_filtered_state(result, theme=my_theme)
```

### Extending a Built-in Theme

Start from a built-in theme and override specific properties:

```python
from particlefilterbox.viz import get_theme_params

# Get the full rcParams dict for "paper"
paper_params = get_theme_params("paper")

# Modify a few properties
paper_params["font.size"] = 9
paper_params["axes.prop_cycle"] = plt.cycler(color=["#000000", "#888888", "#cccccc"])

set_theme(paper_params)
```

### Registering a Named Theme

Register a custom theme so it can be referenced by name:

```python
from particlefilterbox.viz import register_theme

corporate_theme = {
    "font.family": "Helvetica",
    "font.size": 13,
    "axes.prop_cycle": plt.cycler(color=["#0033a0", "#ff6600", "#00994c"]),
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
}

register_theme("corporate", corporate_theme)

# Now use by name
set_theme("corporate")
plot_filtered_state(result, theme="corporate")
```

---

## Color Palettes

### Default Palette

The particlefilterbox default palette is designed for clarity and distinctiveness:

| Element | Color | Hex |
|---------|-------|-----|
| Primary (states, estimates) | Indigo | `#4051B5` |
| Secondary (observations) | Deep Orange | `#FF5722` |
| Tertiary (true state) | Black | `#212121` |
| Particles (low weight) | Light Blue | `#90CAF9` |
| Particles (high weight) | Deep Purple | `#7C4DFF` |
| Credible bands | Indigo (alpha) | `#4051B5` @ 20% |
| ESS threshold | Red | `#F44336` |
| Grid | Gray | `#E0E0E0` |

### Categorical Palettes

For plots requiring distinct colors (multiple chains, filters, regimes):

```python
from particlefilterbox.viz import get_palette

# Get a list of N distinct colors
colors = get_palette(n=5)
# ['#4051B5', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0']

# Named palettes
colors = get_palette("warm", n=4)
colors = get_palette("cool", n=4)
colors = get_palette("colorblind", n=8)
```

### Colorblind-Friendly Palettes

All default palettes are tested against the three main types of color vision deficiency (deuteranopia, protanopia, tritanopia).

```python
from particlefilterbox.viz import get_palette

# Explicitly request a colorblind-safe palette
colors = get_palette("colorblind", n=8)
```

The `"colorblind"` palette is based on the [Wong (2011)](https://doi.org/10.1038/nmeth.1618) palette, optimized for universal readability:

| Index | Color | Hex | Name |
|-------|-------|-----|------|
| 0 | Black | `#000000` | Black |
| 1 | Orange | `#E69F00` | Orange |
| 2 | Sky Blue | `#56B4E9` | Sky Blue |
| 3 | Bluish Green | `#009E73` | Teal |
| 4 | Yellow | `#F0E442` | Yellow |
| 5 | Blue | `#0072B2` | Blue |
| 6 | Vermillion | `#D55E00` | Red-Orange |
| 7 | Reddish Purple | `#CC79A7` | Pink |

!!! warning "Accessibility"
    When publishing figures, always consider readers with color vision deficiency. Use the `"colorblind"` palette or combine color with other visual channels (line style, markers, annotations) to ensure all information is accessible.

```python
# Best practice: combine color with line style
import matplotlib.pyplot as plt

styles = ["-", "--", "-.", ":"]
colors = get_palette("colorblind", n=4)

for i, (name, chain) in enumerate(chains.items()):
    plt.plot(chain, color=colors[i], linestyle=styles[i], label=name)
```

---

## Font Customization

### Setting Fonts

```python
set_theme({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
```

### Math Fonts

```python
# Computer Modern (matches LaTeX default)
set_theme({"mathtext.fontset": "cm"})

# STIX (matches Times New Roman math)
set_theme({"mathtext.fontset": "stix"})

# Use actual LaTeX rendering (requires LaTeX installation)
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
```

---

## Size Customization

### Figure Sizes

Common figure sizes for different contexts:

| Context | Size (inches) | Aspect Ratio |
|---------|---------------|--------------|
| Single-column journal | `(3.5, 2.5)` | ~1.4:1 |
| Double-column journal | `(7.0, 4.5)` | ~1.6:1 |
| Full-page journal | `(7.0, 9.0)` | ~0.8:1 |
| 16:9 slide | `(10, 5.6)` | 16:9 |
| 4:3 slide | `(8, 6)` | 4:3 |
| Dashboard widget | `(5, 3)` | ~1.7:1 |

```python
# Single-column figure for a journal
plot_filtered_state(result, theme="paper", figsize=(3.5, 2.5))

# Full-width slide figure
plot_filtered_state(result, theme="presentation", figsize=(10, 5.6))
```

### DPI Settings

```python
# Screen resolution (default)
fig.savefig("plot.png", dpi=150)

# Print resolution
fig.savefig("plot.png", dpi=300)

# High-quality poster
fig.savefig("plot.png", dpi=600)
```

---

## matplotlib rcParams Integration

The theming system is a thin wrapper around matplotlib's `rcParams`. Any valid rcParam can be used:

```python
import matplotlib.pyplot as plt

# Direct rcParams access (affects all matplotlib plots)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

# particlefilterbox themes compose with existing rcParams
set_theme("paper")  # Overrides only theme-managed keys
```

!!! info "Scope of themes"
    `set_theme()` only modifies the rcParams it explicitly defines. Any rcParams you set manually before calling `set_theme()` will persist unless the theme overrides them. Use `reset_theme()` to restore all rcParams to matplotlib defaults.

---

## Export

### Supported Formats

| Format | Type | Best For |
|--------|------|----------|
| PNG | Raster | Reports, web, notebooks |
| PDF | Vector | Journal submission, LaTeX `\includegraphics` |
| SVG | Vector | Web pages, interactive dashboards |
| EPS | Vector | Legacy LaTeX workflows |
| TIFF | Raster | Some journal requirements |

### Export Examples

```python
fig, ax = plot_filtered_state(result, theme="paper", show=False)

# PDF for LaTeX (vector, no quality loss at any zoom)
fig.savefig("figure1.pdf", bbox_inches="tight")

# High-resolution PNG
fig.savefig("figure1.png", dpi=300, bbox_inches="tight")

# SVG for web embedding
fig.savefig("figure1.svg", bbox_inches="tight")

# TIFF (some journals require this)
fig.savefig("figure1.tiff", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
```

### Batch Export Utility

```python
from particlefilterbox.viz import export_figure

# Export to multiple formats at once
fig, ax = plot_filtered_state(result, show=False)

export_figure(
    fig,
    filename="filtered_state",
    formats=["pdf", "png", "svg"],
    dpi=300,
    output_dir="figures/",
)
# Creates: figures/filtered_state.pdf, .png, .svg
```

### Configuring Default Export

```python
from particlefilterbox.viz import set_export_defaults

set_export_defaults(
    dpi=300,
    bbox_inches="tight",
    transparent=False,
    facecolor="white",
)
```

---

## Plotly Backend Themes

When using the plotly backend, themes map to plotly templates:

| particlefilterbox Theme | Plotly Template |
|------------------------|-----------------|
| `"default"` | `"plotly_white"` |
| `"paper"` | `"simple_white"` |
| `"presentation"` | `"plotly_white"` + large fonts |
| `"dark"` | `"plotly_dark"` |

```python
fig = plot_filtered_state(result, backend="plotly", theme="dark")
fig.show()
```

!!! tip "Interactive exports"
    Plotly figures can be exported as interactive HTML:

    ```python
    fig = plot_filtered_state(result, backend="plotly")
    fig.write_html("interactive_plot.html")
    ```

---

## Complete Example

A full workflow combining themes, customization, and export:

```python
import matplotlib.pyplot as plt
from particlefilterbox import BootstrapFilter
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.viz import (
    set_theme,
    get_palette,
    plot_filtered_state,
    plot_ess_over_time,
    plot_volatility,
    export_figure,
)

# Set publication theme
set_theme("paper")

# Setup and run
model = StochasticVolatility(phi=0.97, sigma=0.15, beta=0.65)
true_states, observations = model.simulate(T=500, seed=42)

pf = BootstrapFilter(model, n_particles=1000)
result = pf.filter(observations)

# Create a multi-panel figure
fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True)

plot_filtered_state(result, true_state=true_states, ax=axes[0], show=False)
plot_ess_over_time(result, ax=axes[1], show=False)
plot_volatility(result, returns=observations, ax=axes[2], show=False)

axes[0].set_title("Filtered Log-Volatility")
axes[1].set_title("Effective Sample Size")
axes[2].set_title("Volatility with Returns")

fig.tight_layout()

# Export to multiple formats
export_figure(fig, "sv_analysis", formats=["pdf", "png"], dpi=300, output_dir="figures/")
plt.show()
```
