---
title: Installation
description: Install particlefilterbox and its dependencies in any Python environment
---

# Installation

## Quick Install

```bash
pip install particlefilterbox
```

That's it. particlefilterbox and all core dependencies (NumPy, SciPy, Pandas) will be installed automatically.

## Requirements

**Python**: >= 3.11 (3.11, 3.12 supported)

**Core dependencies** (installed automatically):

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| NumPy | >= 1.24 | Array operations and particle storage |
| SciPy | >= 1.10 | Statistical distributions and resampling |
| Pandas | >= 2.0 | Data handling and results output |

## Installation Options

=== "pip (Recommended)"

    ```bash
    pip install particlefilterbox
    ```

    Upgrade to the latest version:

    ```bash
    pip install --upgrade particlefilterbox
    ```

=== "From Source"

    ```bash
    git clone https://github.com/nodesecon/particlefilterbox.git
    cd particlefilterbox
    pip install -e .
    ```

=== "Development Mode"

    ```bash
    git clone https://github.com/nodesecon/particlefilterbox.git
    cd particlefilterbox
    pip install -e ".[dev]"
    ```

### Optional Extras

particlefilterbox supports several optional dependency groups for extended functionality:

```bash
# Visualization support (matplotlib)
pip install particlefilterbox[viz]

# GPU acceleration (CuPy)
pip install particlefilterbox[gpu]

# Numba JIT compilation
pip install particlefilterbox[accel]

# CLI tools
pip install particlefilterbox[cli]

# Everything (viz + cli + accel)
pip install particlefilterbox[all]
```

### Optional Dependencies Table

| Extra | Packages | What It Enables |
|-------|----------|-----------------|
| `[viz]` | matplotlib >= 3.7 | Particle plots, weight histograms, state trajectory charts |
| `[gpu]` | cupy >= 12.0 | GPU-accelerated particle propagation and weight computation |
| `[accel]` | numba >= 0.58 | JIT-compiled transition and observation functions |
| `[cli]` | typer >= 0.9 | Command-line interface (`pfbox filter`, `pfbox estimate`) |
| `[docs]` | mkdocs, mkdocs-material, ... | Documentation building |
| `[dev]` | pytest, ruff, pyright, ... | Testing, linting, type checking |
| `[all]` | viz + cli + accel | Full installation (everything except GPU and docs) |

!!! info "kalmanbox Integration"
    For Rao-Blackwellized Particle Filters (RBPF) and Unscented Particle Filters (UPF),
    you also need [kalmanbox](https://github.com/nodesecon/kalmanbox):

    ```bash
    pip install kalmanbox
    ```

    kalmanbox provides the Kalman filter sub-updates used inside the RBPF's linear
    sub-state and the UPF's sigma-point transformations. See the
    [RBPF Guide](../user-guide/filters/rbpf.md) for details.

## Verify Installation

Verify that particlefilterbox is installed correctly:

```python
import particlefilterbox as pfb

# Check version
print(f"particlefilterbox version: {pfb.__version__}")

# Quick smoke test: simulate and filter
import numpy as np
from particlefilterbox.models.sv import SVModel
from particlefilterbox.filters.bootstrap import BootstrapFilter

# Create a stochastic volatility model
model = SVModel(mu=0.0, phi=0.97, sigma_eta=0.15)

# Simulate data
np.random.seed(42)
states, obs = model.simulate(n_obs=100)

# Run the Bootstrap Particle Filter
pf = BootstrapFilter(model=model, n_particles=500)
results = pf.filter(obs)

print(f"Filtered {len(obs)} observations with {pf.n_particles} particles")
print(f"Final state estimate: {results.filtered_mean[-1]:.4f}")
print(f"Final ESS: {results.ess[-1]:.1f}")
```

Expected output:

```text
particlefilterbox version: 0.x.x
Filtered 100 observations with 500 particles
Final state estimate: -0.xxxx
Final ESS: xxx.x
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'particlefilterbox'`

Ensure particlefilterbox is installed in the active Python environment:

```bash
pip list | grep particlefilterbox
```

If not found, install it. If using Jupyter, make sure the notebook kernel matches the environment where particlefilterbox is installed.

### NumPy / SciPy Conflicts

If you encounter version conflicts with NumPy or SciPy, create a fresh virtual environment:

```bash
python -m venv pfb_env
source pfb_env/bin/activate   # Linux/macOS
pfb_env\Scripts\activate      # Windows
pip install particlefilterbox
```

!!! warning "NumPy 2.x Compatibility"
    particlefilterbox is tested with NumPy >= 1.24. If you encounter issues with
    NumPy 2.x, pin to a compatible version:

    ```bash
    pip install "numpy>=1.24,<2.0" particlefilterbox
    ```

### CUDA / GPU Setup

GPU acceleration requires CuPy with a CUDA-compatible GPU:

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

!!! warning "CUDA Toolkit Required"
    CuPy requires the NVIDIA CUDA Toolkit installed on your system.
    Check your CUDA version with `nvcc --version` and install the matching
    CuPy package. See the [GPU Acceleration Guide](../acceleration/gpu.md) for
    detailed setup instructions.

### kalmanbox Version Compatibility

particlefilterbox works with kalmanbox >= 0.1.0. If you have an older version:

```bash
pip install --upgrade kalmanbox
```

To verify compatibility:

```python
import kalmanbox
print(f"kalmanbox version: {kalmanbox.__version__}")
```

### System Information for Bug Reports

```python
import sys, platform
import particlefilterbox as pfb

print(f"Python:              {sys.version}")
print(f"Platform:            {platform.platform()}")
print(f"particlefilterbox:   {pfb.__version__}")

import numpy, scipy, pandas
print(f"NumPy:               {numpy.__version__}")
print(f"SciPy:               {scipy.__version__}")
print(f"Pandas:              {pandas.__version__}")

try:
    import kalmanbox
    print(f"kalmanbox:           {kalmanbox.__version__}")
except ImportError:
    print("kalmanbox:           not installed")
```

Include this output when [reporting issues](https://github.com/nodesecon/particlefilterbox/issues).

## Next Steps

- **[Quickstart](quickstart.md)** -- Run your first particle filter in 5 minutes
- **[Core Concepts](core-concepts.md)** -- Understand particles, weights, and resampling
- **[Choosing a Filter](choosing-filter.md)** -- Find the right algorithm for your model
