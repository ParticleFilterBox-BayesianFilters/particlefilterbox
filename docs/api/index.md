---
title: "API Reference"
description: "Complete API reference for particlefilterbox — particle filters, SMC, PMCMC, and diagnostics"
---

# API Reference

!!! info "Package"
    **Install**: `pip install particlefilterbox`
    **Import**: `import particlefilterbox as pfb`
    **Source**: [github.com/nodesecon/particlefilterbox](https://github.com/nodesecon/particlefilterbox)

## Overview

The `particlefilterbox` API is organized into modules that follow the natural workflow of Sequential Monte Carlo inference: define a model, choose a filter, run inference, smooth estimates, and diagnose results.

All modules share consistent conventions for parameters, return types, and error handling. Where possible, API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

---

## Module Index

| Module | Description | Key Classes / Functions |
|--------|-------------|------------------------|
| [Core](core.md) | Central data structures and base classes | `ParticleCloud`, `ParticleFilterModel`, `ParticleFilterResults`, `PFConfig` |
| [Resampling](resampling.md) | Resampling algorithms | `systematic_resample`, `multinomial_resample`, `stratified_resample`, `residual_resample` |
| [Filters](filters.md) | Particle filter implementations | `BootstrapPF`, `SIR`, `AuxiliaryPF`, `RaoBlackwellizedPF`, `UnscentedPF`, ... |
| [Smoothers](smoothers.md) | Backward smoothing algorithms | `FFBSm`, `FFBSi`, `TwoFilterSmoother`, `FixedLagSmoother` |
| [SMC](smc.md) | SMC samplers and tempering | `SMCSampler`, `SMCSquared`, `IBIS`, `WasteFreeSMC` |
| [PMCMC](pmcmc.md) | Particle MCMC methods | `PMMH`, `ParticleGibbs`, `PGAS`, `SMC2Online` |
| [Models](models.md) | Pre-built state-space models | `StochasticVolatility`, `DSGE`, `JumpDiffusion`, `CountStateSpace` |
| [Diagnostics](diagnostics.md) | Convergence and degeneracy checks | `ESSMonitor`, `WeightAnalysis`, `ConvergenceStudy` |
| [Acceleration](acceleration.md) | Numba JIT, GPU, parallelization | `enable_numba`, `GPUBackend`, `ParallelRunner` |
| [Visualization](visualization.md) | Plotting functions | `plot_filtered_state`, `plot_trace`, `plot_particle_cloud` |
| [Reports](reports.md) | HTML/LaTeX report generation | `PFReportTransformer`, `PMCMCReportTransformer` |
| [Experiment](experiment.md) | Reproducible experiment framework | `PFExperiment`, `ExperimentResult` |
| [Datasets](datasets.md) | Built-in datasets | `load_dataset`, `load_sp500_returns`, `generate_sv_data` |
| [CLI](cli.md) | Command-line interface | `pfbox filter`, `pfbox estimate`, `pfbox compare` |

---

## Conventions

### Parameters

- **`model`** — Always a `ParticleFilterModel` subclass instance.
- **`config`** — A `PFConfig` dataclass controlling filter behavior (number of particles, resampling scheme, etc.).
- **`endog`** — Observed data as a NumPy array of shape `(T,)` or `(T, k_obs)`.
- **`rng`** — A `numpy.random.Generator` instance for reproducibility.

### Return Types

- Filters return `ParticleFilterResults` with filtered means, covariances, log-likelihood, and ESS history.
- Smoothers return `ParticleSmootherResults` with smoothed means, covariances, and optional trajectories.
- PMCMC methods return `PMCMCResults` with posterior chains and diagnostics.

### Type Annotations

All public functions and classes use Python type annotations. Common types:

```python
import numpy as np
from numpy.typing import NDArray

# Particle arrays
particles: NDArray[np.float64]    # shape (N, k)
log_weights: NDArray[np.float64]  # shape (N,)
observations: NDArray[np.float64] # shape (T,) or (T, k_obs)
```

### Numerical Stability

The library uses **log-weight arithmetic** throughout. Weights are stored and manipulated in log-space to avoid underflow. Normalized weights are computed on-demand via `log_sum_exp`.

### Error Handling

- `ValueError` — Invalid parameter values (e.g., negative particle count).
- `TypeError` — Wrong parameter types.
- `RuntimeError` — Filter divergence or numerical issues.
- `NotImplementedError` — Unimplemented abstract methods.

---

## Quick Start

```python
import numpy as np
import particlefilterbox as pfb

# 1. Define a model
model = pfb.models.StochasticVolatility(variant='basic')

# 2. Configure the filter
config = pfb.PFConfig(n_particles=5000, resampling='systematic')

# 3. Run the filter
pf = pfb.BootstrapPF(model, config)
results = pf.filter(observations)

# 4. Smooth estimates
smoother = pfb.FFBSm()
smoothed = smoother.smooth(results, model)

# 5. Inspect results
print(results.summary())
print(f"Log-likelihood: {results.log_likelihood:.2f}")
```

---

## See Also

- [Getting Started](../getting-started/index.md) — Installation and quickstart guide
- [User Guide](../user-guide/index.md) — In-depth usage documentation
- [Tutorials](../tutorials/index.md) — Step-by-step worked examples
- [Theory](../theory/index.md) — Mathematical foundations
