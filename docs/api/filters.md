---
title: "Filters API"
description: "API reference for particlefilterbox.filters — Bootstrap, SIR, Auxiliary, Rao-Blackwellized, Unscented, Regularized, Ensemble, Guided, and Locally Optimal particle filters"
---

# Filters API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.filters import BootstrapPF, SIR, AuxiliaryPF, RaoBlackwellizedPF, UnscentedPF, RegularizedPF, EnsemblePF, GuidedPF, LocallyOptimalPF`
    **Source**: `particlefilterbox/filters/`

## Overview

The filters module provides nine particle filter implementations, all sharing a common interface. Each filter implements a different proposal or weighting strategy for the Sequential Monte Carlo approximation:

$$
\hat{p}(x_t \mid y_{1:t}) \approx \sum_{i=1}^{N} W_t^{(i)} \, \delta_{x_t^{(i)}}(x_t)
$$

| Filter | Proposal | Best For |
|--------|----------|----------|
| `BootstrapPF` | Transition prior $f(x_t \mid x_{t-1})$ | Simple models, baseline |
| `SIR` | User-defined $q(x_t \mid x_{t-1}, y_t)$ | Custom proposals |
| `AuxiliaryPF` | Pre-selected via first-stage weights | Informative observations |
| `RaoBlackwellizedPF` | Marginalizes linear sub-state | Mixed linear/nonlinear |
| `UnscentedPF` | UKF-based proposal | Nonlinear with Gaussian noise |
| `RegularizedPF` | Kernel-jittered post-resampling | Continuous distributions |
| `EnsemblePF` | Ensemble Kalman-like updates | High-dimensional states |
| `GuidedPF` | Observation-guided proposal | Precise observations |
| `LocallyOptimalPF` | Analytic optimal proposal | Gaussian transition + observation |

---

## Common Interface

All filters inherit from `BaseParticleFilter` and share these methods:

### Constructor

```python
BaseParticleFilter(
    model: ParticleFilterModel,
    config: PFConfig,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model definition |
| `config` | `PFConfig` | *required* | Filter configuration (particles, resampling, etc.) |

### Common Methods

##### `filter()`

Run the particle filter over the full observation sequence.

```python
def filter(
    self,
    endog: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
) -> ParticleFilterResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `NDArray[np.float64]` | *required* | Observations, shape `(T,)` or `(T, k_obs)` |
| `mask` | `NDArray[np.bool_] \| None` | `None` | Missing data mask, shape `(T,)`. `True` = observed |

**Returns**: `ParticleFilterResults` — Filtered estimates and diagnostics.

**Raises**:

- `ValueError` if `endog` shape is incompatible with `model.k_obs`.
- `RuntimeError` if all particles have zero weight (filter divergence).

---

##### `initialize()`

Initialize the particle cloud from the model's initial distribution.

```python
def initialize(self) -> ParticleCloud
```

**Returns**: `ParticleCloud` — Initialized particle cloud with uniform weights.

---

##### `filter_step()`

Perform a single filter step (propagate, weight, optionally resample).

```python
def filter_step(
    self,
    cloud: ParticleCloud,
    observation: NDArray[np.float64],
    t: int,
) -> ParticleCloud
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cloud` | `ParticleCloud` | *required* | Current particle cloud |
| `observation` | `NDArray[np.float64]` | *required* | Current observation $y_t$ |
| `t` | `int` | *required* | Time index |

**Returns**: `ParticleCloud` — Updated particle cloud.

---

## BootstrapPF

The Bootstrap Particle Filter (Gordon, Salmond & Smith, 1993) uses the transition prior as the proposal distribution. The simplest particle filter and a natural baseline.

**Proposal**: $q(x_t \mid x_{t-1}, y_t) = f(x_t \mid x_{t-1})$

**Weights**: $w_t^{(i)} \propto g(y_t \mid x_t^{(i)})$

### Constructor

```python
BootstrapPF(
    model: ParticleFilterModel,
    config: PFConfig,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `config` | `PFConfig` | *required* | Filter configuration |

### Example

```python
import numpy as np
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='basic')
config = pfb.PFConfig(n_particles=5000, resampling='systematic', seed=42)

pf = pfb.BootstrapPF(model, config)
results = pf.filter(observations)

print(results.summary())
print(f"Log-likelihood: {results.log_likelihood:.2f}")
```

!!! tip
    The bootstrap filter is a good starting point for any new model. If ESS drops frequently below the threshold, consider switching to a filter with a better proposal (SIR, Guided, or Locally Optimal).

---

## SIR

Sequential Importance Resampling with a user-defined proposal distribution $q(x_t \mid x_{t-1}, y_t)$. Generalizes the bootstrap filter — when `model.proposal()` is not implemented, falls back to bootstrap behavior.

**Proposal**: $q(x_t \mid x_{t-1}, y_t)$ (user-defined via `model.proposal()`)

**Weights**: $w_t^{(i)} \propto \dfrac{g(y_t \mid x_t^{(i)}) \, f(x_t^{(i)} \mid x_{t-1}^{(i)})}{q(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t)}$

### Constructor

```python
SIR(
    model: ParticleFilterModel,
    config: PFConfig,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | Model with `proposal()` method implemented |
| `config` | `PFConfig` | *required* | Filter configuration |

!!! note
    The model must implement `proposal()` and the corresponding `log_proposal_density()` for correct importance weight computation.

### Example

```python
import particlefilterbox as pfb

# Model with custom proposal
model = MyModelWithProposal()
config = pfb.PFConfig(n_particles=3000)

pf = pfb.SIR(model, config)
results = pf.filter(observations)
```

---

## AuxiliaryPF

The Auxiliary Particle Filter (Pitt & Shephard, 1999) introduces a first-stage pre-selection step that favors particles likely to match the next observation before propagation.

**First stage**: Pre-weight particles using a point estimate of $g(y_t \mid \mu_t^{(i)})$ where $\mu_t^{(i)} = \mathbb{E}[x_t \mid x_{t-1}^{(i)}]$.

**Second stage**: Propagate pre-selected particles and compute adjustment weights.

### Constructor

```python
AuxiliaryPF(
    model: ParticleFilterModel,
    config: PFConfig,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `config` | `PFConfig` | *required* | Filter configuration |

!!! note
    The auxiliary PF works best when the model provides a good predictive mean for the first-stage weights. If the transition is highly nonlinear, the first-stage approximation may be poor.

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='basic')
config = pfb.PFConfig(n_particles=3000, seed=42)

pf = pfb.AuxiliaryPF(model, config)
results = pf.filter(observations)
print(f"Log-likelihood: {results.log_likelihood:.2f}")
```

---

## RaoBlackwellizedPF

The Rao-Blackwellized Particle Filter (Doucet, de Freitas & Gordon, 2000; Schon, Gustafsson & Nordlund, 2005) analytically marginalizes a conditionally linear sub-state using a Kalman filter, reducing variance.

For a model with state $x_t = (x_t^{(n)}, x_t^{(l)})$ where the linear component $x_t^{(l)}$ is conditionally Gaussian given $x_t^{(n)}$:

$$
x_t^{(l)} \mid x_t^{(n)}, y_{1:t} \sim \mathcal{N}(\hat{x}_t^{(l)}, P_t^{(l)})
$$

The nonlinear component $x_t^{(n)}$ is handled by particles, while $x_t^{(l)}$ is tracked analytically.

### Constructor

```python
RaoBlackwellizedPF(
    model: ParticleFilterModel,
    config: PFConfig,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | Model with `has_linear_substate()` returning `True` |
| `config` | `PFConfig` | *required* | Filter configuration |

!!! warning
    The model must implement `has_linear_substate()` returning `True` and provide the linear sub-system matrices. Uses [kalmanbox](https://github.com/nodesecon/kalmanbox) internally.

### Example

```python
import particlefilterbox as pfb

# Model with conditionally linear sub-state
model = MyRBModel()  # has_linear_substate() -> True
config = pfb.PFConfig(n_particles=1000)

pf = pfb.RaoBlackwellizedPF(model, config)
results = pf.filter(observations)
```

---

## UnscentedPF

The Unscented Particle Filter (van der Merwe et al., 2001) uses the Unscented Kalman Filter (UKF) to construct a Gaussian proposal that incorporates the current observation.

**Proposal**: UKF posterior approximation $q(x_t \mid x_{t-1}^{(i)}, y_t) = \mathcal{N}(\hat{x}_t^{(i)}, P_t^{(i)})$

This proposal is closer to the optimal proposal than the prior, especially when observations are informative.

### Constructor

```python
UnscentedPF(
    model: ParticleFilterModel,
    config: PFConfig,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `config` | `PFConfig` | *required* | Filter configuration |

!!! note
    Requires [kalmanbox](https://github.com/nodesecon/kalmanbox) for the internal UKF. Installed automatically as a dependency.

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='basic')
config = pfb.PFConfig(n_particles=2000, seed=42)

pf = pfb.UnscentedPF(model, config)
results = pf.filter(observations)
```

---

## RegularizedPF

The Regularized Particle Filter (Musso, Oudjane & Le Gland, 2001) applies kernel jittering after resampling to avoid sample impoverishment. Adds a small amount of noise to resampled particles using a kernel density estimate.

$$
x_t^{(i)} \leftarrow x_t^{(a_i)} + h \cdot K(\epsilon), \qquad \epsilon \sim K
$$

where $h$ is the bandwidth and $K$ is the regularization kernel (typically Gaussian or Epanechnikov).

### Constructor

```python
RegularizedPF(
    model: ParticleFilterModel,
    config: PFConfig,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `config` | `PFConfig` | *required* | Filter configuration |

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='basic')
config = pfb.PFConfig(n_particles=3000)

pf = pfb.RegularizedPF(model, config)
results = pf.filter(observations)
```

---

## EnsemblePF

The Ensemble Particle Filter combines ideas from Ensemble Kalman Filters (EnKF) with particle filtering. Uses ensemble-based covariance estimates for proposal construction and supports localization and inflation for high-dimensional states.

### Constructor

```python
EnsemblePF(
    model: ParticleFilterModel,
    config: PFConfig,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `config` | `PFConfig` | *required* | Filter configuration |

!!! tip
    The ensemble filter scales better to high-dimensional state spaces than standard particle filters due to its use of ensemble covariance rather than full particle weights.

### Example

```python
import particlefilterbox as pfb

model = pfb.models.DSGE(variant='small_nk')
config = pfb.PFConfig(n_particles=500)

pf = pfb.EnsemblePF(model, config)
results = pf.filter(observations)
```

---

## GuidedPF

The Guided Particle Filter uses an observation-guided proposal that shifts particles toward regions of high observation likelihood. The guide function steers the proposal using gradient or moment information from the observation density.

**Proposal**: $q(x_t \mid x_{t-1}^{(i)}, y_t) \propto f(x_t \mid x_{t-1}^{(i)}) \cdot \tilde{g}(y_t \mid x_t)$

where $\tilde{g}$ is an approximation to the observation density used for guidance.

### Constructor

```python
GuidedPF(
    model: ParticleFilterModel,
    config: PFConfig,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `config` | `PFConfig` | *required* | Filter configuration |

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='leverage')
config = pfb.PFConfig(n_particles=3000, seed=42)

pf = pfb.GuidedPF(model, config)
results = pf.filter(observations)
```

---

## LocallyOptimalPF

The Locally Optimal Particle Filter uses the analytic optimal proposal distribution, available when both the transition and observation densities are Gaussian (or can be approximated as such).

The optimal proposal minimizes the variance of importance weights:

$$
q^*(x_t \mid x_{t-1}^{(i)}, y_t) = p(x_t \mid x_{t-1}^{(i)}, y_t) \propto g(y_t \mid x_t) \, f(x_t \mid x_{t-1}^{(i)})
$$

For linear-Gaussian sub-problems, this yields a Gaussian proposal with analytically computed mean and covariance.

### Constructor

```python
LocallyOptimalPF(
    model: ParticleFilterModel,
    config: PFConfig,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `config` | `PFConfig` | *required* | Filter configuration |

!!! tip
    The locally optimal filter provides the best possible importance weights among all proposal distributions. However, it requires that the optimal proposal is available in closed form (typically Gaussian transitions and observations).

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='basic')
config = pfb.PFConfig(n_particles=1000, seed=42)

pf = pfb.LocallyOptimalPF(model, config)
results = pf.filter(observations)
print(f"Mean ESS: {results.ess_history.mean():.0f}")
```

---

## Filter Comparison

A quick reference for selecting the right filter:

| Filter | Proposal Quality | Complexity per Step | Requirements |
|--------|-----------------|--------------------:|--------------|
| `BootstrapPF` | Prior | $O(N)$ | None |
| `SIR` | Custom | $O(N)$ | `model.proposal()` |
| `AuxiliaryPF` | Adapted prior | $O(N)$ | Predictive mean |
| `RaoBlackwellizedPF` | Marginalized | $O(N \cdot k_l^3)$ | Linear sub-state |
| `UnscentedPF` | UKF-based | $O(N \cdot k^3)$ | kalmanbox |
| `RegularizedPF` | Prior + kernel | $O(N)$ | None |
| `EnsemblePF` | Ensemble-based | $O(N \cdot k^2)$ | None |
| `GuidedPF` | Guided prior | $O(N)$ | None |
| `LocallyOptimalPF` | Optimal | $O(N \cdot k^3)$ | Gaussian noise |

---

## See Also

- [User Guide: Filters](../user-guide/filters/index.md) — In-depth filter usage guide
- [Getting Started: Choosing a Filter](../getting-started/choosing-filter.md) — Decision guide
- [Core API](core.md) — `ParticleFilterModel` and `ParticleFilterResults`
- [Resampling API](resampling.md) — Resampling algorithms
- [Theory: Particle Filters](../theory/particle-filter-theory.md) — Mathematical foundations
- [Tutorials: Fundamentals](../tutorials/fundamentals.md) — Step-by-step introduction
- [Tutorials: Auxiliary PF](../tutorials/auxiliary-pf.md) — Auxiliary filter tutorial
- [Tutorials: RBPF](../tutorials/rbpf.md) — Rao-Blackwellized filter tutorial
- [Benchmarks: Filters](../benchmarks/filters.md) — Performance comparison
