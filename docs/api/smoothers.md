---
title: "Smoothers API"
description: "API reference for particlefilterbox.smoothers — FFBSm, FFBSi, TwoFilterSmoother, FixedLagSmoother, and ParticleSmootherResults"
---

# Smoothers API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.smoothers import FFBSm, FFBSi, TwoFilterSmoother, FixedLagSmoother`
    **Source**: `particlefilterbox/smoothers/`

## Overview

Particle smoothers compute the smoothing distribution $p(x_t \mid y_{1:T})$ which conditions on **all** observations, including future ones. This provides more accurate state estimates than filtering alone.

The smoothing distribution differs from the filtering distribution:

$$
p(x_t \mid y_{1:T}) \neq p(x_t \mid y_{1:t}), \qquad t < T
$$

All smoothers require filter results with stored particle and weight histories (`store_particles=True`, `store_weights=True` in `PFConfig`).

| Smoother | Method | Complexity | Output |
|----------|--------|-----------|--------|
| `FFBSm` | Forward Filtering Backward Smoothing | $O(T \cdot N^2)$ | Smoothed moments |
| `FFBSi` | Forward Filtering Backward Simulation | $O(T \cdot N \cdot M)$ | $M$ trajectories |
| `TwoFilterSmoother` | Forward + Backward filters | $O(T \cdot N^2)$ | Smoothed moments |
| `FixedLagSmoother` | Online smoothing with fixed lag | $O(N \cdot L)$ per step | Lagged states |

---

## FFBSm

Forward Filtering Backward Smoothing computes smoothed **marginal** distributions by reweighting filter particles using backward transition kernels (Doucet, Godsill & Andrieu, 2000).

At each time step $t$, the smoothed weights are:

$$
W_{t|T}^{(i)} \propto \sum_{j=1}^{N} W_{t+1|T}^{(j)} \frac{W_{t|t}^{(i)} \, f(x_{t+1}^{(j)} \mid x_t^{(i)})}{\sum_{l=1}^{N} W_{t|t}^{(l)} \, f(x_{t+1}^{(j)} \mid x_t^{(l)})}
$$

### Constructor

```python
FFBSm()
```

No constructor parameters. The smoother is configured at smooth-time.

### Methods

##### `smooth()`

Run backward smoothing on filter results.

```python
def smooth(
    self,
    filter_results: ParticleFilterResults,
    model: ParticleFilterModel,
) -> ParticleSmootherResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_results` | `ParticleFilterResults` | *required* | Output from a particle filter with stored particles/weights |
| `model` | `ParticleFilterModel` | *required* | State-space model (needed for transition density) |

**Returns**: `ParticleSmootherResults` — Smoothed means, covariances, and weights.

**Raises**:

- `ValueError` if `filter_results` does not contain particle/weight history.
- `RuntimeError` if transition density evaluation fails.

!!! warning
    FFBSm has $O(T \cdot N^2)$ complexity. For long time series with many particles, consider FFBSi or FixedLagSmoother instead.

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='basic')
config = pfb.PFConfig(
    n_particles=2000,
    store_particles=True,
    store_weights=True,
    seed=42,
)

# Run filter
pf = pfb.BootstrapPF(model, config)
filter_results = pf.filter(observations)

# Run smoother
smoother = pfb.FFBSm()
smoothed = smoother.smooth(filter_results, model)

print(smoothed.summary())
print(f"Smoothed mean at t=0: {smoothed.smoothed_mean[0]}")
```

---

## FFBSi

Forward Filtering Backward Simulation generates $M$ complete state trajectories $x_{0:T}$ from the joint smoothing distribution (Godsill, Doucet & West, 2004).

At each backward step, sample ancestor indices:

$$
a_t^{(m)} \sim \text{Categorical}\!\left(\frac{W_{t|t}^{(i)} \, f(x_{t+1}^{(m)} \mid x_t^{(i)})}{\sum_{l} W_{t|t}^{(l)} \, f(x_{t+1}^{(m)} \mid x_t^{(l)})}\right)
$$

then set $x_t^{(m)} = x_t^{(a_t^{(m)})}$.

### Constructor

```python
FFBSi(
    n_trajectories: int = 100,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_trajectories` | `int` | `100` | Number of backward trajectories $M$ to sample |

### Methods

##### `smooth()`

Run backward simulation on filter results.

```python
def smooth(
    self,
    filter_results: ParticleFilterResults,
    model: ParticleFilterModel,
) -> ParticleSmootherResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_results` | `ParticleFilterResults` | *required* | Output from a particle filter with stored particles/weights |
| `model` | `ParticleFilterModel` | *required* | State-space model (needed for transition density) |

**Returns**: `ParticleSmootherResults` — Smoothed estimates with `trajectories` attribute of shape `(M, T, k)`.

**Raises**: `ValueError` if `filter_results` does not contain particle/weight history.

!!! tip
    FFBSi produces full trajectories, which are essential for:

    - Path-dependent functionals $\mathbb{E}[\phi(x_{0:T}) \mid y_{1:T}]$
    - Particle Gibbs samplers (PGAS)
    - Visualization of state trajectories

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='basic')
config = pfb.PFConfig(
    n_particles=2000,
    store_particles=True,
    store_weights=True,
    seed=42,
)

# Run filter
pf = pfb.BootstrapPF(model, config)
filter_results = pf.filter(observations)

# Run backward simulation with 200 trajectories
smoother = pfb.FFBSi(n_trajectories=200)
smoothed = smoother.smooth(filter_results, model)

print(f"Trajectories shape: {smoothed.trajectories.shape}")  # (200, T, k)

# Compute functional estimate
import numpy as np
mean_trajectory = np.mean(smoothed.trajectories, axis=0)  # (T, k)
```

---

## TwoFilterSmoother

The Two-Filter Smoother (Briers, Doucet & Maskell, 2010) runs a forward filter and a backward information filter, then combines them to produce the smoothing distribution.

$$
p(x_t \mid y_{1:T}) \propto p(x_t \mid y_{1:t}) \cdot \tilde{p}(y_{t+1:T} \mid x_t)
$$

The backward pass runs a separate particle filter on the reversed observation sequence using an artificial backward kernel.

### Constructor

```python
TwoFilterSmoother(
    model: ParticleFilterModel,
    config: PFConfig,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `config` | `PFConfig` | *required* | Configuration (applies to both forward and backward filters) |

### Methods

##### `smooth()`

Run two-filter smoothing over the full observation sequence.

```python
def smooth(
    self,
    endog: NDArray[np.float64],
) -> ParticleSmootherResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `NDArray[np.float64]` | *required* | Observations, shape `(T,)` or `(T, k_obs)` |

**Returns**: `ParticleSmootherResults` — Smoothed means, covariances, and combined weights.

!!! note
    Unlike FFBSm and FFBSi, the two-filter smoother takes raw observations rather than filter results. It runs the forward filter internally.

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='basic')
config = pfb.PFConfig(
    n_particles=2000,
    store_particles=True,
    store_weights=True,
    seed=42,
)

smoother = pfb.TwoFilterSmoother(model, config)
smoothed = smoother.smooth(observations)

print(smoothed.summary())
```

---

## FixedLagSmoother

The Fixed-Lag Smoother provides online smoothed estimates with a fixed delay of $L$ time steps. At time $t$, it outputs the smoothed estimate of $x_{t-L}$ using ancestor tracing through the particle genealogy.

$$
\hat{p}(x_{t-L} \mid y_{1:t}) = \sum_{i=1}^{N} W_t^{(i)} \, \delta_{x_{t-L}^{(b_t^{(i)})}}(x_{t-L})
$$

where $b_t^{(i)}$ is the ancestor of particle $i$ at lag $L$.

### Constructor

```python
FixedLagSmoother(
    model: ParticleFilterModel,
    config: PFConfig,
    lag: int = 10,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `config` | `PFConfig` | *required* | Filter configuration |
| `lag` | `int` | `10` | Smoothing lag $L$ |

!!! warning
    The config must have `store_ancestors=True` for ancestor tracing. The lag should be large enough to capture the mixing time of the state process but small enough to avoid path degeneracy.

### Methods

##### `smooth()`

Run fixed-lag smoothing over the full observation sequence.

```python
def smooth(
    self,
    endog: NDArray[np.float64],
) -> ParticleSmootherResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `NDArray[np.float64]` | *required* | Observations, shape `(T,)` or `(T, k_obs)` |

**Returns**: `ParticleSmootherResults` — Smoothed means and covariances (available from $t = L$ onward).

---

##### `step()`

Process a single observation and return the smoothed state at lag $L$.

```python
def step(
    self,
    observation: NDArray[np.float64],
    t: int,
) -> NDArray[np.float64] | None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observation` | `NDArray[np.float64]` | *required* | Current observation $y_t$ |
| `t` | `int` | *required* | Time index |

**Returns**: `NDArray[np.float64] | None` — Smoothed state mean at $t - L$, or `None` if $t < L$.

!!! tip
    The `step()` method enables **online** smoothing in streaming applications. Each call processes one observation and returns the lagged smoothed estimate.

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='basic')
config = pfb.PFConfig(
    n_particles=2000,
    store_ancestors=True,
    seed=42,
)

# Batch mode
smoother = pfb.FixedLagSmoother(model, config, lag=15)
smoothed = smoother.smooth(observations)
print(smoothed.summary())

# Online mode
smoother = pfb.FixedLagSmoother(model, config, lag=15)
for t, y_t in enumerate(observations):
    state = smoother.step(y_t, t)
    if state is not None:
        print(f"Smoothed x[{t-15}] = {state}")
```

---

## ParticleSmootherResults

Dataclass containing the output of a particle smoother. See [Core API: ParticleSmootherResults](core.md#particlesmootherresults) for the full attribute and method reference.

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `smoothed_mean` | `NDArray[np.float64]` | Smoothed state means, shape `(T, k)` |
| `smoothed_cov` | `NDArray[np.float64]` | Smoothed covariances, shape `(T, k, k)` |
| `smoothed_quantiles` | `dict[float, NDArray]` | Smoothed quantile bands |
| `trajectories` | `NDArray \| None` | Sampled trajectories, shape `(M, T, k)` (FFBSi only) |
| `method` | `str` | Smoother method name |
| `filter_results` | `ParticleFilterResults` | Associated filter output |
| `computation_time_seconds` | `float` | Wall-clock time |
| `n_particles` | `int` | Number of particles |
| `n_timesteps` | `int` | Number of time steps $T$ |
| `state_dim` | `int` | State dimension $k$ |

### Methods

| Method | Description |
|--------|-------------|
| `summary()` | Formatted summary string |
| `functional_estimate(func)` | Compute $\mathbb{E}[\phi(x_{0:T}) \mid y_{1:T}]$ |
| `to_dataframe(state_names)` | Convert to pandas DataFrame |

---

## Smoother Comparison

| Smoother | Type | Complexity | Produces Trajectories | Online |
|----------|------|-----------|:---------------------:|:------:|
| `FFBSm` | Marginal | $O(T \cdot N^2)$ | No | No |
| `FFBSi` | Joint | $O(T \cdot N \cdot M)$ | Yes | No |
| `TwoFilterSmoother` | Marginal | $O(T \cdot N^2)$ | No | No |
| `FixedLagSmoother` | Marginal (lagged) | $O(N \cdot L)$ per step | No | Yes |

!!! tip "Choosing a Smoother"
    - **FFBSm**: Best for smoothed moments when $N$ is moderate. Exact marginal smoothing.
    - **FFBSi**: Required when you need full trajectories (e.g., for Particle Gibbs). Lower per-trajectory cost than FFBSm.
    - **TwoFilterSmoother**: Alternative to FFBSm that avoids storing the full forward particle history.
    - **FixedLagSmoother**: Only option for online/streaming applications. Approximation quality depends on lag.

---

## See Also

- [User Guide: Smoothers](../user-guide/smoothers/index.md) — In-depth smoother usage guide
- [User Guide: FFBSm](../user-guide/smoothers/ffbsm.md) — Forward Filtering Backward Smoothing guide
- [User Guide: FFBSi](../user-guide/smoothers/ffbsi.md) — Forward Filtering Backward Simulation guide
- [Core API](core.md) — `ParticleSmootherResults` full reference
- [Filters API](filters.md) — Particle filter reference (run before smoothing)
- [Theory: Smoothing](../theory/smoothing-theory.md) — Mathematical foundations
- [Tutorials: Smoothing](../tutorials/smoothing.md) — Step-by-step smoothing tutorial
- [PMCMC API](pmcmc.md) — Smoothers used within PMCMC
