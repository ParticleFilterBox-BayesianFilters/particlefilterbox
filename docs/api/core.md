---
title: "Core API"
description: "API reference for particlefilterbox.core — ParticleCloud, ParticleFilterModel, ParticleFilterResults, and PFConfig"
---

# Core API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.core import ParticleCloud, ParticleFilterModel, ParticleFilterResults, PFConfig`
    **Source**: `particlefilterbox/core/`

## Overview

The core module provides the fundamental data structures and abstract base classes for the entire library. Every filter, smoother, and SMC method builds on these components.

| Class | Role | Description |
|-------|------|-------------|
| `ParticleCloud` | Data container | Weighted particle ensemble in state space |
| `ParticleFilterModel` | Abstract base | Interface for state-space model definitions |
| `ParticleFilterResults` | Output | Filter output with means, covariances, diagnostics |
| `ParticleSmootherResults` | Output | Smoother output with smoothed estimates |
| `PFConfig` | Configuration | Filter hyperparameters and runtime options |

---

## ParticleCloud

The central data structure representing a weighted set of $N$ particles in a $k$-dimensional state space. Each particle $x^{(i)}$ carries an unnormalized log-weight $\log w^{(i)}$.

$$
\hat{p}(x_t \mid y_{1:t}) = \sum_{i=1}^{N} W_t^{(i)} \, \delta_{x_t^{(i)}}(x_t), \qquad W_t^{(i)} = \frac{w_t^{(i)}}{\sum_{j=1}^{N} w_t^{(j)}}
$$

### Constructor

```python
ParticleCloud(
    n_particles: int,
    k_states: int,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | *required* | Number of particles $N$ |
| `k_states` | `int` | *required* | Dimension of the state space $k$ |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator for reproducibility |

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `particles` | `NDArray[np.float64]` | Particle positions, shape `(N, k)` |
| `log_weights` | `NDArray[np.float64]` | Unnormalized log-weights, shape `(N,)` |
| `normalized_weights` | `NDArray[np.float64]` | Normalized weights $W^{(i)}$, shape `(N,)` |
| `ess` | `float` | Effective sample size $\text{ESS} = 1 / \sum_i (W^{(i)})^2$ |
| `log_likelihood_increment` | `float` | Log-likelihood contribution $\log \hat{p}(y_t \mid y_{1:t-1})$ |

### Methods

##### `set_uniform_weights()`

Reset all log-weights to uniform ($\log w^{(i)} = 0$ for all $i$).

```python
def set_uniform_weights(self) -> None
```

---

##### `set_log_weights()`

Set log-weights from an external array.

```python
def set_log_weights(
    self,
    log_weights: NDArray[np.float64],
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_weights` | `NDArray[np.float64]` | *required* | New log-weights, shape `(N,)` |

**Raises**: `ValueError` if shape does not match `(N,)`.

---

##### `add_log_weights()`

Increment log-weights (multiply weights in log-space).

```python
def add_log_weights(
    self,
    log_increments: NDArray[np.float64],
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_increments` | `NDArray[np.float64]` | *required* | Log-weight increments, shape `(N,)` |

---

##### `resample()`

Resample particles according to current weights using the specified scheme.

```python
def resample(
    self,
    method: str = "systematic",
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"systematic"` | Resampling algorithm (`"systematic"`, `"multinomial"`, `"stratified"`, `"residual"`) |
| `rng` | `np.random.Generator \| None` | `None` | Random generator (uses internal if `None`) |

**Returns**: `NDArray[np.int64]` — Ancestor indices, shape `(N,)`.

---

##### `weighted_mean()`

Compute the weighted mean of the particle cloud.

```python
def weighted_mean(self) -> NDArray[np.float64]
```

**Returns**: `NDArray[np.float64]` — Weighted mean $\hat{\mu} = \sum_i W^{(i)} x^{(i)}$, shape `(k,)`.

---

##### `weighted_cov()`

Compute the weighted covariance matrix.

```python
def weighted_cov(self) -> NDArray[np.float64]
```

**Returns**: `NDArray[np.float64]` — Weighted covariance $\hat{\Sigma}$, shape `(k, k)`.

---

##### `weighted_quantile()`

Compute weighted quantiles of the particle distribution.

```python
def weighted_quantile(
    self,
    quantiles: list[float] | NDArray[np.float64],
) -> NDArray[np.float64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantiles` | `list[float] \| NDArray` | *required* | Quantile levels, e.g. `[0.025, 0.5, 0.975]` |

**Returns**: `NDArray[np.float64]` — Quantile values, shape `(len(quantiles), k)`.

---

##### `clone()`

Create an independent deep copy of the particle cloud.

```python
def clone(self) -> ParticleCloud
```

**Returns**: `ParticleCloud` — A new `ParticleCloud` with copied particles and weights.

---

### Magic Methods

| Method | Description |
|--------|-------------|
| `__len__()` | Returns $N$ (number of particles) |
| `__getitem__(idx)` | Index into particles array |
| `__repr__()` | String representation with $N$, $k$, and ESS |

### Example

```python
import numpy as np
from particlefilterbox.core import ParticleCloud

# Create a cloud of 1000 particles in 2D
cloud = ParticleCloud(n_particles=1000, k_states=2)
cloud.particles[:] = np.random.randn(1000, 2)
cloud.set_uniform_weights()

print(f"ESS: {cloud.ess:.1f}")           # ESS: 1000.0
print(f"Mean: {cloud.weighted_mean()}")   # Mean: ~[0, 0]

# Add log-likelihood weights
log_liks = -0.5 * np.sum(cloud.particles**2, axis=1)
cloud.add_log_weights(log_liks)
print(f"ESS after weighting: {cloud.ess:.1f}")

# Resample
ancestors = cloud.resample(method="systematic")
print(f"ESS after resampling: {cloud.ess:.1f}")  # ESS: 1000.0
```

---

## ParticleFilterModel

Abstract base class defining the state-space model interface. All models used with particlefilterbox filters must inherit from this class and implement the three core methods.

The generic state-space model is:

$$
\begin{aligned}
x_0 &\sim \mu(\cdot) \\
x_t &\sim f(x_t \mid x_{t-1}, \theta) \\
y_t &\sim g(y_t \mid x_t, \theta)
\end{aligned}
$$

### Constructor

```python
ParticleFilterModel(
    k_states: int,
    k_obs: int,
    param_names: list[str] | None = None,
    params: dict[str, float] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_states` | `int` | *required* | Dimension of the state vector $x_t$ |
| `k_obs` | `int` | *required* | Dimension of the observation vector $y_t$ |
| `param_names` | `list[str] \| None` | `None` | Names of model parameters $\theta$ |
| `params` | `dict[str, float] \| None` | `None` | Parameter values |

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `k_states` | `int` | State dimension |
| `k_obs` | `int` | Observation dimension |
| `param_names` | `list[str]` | Parameter names |
| `params` | `dict[str, float]` | Current parameter values |

### Abstract Methods

##### `initial_distribution()`

Sample initial particles $x_0^{(i)} \sim \mu(\cdot)$.

```python
@abstractmethod
def initial_distribution(
    self,
    n_particles: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | *required* | Number of particles to sample |
| `rng` | `np.random.Generator` | *required* | Random generator |

**Returns**: `NDArray[np.float64]` — Initial particles, shape `(N, k_states)`.

---

##### `transition()`

Sample from the transition density $x_t^{(i)} \sim f(\cdot \mid x_{t-1}^{(i)}, \theta)$.

```python
@abstractmethod
def transition(
    self,
    particles: NDArray[np.float64],
    t: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `particles` | `NDArray[np.float64]` | *required* | Current particles, shape `(N, k_states)` |
| `t` | `int` | *required* | Time index |
| `rng` | `np.random.Generator` | *required* | Random generator |

**Returns**: `NDArray[np.float64]` — Propagated particles, shape `(N, k_states)`.

---

##### `log_observation_likelihood()`

Compute log-likelihood $\log g(y_t \mid x_t^{(i)}, \theta)$ for each particle.

```python
@abstractmethod
def log_observation_likelihood(
    self,
    particles: NDArray[np.float64],
    observation: NDArray[np.float64],
    t: int,
) -> NDArray[np.float64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `particles` | `NDArray[np.float64]` | *required* | Current particles, shape `(N, k_states)` |
| `observation` | `NDArray[np.float64]` | *required* | Observation $y_t$, shape `(k_obs,)` |
| `t` | `int` | *required* | Time index |

**Returns**: `NDArray[np.float64]` — Log-likelihoods, shape `(N,)`.

### Optional Methods

##### `proposal()`

Custom proposal distribution $q(x_t \mid x_{t-1}, y_t)$. If not implemented, the filter uses the transition as the proposal (bootstrap filter).

```python
def proposal(
    self,
    particles: NDArray[np.float64],
    observation: NDArray[np.float64],
    t: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]
```

**Returns**: `NDArray[np.float64]` — Proposed particles, shape `(N, k_states)`.

---

##### `log_transition_density()`

Evaluate the transition density $\log f(x_t \mid x_{t-1}, \theta)$. Required for smoothers and some advanced filters.

```python
def log_transition_density(
    self,
    particles_curr: NDArray[np.float64],
    particles_prev: NDArray[np.float64],
    t: int,
) -> NDArray[np.float64]
```

**Returns**: `NDArray[np.float64]` — Log-densities, shape `(N,)` or `(N, N)`.

---

##### `has_linear_substate()`

Indicate whether the model has a conditionally linear sub-state (for Rao-Blackwellization).

```python
def has_linear_substate(self) -> bool
```

**Returns**: `bool` — `True` if the model supports Rao-Blackwellization.

### Example

```python
import numpy as np
from particlefilterbox.core import ParticleFilterModel

class LocalLevelModel(ParticleFilterModel):
    """Local level (random walk + noise) model."""

    def __init__(self, sigma_state: float = 1.0, sigma_obs: float = 1.0):
        super().__init__(
            k_states=1, k_obs=1,
            param_names=['sigma_state', 'sigma_obs'],
            params={'sigma_state': sigma_state, 'sigma_obs': sigma_obs},
        )

    def initial_distribution(self, n_particles, rng):
        return rng.normal(0, 10, size=(n_particles, 1))

    def transition(self, particles, t, rng):
        sigma = self.params['sigma_state']
        return particles + rng.normal(0, sigma, size=particles.shape)

    def log_observation_likelihood(self, particles, observation, t):
        sigma = self.params['sigma_obs']
        return -0.5 * ((observation - particles[:, 0]) / sigma) ** 2 \
               - np.log(sigma) - 0.5 * np.log(2 * np.pi)

# Use the model
model = LocalLevelModel(sigma_state=0.5, sigma_obs=1.0)
```

---

## ParticleFilterResults

Dataclass containing the output of a particle filter run. Stores filtered estimates, log-likelihood, ESS history, and optionally the full particle history.

### Constructor

```python
@dataclass
class ParticleFilterResults:
    filtered_mean: NDArray[np.float64]
    filtered_cov: NDArray[np.float64]
    filtered_quantiles: dict[float, NDArray[np.float64]]
    log_likelihood: float
    log_likelihood_increments: NDArray[np.float64]
    ess_history: NDArray[np.float64]
    resampled: NDArray[np.bool_]
    n_particles: int
    nobs: int
    computation_time: float
    particle_history: NDArray[np.float64] | None = None
    weight_history: NDArray[np.float64] | None = None
    ancestor_history: NDArray[np.int64] | None = None
```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `filtered_mean` | `NDArray[np.float64]` | Filtered state means, shape `(T, k)` |
| `filtered_cov` | `NDArray[np.float64]` | Filtered state covariances, shape `(T, k, k)` |
| `filtered_quantiles` | `dict[float, NDArray]` | Quantile bands, e.g. `{0.025: ..., 0.975: ...}` |
| `log_likelihood` | `float` | Total log-marginal likelihood $\log \hat{p}(y_{1:T})$ |
| `log_likelihood_increments` | `NDArray[np.float64]` | Per-step log-likelihoods, shape `(T,)` |
| `ess_history` | `NDArray[np.float64]` | ESS at each time step, shape `(T,)` |
| `resampled` | `NDArray[np.bool_]` | Whether resampling occurred at each step, shape `(T,)` |
| `n_particles` | `int` | Number of particles used |
| `nobs` | `int` | Number of observations $T$ |
| `computation_time` | `float` | Wall-clock time in seconds |
| `particle_history` | `NDArray \| None` | Full particle trajectories, shape `(T, N, k)` (if stored) |
| `weight_history` | `NDArray \| None` | Weight history, shape `(T, N)` (if stored) |
| `ancestor_history` | `NDArray \| None` | Ancestor indices, shape `(T, N)` (if stored) |

### Methods

##### `summary()`

Print a formatted summary of filter results.

```python
def summary(self) -> str
```

**Returns**: `str` — Multi-line summary with log-likelihood, mean ESS, resampling rate, and timing.

---

##### `to_dataframe()`

Convert filtered estimates to a pandas DataFrame.

```python
def to_dataframe(
    self,
    state_names: list[str] | None = None,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_names` | `list[str] \| None` | `None` | Column names for states (defaults to `state_0`, `state_1`, ...) |

**Returns**: `pd.DataFrame` — DataFrame with filtered means, standard deviations, and quantiles.

---

##### `save()`

Serialize results to disk.

```python
def save(self, path: str | Path) -> None
```

---

##### `load()`

Load results from disk.

```python
@classmethod
def load(cls, path: str | Path) -> ParticleFilterResults
```

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='basic')
config = pfb.PFConfig(n_particles=2000, store_particles=True)
pf = pfb.BootstrapPF(model, config)
results = pf.filter(observations)

# Inspect results
print(results.summary())
print(f"Log-likelihood: {results.log_likelihood:.2f}")
print(f"Mean ESS: {results.ess_history.mean():.0f}")

# Convert to DataFrame
df = results.to_dataframe(state_names=['log_volatility'])
print(df.head())
```

---

## ParticleSmootherResults

Dataclass containing the output of a particle smoother. Extends filter results with backward-smoothed estimates.

### Constructor

```python
@dataclass
class ParticleSmootherResults:
    smoothed_mean: NDArray[np.float64]
    smoothed_cov: NDArray[np.float64]
    smoothed_quantiles: dict[float, NDArray[np.float64]]
    smoothed_weights: NDArray[np.float64] | None
    trajectories: NDArray[np.float64] | None
    method: str
    filter_results: ParticleFilterResults
    computation_time_seconds: float
    n_particles: int
    n_timesteps: int
    state_dim: int
```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `smoothed_mean` | `NDArray[np.float64]` | Smoothed means, shape `(T, k)` |
| `smoothed_cov` | `NDArray[np.float64]` | Smoothed covariances, shape `(T, k, k)` |
| `smoothed_quantiles` | `dict[float, NDArray]` | Smoothed quantile bands |
| `trajectories` | `NDArray \| None` | Sampled trajectories, shape `(M, T, k)` (FFBSi only) |
| `method` | `str` | Smoother method name |
| `filter_results` | `ParticleFilterResults` | Associated filter output |

### Methods

##### `summary()`

Print a formatted summary of smoother results.

```python
def summary(self) -> str
```

---

##### `functional_estimate()`

Compute a functional of the smoothing distribution $\mathbb{E}[\phi(x_{0:T}) \mid y_{1:T}]$.

```python
def functional_estimate(
    self,
    func: Callable[[NDArray], NDArray],
) -> NDArray[np.float64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | *required* | Function to apply to trajectories |

**Returns**: `NDArray[np.float64]` — Weighted average of `func` applied to trajectories.

---

##### `to_dataframe()`

Convert smoothed estimates to a pandas DataFrame.

```python
def to_dataframe(
    self,
    state_names: list[str] | None = None,
) -> pd.DataFrame
```

---

## PFConfig

Configuration dataclass controlling particle filter behavior.

### Constructor

```python
@dataclass
class PFConfig:
    n_particles: int = 1000
    resampling: str = "systematic"
    ess_threshold: float = 0.5
    seed: int | None = None
    store_particles: bool = False
    store_ancestors: bool = False
    store_weights: bool = False
    log_level: str = "WARNING"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `1000` | Number of particles $N$ |
| `resampling` | `str` | `"systematic"` | Resampling method (`"systematic"`, `"multinomial"`, `"stratified"`, `"residual"`) |
| `ess_threshold` | `float` | `0.5` | Resample when $\text{ESS} / N < \text{threshold}$ |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `store_particles` | `bool` | `False` | Store full particle history (required for smoothers) |
| `store_ancestors` | `bool` | `False` | Store ancestor indices (required for some smoothers) |
| `store_weights` | `bool` | `False` | Store weight history (required for smoothers) |
| `log_level` | `str` | `"WARNING"` | Logging verbosity |

!!! warning
    Setting `store_particles=True` can use significant memory for long time series. For $T = 1000$, $N = 5000$, $k = 3$: approximately 114 MB.

### Methods

##### `validate()`

Check configuration for consistency and raise `ValueError` for invalid settings.

```python
def validate(self) -> None
```

---

##### `effective_threshold()`

Return the absolute ESS threshold (i.e., `ess_threshold * n_particles`).

```python
def effective_threshold(self) -> float
```

**Returns**: `float` — Absolute ESS threshold.

### Example

```python
from particlefilterbox.core import PFConfig

# Basic configuration
config = PFConfig(n_particles=5000, resampling='systematic')

# Full storage for smoothing
config_smooth = PFConfig(
    n_particles=2000,
    store_particles=True,
    store_weights=True,
    store_ancestors=True,
    seed=42,
)
config_smooth.validate()
```

---

## See Also

- [User Guide: ParticleCloud](../user-guide/core/particle-cloud.md) — In-depth usage guide
- [User Guide: Resampling](../user-guide/core/resampling.md) — Resampling concepts and tuning
- [User Guide: ESS](../user-guide/core/ess.md) — Effective sample size monitoring
- [Tutorials: Fundamentals](../tutorials/fundamentals.md) — Step-by-step introduction
- [Resampling API](resampling.md) — Resampling function reference
- [Filters API](filters.md) — Particle filter reference
