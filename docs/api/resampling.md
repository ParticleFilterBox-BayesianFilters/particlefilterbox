---
title: "Resampling API"
description: "API reference for particlefilterbox.resampling — systematic, multinomial, stratified, residual, and adaptive resampling"
---

# Resampling API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.resampling import systematic_resample, multinomial_resample, stratified_resample, residual_resample`
    **Source**: `particlefilterbox/resampling/`

## Overview

Resampling is the mechanism that prevents particle degeneracy by duplicating high-weight particles and eliminating low-weight ones. The resampling module provides multiple algorithms with different variance-efficiency trade-offs.

All resampling functions share a common interface: they accept normalized weights and a random generator, and return ancestor indices.

| Function | Complexity | Variance | Description |
|----------|-----------|----------|-------------|
| `systematic_resample` | $O(N)$ | Low | Low-variance, single uniform draw (recommended default) |
| `multinomial_resample` | $O(N)$ | High | Independent categorical draws |
| `stratified_resample` | $O(N)$ | Low | Stratified uniform draws |
| `residual_resample` | $O(N)$ | Low | Deterministic bulk + stochastic residual |
| `optimal_transport_resample` | $O(N^2)$ | Minimal | Minimizes particle displacement |
| `killing_resample` | $O(N)$ | High | Survival-based elimination |

---

## systematic_resample

Low-variance resampling using a single uniform draw with evenly spaced increments. The default and generally recommended method.

Given normalized weights $W^{(1)}, \ldots, W^{(N)}$, draw $U \sim \text{Uniform}(0, 1/N)$ and select particle $i$ when the cumulative sum of weights crosses $(U + (j-1)/N)$ for $j = 1, \ldots, N$.

```python
def systematic_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `NDArray[np.float64]` | *required* | Normalized weights, shape `(N,)`, must sum to 1 |
| `rng` | `np.random.Generator` | *required* | Random number generator |

**Returns**: `NDArray[np.int64]` — Ancestor indices, shape `(N,)`.

**Raises**: `ValueError` if weights do not sum to approximately 1.

**Complexity**: $O(N)$ time, $O(N)$ space.

### Example

```python
import numpy as np
from particlefilterbox.resampling import systematic_resample

rng = np.random.default_rng(42)
weights = np.array([0.1, 0.3, 0.05, 0.55])
indices = systematic_resample(weights, rng)
print(indices)  # e.g., [1, 3, 3, 3]
```

---

## multinomial_resample

Classical resampling: draw $N$ independent samples from the categorical distribution defined by the weights. Simple but has higher variance than systematic/stratified methods.

$$
a^{(j)} \sim \text{Categorical}(W^{(1)}, \ldots, W^{(N)}), \quad j = 1, \ldots, N
$$

```python
def multinomial_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `NDArray[np.float64]` | *required* | Normalized weights, shape `(N,)` |
| `rng` | `np.random.Generator` | *required* | Random number generator |

**Returns**: `NDArray[np.int64]` — Ancestor indices, shape `(N,)`.

**Complexity**: $O(N)$ time (using alias method), $O(N)$ space.

### Example

```python
import numpy as np
from particlefilterbox.resampling import multinomial_resample

rng = np.random.default_rng(42)
weights = np.array([0.1, 0.3, 0.05, 0.55])
indices = multinomial_resample(weights, rng)
print(indices)  # e.g., [1, 3, 3, 3]
```

---

## stratified_resample

Stratified resampling divides $[0, 1)$ into $N$ equal strata and draws one uniform sample per stratum. This guarantees at least $\lfloor N W^{(i)} \rfloor$ copies of particle $i$.

$$
U^{(j)} \sim \text{Uniform}\!\left(\frac{j-1}{N}, \frac{j}{N}\right), \quad j = 1, \ldots, N
$$

```python
def stratified_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `NDArray[np.float64]` | *required* | Normalized weights, shape `(N,)` |
| `rng` | `np.random.Generator` | *required* | Random number generator |

**Returns**: `NDArray[np.int64]` — Ancestor indices, shape `(N,)`.

**Complexity**: $O(N)$ time, $O(N)$ space.

### Example

```python
import numpy as np
from particlefilterbox.resampling import stratified_resample

rng = np.random.default_rng(42)
weights = np.array([0.1, 0.3, 0.05, 0.55])
indices = stratified_resample(weights, rng)
print(indices)  # e.g., [1, 3, 3, 3]
```

---

## residual_resample

Two-phase resampling: first assign $\lfloor N W^{(i)} \rfloor$ copies deterministically, then resample the residual counts stochastically. Combines low variance with exact first-moment matching.

$$
n_i = \lfloor N W^{(i)} \rfloor + \tilde{n}_i, \qquad \tilde{n}_i \sim \text{Multinomial}\!\left(N - \sum_j \lfloor N W^{(j)} \rfloor, \; \tilde{W}\right)
$$

where $\tilde{W}^{(i)} \propto N W^{(i)} - \lfloor N W^{(i)} \rfloor$.

```python
def residual_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `NDArray[np.float64]` | *required* | Normalized weights, shape `(N,)` |
| `rng` | `np.random.Generator` | *required* | Random number generator |

**Returns**: `NDArray[np.int64]` — Ancestor indices, shape `(N,)`.

**Complexity**: $O(N)$ time, $O(N)$ space.

### Example

```python
import numpy as np
from particlefilterbox.resampling import residual_resample

rng = np.random.default_rng(42)
weights = np.array([0.1, 0.3, 0.05, 0.55])
indices = residual_resample(weights, rng)
print(indices)  # e.g., [1, 3, 3, 3]
```

---

## optimal_transport_resample

Resampling that minimizes particle displacement by solving an optimal transport problem. Preserves spatial structure better than other methods, but at higher computational cost.

```python
def optimal_transport_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `NDArray[np.float64]` | *required* | Normalized weights, shape `(N,)` |
| `rng` | `np.random.Generator` | *required* | Random number generator |

**Returns**: `NDArray[np.int64]` — Ancestor indices, shape `(N,)`.

**Complexity**: $O(N^2)$ time, $O(N)$ space.

!!! tip
    Use optimal transport resampling when particle positions carry geometric meaning (e.g., spatial models) and the additional cost is acceptable.

---

## killing_resample

Survival-based resampling where each particle independently survives with probability proportional to its weight. Simple but can produce variable output sizes — internally padded to $N$.

```python
def killing_resample(
    log_weights: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_weights` | `NDArray[np.float64]` | *required* | **Unnormalized** log-weights, shape `(N,)` |
| `rng` | `np.random.Generator` | *required* | Random number generator |

**Returns**: `NDArray[np.int64]` — Ancestor indices, shape `(N,)`.

!!! note
    Unlike the other resampling functions, `killing_resample` takes **log-weights** (unnormalized), not normalized weights.

---

## Utility Functions

### get_resampling_fn

Registry lookup that returns a resampling function by name.

```python
def get_resampling_fn(
    method: str,
) -> Callable[[NDArray, np.random.Generator], NDArray[np.int64]]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | *required* | Method name: `"systematic"`, `"multinomial"`, `"stratified"`, `"residual"`, `"optimal_transport"`, `"killing"` |

**Returns**: The corresponding resampling function.

**Raises**: `ValueError` if `method` is not recognized.

```python
from particlefilterbox.resampling import get_resampling_fn

resample_fn = get_resampling_fn("systematic")
indices = resample_fn(weights, rng)
```

---

### should_resample

Check whether resampling should be triggered based on the ESS threshold.

```python
def should_resample(
    ess: float,
    threshold: float,
) -> bool
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ess` | `float` | *required* | Current effective sample size |
| `threshold` | `float` | *required* | Absolute ESS threshold |

**Returns**: `bool` — `True` if `ess < threshold`.

---

### adaptive_resample

Conditionally resample only when ESS falls below a threshold. Combines `should_resample` with the actual resampling step.

```python
def adaptive_resample(
    particles: NDArray[np.float64],
    log_weights: NDArray[np.float64],
    threshold: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `particles` | `NDArray[np.float64]` | *required* | Current particles, shape `(N, k)` |
| `log_weights` | `NDArray[np.float64]` | *required* | Unnormalized log-weights, shape `(N,)` |
| `threshold` | `float` | *required* | Absolute ESS threshold |
| `rng` | `np.random.Generator` | *required* | Random generator |

**Returns**: `tuple` — `(resampled_particles, new_log_weights, did_resample)`.

---

## Comparison

!!! tip "Choosing a Resampling Method"
    For most applications, **systematic resampling** provides the best trade-off between variance and computational cost. Use **stratified** when you need independent strata guarantees, **residual** for exact first-moment matching, and **optimal transport** for spatial models where particle positions are geometrically meaningful.

| Property | Systematic | Multinomial | Stratified | Residual |
|----------|-----------|-------------|------------|----------|
| Variance | Low | High | Low | Low |
| $\mathbb{E}[n_i] = N W^{(i)}$ | Yes | Yes | Yes | Yes |
| $n_i \geq \lfloor NW^{(i)} \rfloor$ | No | No | Yes | Yes |
| Computational cost | $O(N)$ | $O(N)$ | $O(N)$ | $O(N)$ |
| Independence | No | Yes | Partial | Partial |
| Recommended for | Default | Theoretical analysis | Guaranteed copies | Low variance |

---

## See Also

- [User Guide: Resampling](../user-guide/core/resampling.md) — Conceptual guide to resampling
- [User Guide: ESS](../user-guide/core/ess.md) — When and why to resample
- [Core API](core.md) — `ParticleCloud.resample()` method
- [Theory: Particle Filters](../theory/particle-filter-theory.md) — Mathematical foundations
- [Acceleration: Numba](../acceleration/numba.md) — JIT-compiled resampling kernels
