---
title: "Datasets API"
description: "API reference for particlefilterbox.datasets — built-in financial/macro datasets and simulators for SV, DSGE, jump-diffusion, and regime-switching models"
---

# Datasets API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.datasets import load_sp500, load_exchange_rates, simulate_sv, simulate_dsge, simulate_jump_diffusion, simulate_regime`
    **Source**: `particlefilterbox/datasets/`

## Overview

The datasets module exposes two kinds of helpers:

- **Loaders** — curated empirical datasets (equity returns, FX rates) bundled with the package.
- **Simulators** — fast Monte Carlo generators for the most common state-space specifications (SV, DSGE, jump-diffusion, regime-switching), convenient for unit tests, benchmarks, and tutorials.

All helpers return a `DatasetBundle`-like container with:

| Field | Type | Description |
|-------|------|-------------|
| `y` | `NDArray[np.float64]` | Observed series, shape `(T,)` or `(T, k_obs)` |
| `x` | `NDArray[np.float64] \| None` | Latent states (simulators only) |
| `params` | `dict[str, Any]` | Parameters used / metadata |
| `dates` | `pd.DatetimeIndex \| None` | Timestamps (loaders only) |
| `description` | `str` | Short textual description |

| Function | Kind | Returns |
|----------|------|---------|
| `load_sp500()` | Loader | Daily S&P 500 log-returns |
| `load_exchange_rates()` | Loader | Major-pair FX log-returns |
| `simulate_sv()` | Simulator | Stochastic-volatility series |
| `simulate_dsge()` | Simulator | Small-scale DSGE series |
| `simulate_jump_diffusion()` | Simulator | Merton-style jump-diffusion series |
| `simulate_regime()` | Simulator | Markov-switching series |

---

## Loaders

### `load_sp500()`

Daily S&P 500 log-returns.

```python
def load_sp500(
    start: str | None = None,
    end: str | None = None,
    as_returns: bool = True,
    frequency: str = "daily",
) -> DatasetBundle
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `str \| None` | `None` | ISO date (inclusive). `None` = earliest available |
| `end` | `str \| None` | `None` | ISO date (inclusive). `None` = latest available |
| `as_returns` | `bool` | `True` | Return log-returns. If `False`, returns price levels |
| `frequency` | `str` | `"daily"` | `"daily"`, `"weekly"`, `"monthly"` |

**Returns**: `DatasetBundle` with `y` = returns (or prices) and `dates`.

**Example:**

```python
from particlefilterbox.datasets import load_sp500

data = load_sp500(start="2010-01-01", end="2025-12-31")
print(data.y.shape, data.dates[:3])
```

---

### `load_exchange_rates()`

Major FX-pair log-returns (EUR/USD, GBP/USD, USD/JPY, USD/CHF).

```python
def load_exchange_rates(
    pair: str = "EURUSD",
    start: str | None = None,
    end: str | None = None,
    as_returns: bool = True,
    frequency: str = "daily",
) -> DatasetBundle
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pair` | `str` | `"EURUSD"` | `"EURUSD"`, `"GBPUSD"`, `"USDJPY"`, `"USDCHF"`, or `"all"` |
| `start`, `end` | `str \| None` | `None` | Date range |
| `as_returns` | `bool` | `True` | Log-returns vs. levels |
| `frequency` | `str` | `"daily"` | `"daily"`, `"weekly"`, `"monthly"` |

**Example:**

```python
from particlefilterbox.datasets import load_exchange_rates

fx = load_exchange_rates(pair="all", start="2015-01-01")
```

---

## Simulators

All simulators accept a `seed` argument (`int | np.random.Generator | None`). When `seed` is an integer, a fresh `np.random.default_rng(seed)` is constructed internally.

### `simulate_sv()`

Stochastic-volatility model:

$$
\begin{aligned}
h_t &= \mu + \phi (h_{t-1} - \mu) + \sigma \eta_t, \quad \eta_t \sim \mathcal{N}(0, 1) \\
y_t &= \exp(h_t / 2)\, \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)
\end{aligned}
$$

```python
def simulate_sv(
    T: int = 500,
    params: dict[str, float] | None = None,
    variant: str = "basic",
    seed: int | np.random.Generator | None = None,
) -> DatasetBundle
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `T` | `int` | `500` | Series length |
| `params` | `dict \| None` | `None` | Overrides defaults: `mu=-0.5, phi=0.95, sigma=0.3, rho=0.0` |
| `variant` | `str` | `"basic"` | `"basic"`, `"leverage"`, `"t"`, `"jump"` |
| `seed` | `int \| Generator \| None` | `None` | RNG seed |

**Returns**: `DatasetBundle` with `y` = returns, `x` = log-volatility path.

**Example:**

```python
from particlefilterbox.datasets import simulate_sv

data = simulate_sv(T=1000, params={"phi": 0.98, "sigma": 0.2}, seed=42)
```

---

### `simulate_dsge()`

Small-scale DSGE model (three-equation New Keynesian by default).

```python
def simulate_dsge(
    T: int = 200,
    params: dict[str, float] | None = None,
    variant: str = "nk3",
    seed: int | np.random.Generator | None = None,
) -> DatasetBundle
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `T` | `int` | `200` | Series length |
| `params` | `dict \| None` | `None` | DSGE structural parameters |
| `variant` | `str` | `"nk3"` | `"nk3"` (3-eq NK), `"rbc"`, `"smets_wouters_small"` |
| `seed` | `int \| Generator \| None` | `None` | RNG seed |

**Default parameters (`nk3`):**

| Param | Value | Role |
|-------|-------|------|
| `beta` | `0.99` | Discount factor |
| `kappa` | `0.17` | NK Phillips slope |
| `sigma` | `1.0` | Risk aversion |
| `phi_pi` | `1.5` | Taylor rule, inflation |
| `phi_y` | `0.5` | Taylor rule, output gap |
| `rho_a`, `rho_g`, `rho_r` | `0.9, 0.8, 0.5` | Shock persistence |
| `sigma_a`, `sigma_g`, `sigma_r` | `0.005, 0.01, 0.003` | Shock std |

**Returns**: `DatasetBundle` with `y` = observables `(π, y, r)`, `x` = latent state trajectory.

---

### `simulate_jump_diffusion()`

Merton-style jump-diffusion:

$$
d\log S_t = \mu\, dt + \sigma\, dW_t + J_t\, dN_t, \quad J_t \sim \mathcal{N}(\mu_J, \sigma_J^2), \quad N_t \sim \text{Poisson}(\lambda\, dt)
$$

```python
def simulate_jump_diffusion(
    T: int = 500,
    params: dict[str, float] | None = None,
    variant: str = "merton",
    dt: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> DatasetBundle
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `T` | `int` | `500` | Series length |
| `params` | `dict \| None` | `None` | `mu=0.0, sigma=0.01, lambda_j=0.05, mu_j=0.0, sigma_j=0.03` |
| `variant` | `str` | `"merton"` | `"merton"`, `"bates"` (stochastic-vol + jumps), `"svcj"` |
| `dt` | `float` | `1.0` | Time increment (days) |
| `seed` | `int \| Generator \| None` | `None` | RNG seed |

**Returns**: `DatasetBundle` with `y` = log-returns, `x` = jump indicator and (optionally) volatility path.

---

### `simulate_regime()`

Markov-switching autoregression:

$$
y_t = \mu_{s_t} + \phi_{s_t}(y_{t-1} - \mu_{s_{t-1}}) + \sigma_{s_t} \varepsilon_t, \quad s_t \sim P
$$

```python
def simulate_regime(
    T: int = 500,
    params: dict[str, Any] | None = None,
    n_regimes: int = 2,
    seed: int | np.random.Generator | None = None,
) -> DatasetBundle
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `T` | `int` | `500` | Series length |
| `params` | `dict \| None` | `None` | `mu`, `phi`, `sigma` arrays of length `n_regimes`; transition matrix `P` shape `(n_regimes, n_regimes)` |
| `n_regimes` | `int` | `2` | Number of hidden regimes |
| `seed` | `int \| Generator \| None` | `None` | RNG seed |

**Returns**: `DatasetBundle` with `y` = returns, `x` = regime indicator $s_t$.

**Example:**

```python
import numpy as np
from particlefilterbox.datasets import simulate_regime

params = {
    "mu": np.array([0.0, 0.0]),
    "phi": np.array([0.5, 0.9]),
    "sigma": np.array([0.01, 0.05]),
    "P": np.array([[0.97, 0.03], [0.05, 0.95]]),
}
data = simulate_regime(T=1000, params=params, n_regimes=2, seed=7)
```

---

## Using Datasets with Filters

```python
from particlefilterbox.datasets import simulate_sv
from particlefilterbox.filters import BootstrapPF
from particlefilterbox.models import StochasticVolatility
from particlefilterbox import PFConfig

data = simulate_sv(T=500, seed=0)

model = StochasticVolatility(variant="basic", **data.params)
pf = BootstrapPF(model, PFConfig(n_particles=2000))
result = pf.filter(data.y)

# Validate against ground truth
from particlefilterbox.visualization import plot_filtered_vs_true
plot_filtered_vs_true(result, true_state=data.x)
```

---

## See Also

- [Models API](models.md) — state-space specifications consumed by simulators
- [Experiment API](experiment.md) — batch simulation for benchmarks
- [Tutorials](../tutorials/index.md) — worked examples using these datasets
