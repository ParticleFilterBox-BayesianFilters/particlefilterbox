---
title: "Models API"
description: "API reference for particlefilterbox.models — pre-built state-space models: Stochastic Volatility, DSGE, Jump-Diffusion, Regime-switching, Count, Bounded, Mixture, Continuous-time"
---

# Models API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.models import StochasticVolatility, DSGEModel, JumpDiffusion, RegimeModel, CountModel, BoundedModel, MixtureModel, ContinuousTimeModel`
    **Source**: `particlefilterbox/models/`

## Overview

The models module provides ready-to-use state-space models covering a wide range of econometric, financial, and scientific applications. All models inherit from `StateSpaceModel` and implement the required densities:

- **Initial**: $p(x_0)$
- **Transition**: $f(x_t \mid x_{t-1}, \theta)$
- **Observation**: $g(y_t \mid x_t, \theta)$

Each model exposes the standard interface used by all filters, smoothers, SMC, and PMCMC methods in the library.

| Model | Domain | Key feature |
|-------|--------|-------------|
| `StochasticVolatility` | Finance | Latent log-volatility, variants with leverage/jumps |
| `DSGEModel` | Macro | General DSGE with symbolic equations |
| `JumpDiffusion` | Finance | Merton-style continuous + jump dynamics |
| `RegimeModel` | General | Markov-switching latent regimes |
| `CountModel` | Epidemiology / reliability | Poisson / NegBin observations |
| `BoundedModel` | Ecology / physics | Constrained state via transforms |
| `MixtureModel` | General | Mixture-of-Gaussians transitions |
| `ContinuousTimeModel` | Finance / biology | SDE-driven states with Euler-Maruyama |

---

## StateSpaceModel (Base Class)

All pre-built models share a common base class. The most relevant inherited methods are:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `simulate` | `(T, rng) -> (x, y)` | Draw a synthetic trajectory |
| `log_initial` | `(x0) -> float` | Log-density of $p(x_0)$ |
| `sample_initial` | `(n, rng) -> NDArray` | Sample from $p(x_0)$ |
| `log_transition` | `(x_prev, x_next) -> NDArray` | $\log f(x_t \mid x_{t-1})$ |
| `sample_transition` | `(x_prev, rng) -> NDArray` | Sample from $f(x_t \mid x_{t-1})$ |
| `log_observation` | `(x, y) -> NDArray` | $\log g(y_t \mid x_t)$ |
| `with_params` | `(**kwargs) -> Self` | Return a copy with updated parameters |

---

## StochasticVolatility

Stochastic volatility model with latent log-variance. The basic variant follows Taylor (1982) / Kim, Shephard & Chib (1998):

$$
\begin{aligned}
x_t &= \mu + \phi (x_{t-1} - \mu) + \sigma_\eta \, \eta_t, \qquad \eta_t \sim \mathcal{N}(0, 1) \\
y_t &= \exp(x_t / 2) \, \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, 1)
\end{aligned}
$$

Four variants are supported via the `variant` argument.

### Constructor

```python
StochasticVolatility(
    mu: float = 0.0,
    phi: float = 0.95,
    sigma_eta: float = 0.2,
    variant: Literal["basic", "leverage", "t", "jumps"] = "basic",
    # Variant-specific parameters
    rho: float = 0.0,        # leverage correlation
    nu: float = 5.0,         # degrees of freedom for t
    jump_prob: float = 0.02, # jump probability for jumps
    jump_std: float = 0.5,   # jump size std for jumps
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu` | `float` | `0.0` | Long-run mean of log-volatility |
| `phi` | `float` | `0.95` | Persistence $\|\phi\| < 1$ for stationarity |
| `sigma_eta` | `float` | `0.2` | Volatility of log-volatility $\sigma_\eta > 0$ |
| `variant` | `str` | `"basic"` | One of `"basic"`, `"leverage"`, `"t"`, `"jumps"` |
| `rho` | `float` | `0.0` | Leverage correlation (leverage variant) |
| `nu` | `float` | `5.0` | Student-t degrees of freedom (t variant) |
| `jump_prob` | `float` | `0.02` | Per-period jump probability (jumps variant) |
| `jump_std` | `float` | `0.5` | Jump size standard deviation (jumps variant) |

### Variants

| Variant | Observation density | Use case |
|---------|---------------------|----------|
| `"basic"` | $y_t \mid x_t \sim \mathcal{N}(0, e^{x_t})$ | Standard SV |
| `"leverage"` | Correlated $(\eta_t, \epsilon_t)$ via $\rho$ | Negative return-volatility correlation |
| `"t"` | $y_t \mid x_t \sim t_\nu(0, e^{x_t/2})$ | Heavy-tailed returns |
| `"jumps"` | Normal + rare jumps | Crisis-like return behavior |

### Example

```python
import particlefilterbox as pfb

# Basic SV
model = pfb.models.StochasticVolatility(mu=0.0, phi=0.95, sigma_eta=0.2)

# Leverage variant
model_lev = pfb.models.StochasticVolatility(variant='leverage', rho=-0.5)

# Heavy-tailed
model_t = pfb.models.StochasticVolatility(variant='t', nu=7.0)

# With jumps
model_j = pfb.models.StochasticVolatility(variant='jumps', jump_prob=0.01, jump_std=1.0)

# Simulate
x, y = model.simulate(T=1000, rng=np.random.default_rng(42))
```

---

## DSGEModel

Linearized Dynamic Stochastic General Equilibrium (DSGE) model. Accepts user-supplied linear rational expectations equations and solves them using a Klein / Sims `gensys` solver, then wraps the solution in a state-space form suitable for particle filtering.

$$
\begin{aligned}
\xi_t &= A(\theta) \, \xi_{t-1} + B(\theta) \, \epsilon_t \\
y_t &= C(\theta) \, \xi_t + D(\theta) \, u_t
\end{aligned}
$$

### Constructor

```python
DSGEModel(
    equations: list[sympy.Expr],
    params: dict[str, float],
    shocks: list[str],
    controls: list[str],
    observables: list[str],
    steady_state: dict[str, float] | None = None,
    solver: Literal["gensys", "klein"] = "gensys",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `equations` | `list[sympy.Expr]` | *required* | Linearized model equations |
| `params` | `dict[str, float]` | *required* | Structural parameter values |
| `shocks` | `list[str]` | *required* | Names of structural shocks |
| `controls` | `list[str]` | *required* | Names of forward-looking controls |
| `observables` | `list[str]` | *required* | Names of observable variables |
| `steady_state` | `dict[str, float] \| None` | `None` | Steady-state values (if not computed by solver) |
| `solver` | `str` | `"gensys"` | RE solution method |

!!! tip
    A pre-configured small New-Keynesian model is available: `NewKeynesianModel()`. See the User Guide for the three-equation and seven-equation versions.

### Example

```python
import particlefilterbox as pfb
import sympy as sp

# Small New-Keynesian model
pi, y, r = sp.symbols('pi y r')
pi_lag, y_lag = sp.symbols('pi_lag y_lag')
e_pi, e_y, e_r = sp.symbols('e_pi e_y e_r')

equations = [
    pi - 0.7*pi_lag - 0.1*y - e_pi,
    y - 0.8*y_lag + 0.1*(r - pi) - e_y,
    r - 1.5*pi - 0.5*y - e_r,
]

model = pfb.models.DSGEModel(
    equations=equations,
    params={},
    shocks=['e_pi', 'e_y', 'e_r'],
    controls=['pi', 'y', 'r'],
    observables=['pi', 'y'],
)

# Or use the shortcut
model = pfb.models.NewKeynesianModel(variant='small_nk')
```

---

## JumpDiffusion

Merton-style jump-diffusion model (Merton, 1976). The log-asset follows a continuous drift-diffusion with compound Poisson jumps:

$$
d \log S_t = \mu \, dt + \sigma \, dW_t + J_t \, dN_t
$$

where $N_t$ is Poisson with intensity $\lambda$ and jump sizes $J_t \sim \mathcal{N}(\mu_J, \sigma_J^2)$.

Discretized over unit intervals, this becomes a Gaussian-mixture state-space model.

### Constructor

```python
JumpDiffusion(
    mu: float = 0.0,
    sigma: float = 0.1,
    jump_intensity: float = 0.05,
    jump_mean: float = 0.0,
    jump_std: float = 0.3,
    dt: float = 1.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu` | `float` | `0.0` | Drift of the continuous component |
| `sigma` | `float` | `0.1` | Diffusion coefficient $\sigma > 0$ |
| `jump_intensity` | `float` | `0.05` | Poisson rate $\lambda > 0$ |
| `jump_mean` | `float` | `0.0` | Mean jump size $\mu_J$ |
| `jump_std` | `float` | `0.3` | Jump size std $\sigma_J > 0$ |
| `dt` | `float` | `1.0` | Time-step size |

### Example

```python
import particlefilterbox as pfb

model = pfb.models.JumpDiffusion(
    mu=0.0003,
    sigma=0.012,
    jump_intensity=0.01,
    jump_mean=-0.02,
    jump_std=0.05,
)

x, y = model.simulate(T=2000, rng=np.random.default_rng(42))
```

---

## RegimeModel

Markov-switching state-space model. The latent state includes a discrete regime indicator $s_t \in \{1, \ldots, K\}$ evolving according to a transition matrix $\Pi$, with regime-dependent continuous dynamics and observations.

$$
\begin{aligned}
s_t \mid s_{t-1} &\sim \text{Cat}(\Pi_{s_{t-1}, \cdot}) \\
x_t \mid x_{t-1}, s_t &\sim f_{s_t}(x_t \mid x_{t-1}) \\
y_t \mid x_t, s_t &\sim g_{s_t}(y_t \mid x_t)
\end{aligned}
$$

### Constructor

```python
RegimeModel(
    n_regimes: int,
    transitions: NDArray[np.float64],
    dynamics: list[Callable],
    obs: list[Callable],
    initial_regime_probs: NDArray[np.float64] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_regimes` | `int` | *required* | Number of regimes $K$ |
| `transitions` | `NDArray[np.float64]` | *required* | Transition matrix $\Pi$, shape `(K, K)`, rows sum to 1 |
| `dynamics` | `list[Callable]` | *required* | Transition density per regime, length $K$ |
| `obs` | `list[Callable]` | *required* | Observation density per regime, length $K$ |
| `initial_regime_probs` | `NDArray[np.float64] \| None` | `None` | Initial regime distribution; defaults to stationary distribution of $\Pi$ |

### Example

```python
import numpy as np
import particlefilterbox as pfb

# Two-regime SV model (calm / crisis)
Pi = np.array([
    [0.98, 0.02],
    [0.10, 0.90],
])

dynamics = [
    lambda x_prev: np.random.normal(0.95 * x_prev, 0.1),   # calm
    lambda x_prev: np.random.normal(0.85 * x_prev, 0.3),   # crisis
]

obs = [
    lambda x: np.random.normal(0, np.exp(x / 2)),   # calm
    lambda x: np.random.normal(0, 3 * np.exp(x / 2)),  # crisis
]

model = pfb.models.RegimeModel(
    n_regimes=2,
    transitions=Pi,
    dynamics=dynamics,
    obs=obs,
)
```

!!! note
    For MCMC with known regime structure, Rao-Blackwellized filtering (RBPF) can marginalize the regime indicator analytically. See [RaoBlackwellizedPF](filters.md#raoblackwellizedpf).

---

## CountModel

State-space model for count / integer-valued observations. The observation distribution is a generalized linear model (Poisson, Negative Binomial, or Binomial) applied to a latent log-rate.

$$
\begin{aligned}
x_t &= \rho \, x_{t-1} + \sigma \, \eta_t \\
\lambda_t &= h^{-1}(x_t) \\
y_t &\sim \text{Dist}(\lambda_t)
\end{aligned}
$$

### Constructor

```python
CountModel(
    distribution: Literal["poisson", "negbin", "binomial"] = "poisson",
    link: Literal["log", "logit", "identity"] = "log",
    dynamics: Callable | dict = "ar1",
    ar_coef: float = 0.9,
    sigma: float = 0.1,
    # Distribution-specific
    overdispersion: float = 1.0,   # negbin
    n_trials: int = 1,              # binomial
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distribution` | `str` | `"poisson"` | Observation distribution |
| `link` | `str` | `"log"` | Link function relating latent state to rate/prob |
| `dynamics` | `Callable \| dict` | `"ar1"` | Latent state dynamics (string name or custom callable) |
| `ar_coef` | `float` | `0.9` | AR(1) coefficient (if dynamics is `"ar1"`) |
| `sigma` | `float` | `0.1` | Latent innovation std |
| `overdispersion` | `float` | `1.0` | Negative binomial dispersion parameter |
| `n_trials` | `int` | `1` | Number of trials (binomial) |

### Example

```python
import particlefilterbox as pfb

# Poisson count model with AR(1) log-rate
model = pfb.models.CountModel(
    distribution='poisson',
    link='log',
    ar_coef=0.95,
    sigma=0.15,
)

x, y = model.simulate(T=500, rng=np.random.default_rng(0))
print(f"Mean count: {y.mean():.2f}, max: {y.max()}")
```

---

## BoundedModel

State-space model where the latent state is constrained to a bounded interval (or rectangle) via an invertible transformation. The library automatically handles Jacobian corrections in log-densities.

$$
\tilde{x}_t = T(x_t), \qquad \tilde{x}_t \text{ evolves unconstrained}
$$

Supported transforms: logit (for $[a, b]$), log (for $[0, \infty)$), and softplus.

### Constructor

```python
BoundedModel(
    bounds: tuple[float, float] | NDArray[np.float64],
    transform: Literal["logit", "log", "softplus"] = "logit",
    dynamics: Callable | dict = "ar1",
    obs: Callable | dict = "gaussian",
    ar_coef: float = 0.9,
    sigma: float = 0.1,
    obs_std: float = 0.1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bounds` | `tuple[float, float] \| NDArray` | *required* | Lower/upper bounds for each state dimension |
| `transform` | `str` | `"logit"` | Transform mapping bounded → unconstrained space |
| `dynamics` | `Callable \| dict` | `"ar1"` | Dynamics on unconstrained scale |
| `obs` | `Callable \| dict` | `"gaussian"` | Observation model |
| `ar_coef` | `float` | `0.9` | AR(1) coefficient |
| `sigma` | `float` | `0.1` | Transition innovation std |
| `obs_std` | `float` | `0.1` | Observation noise std |

### Example

```python
import particlefilterbox as pfb

# Probability tracking: state in [0, 1]
model = pfb.models.BoundedModel(
    bounds=(0.0, 1.0),
    transform='logit',
    ar_coef=0.9,
    sigma=0.5,
    obs_std=0.05,
)

x, y = model.simulate(T=500, rng=np.random.default_rng(0))
assert (x >= 0).all() and (x <= 1).all()
```

!!! warning
    When using `BoundedModel` with `log_transition()`, the returned density is on the **bounded scale** (with Jacobian applied). Filters and smoothers handle this transparently, but custom code should account for it.

---

## MixtureModel

State-space model with mixture-of-Gaussians transition or observation density. Useful for modeling multimodal innovations (e.g., skewed or heavy-tailed distributions approximated by Gaussian mixtures).

$$
f(x_t \mid x_{t-1}) = \sum_{k=1}^{K} \pi_k(x_{t-1}) \, \mathcal{N}(x_t; \mu_k(x_{t-1}), \Sigma_k)
$$

### Constructor

```python
MixtureModel(
    n_components: int,
    dynamics: Callable | None = None,
    obs: Callable | None = None,
    weights: NDArray[np.float64] | None = None,
    means: NDArray[np.float64] | None = None,
    covs: NDArray[np.float64] | None = None,
    mixture_on: Literal["transition", "observation"] = "transition",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | `int` | *required* | Number of mixture components $K$ |
| `dynamics` | `Callable \| None` | `None` | Custom transition (ignored if `weights/means/covs` given) |
| `obs` | `Callable \| None` | `None` | Custom observation |
| `weights` | `NDArray[np.float64] \| None` | `None` | Mixture weights, shape `(K,)` |
| `means` | `NDArray[np.float64] \| None` | `None` | Component means, shape `(K, d)` |
| `covs` | `NDArray[np.float64] \| None` | `None` | Component covariances, shape `(K, d, d)` |
| `mixture_on` | `str` | `"transition"` | Which density is the mixture |

### Example

```python
import numpy as np
import particlefilterbox as pfb

# Two-component skewed mixture transition
model = pfb.models.MixtureModel(
    n_components=2,
    weights=np.array([0.8, 0.2]),
    means=np.array([[0.0], [2.0]]),
    covs=np.array([[[0.1]], [[0.5]]]),
    mixture_on='transition',
)
```

---

## ContinuousTimeModel

State-space model defined by a stochastic differential equation (SDE), discretized via Euler-Maruyama (or Milstein):

$$
dx_t = \mu(x_t, \theta) \, dt + \sigma(x_t, \theta) \, dW_t
$$

Observations are sampled at discrete time points $t_1, t_2, \ldots, t_T$ (possibly irregularly spaced).

### Constructor

```python
ContinuousTimeModel(
    drift: Callable,
    diffusion: Callable,
    dt: float | NDArray[np.float64] = 1.0,
    discretization: Literal["euler", "milstein"] = "euler",
    subdivisions: int = 1,
    obs: Callable | None = None,
    obs_std: float = 0.1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drift` | `Callable` | *required* | Drift function $\mu(x, \theta) \to$ `NDArray` |
| `diffusion` | `Callable` | *required* | Diffusion function $\sigma(x, \theta) \to$ `NDArray` |
| `dt` | `float \| NDArray[np.float64]` | `1.0` | Observation interval(s) |
| `discretization` | `str` | `"euler"` | Numerical scheme |
| `subdivisions` | `int` | `1` | Sub-steps per observation interval (reduces discretization bias) |
| `obs` | `Callable \| None` | `None` | Observation function (default: identity + Gaussian noise) |
| `obs_std` | `float` | `0.1` | Observation noise std |

### Example

```python
import numpy as np
import particlefilterbox as pfb

# Cox-Ingersoll-Ross interest rate model
def drift(x, theta):
    kappa, theta_bar, _ = theta
    return kappa * (theta_bar - x)

def diffusion(x, theta):
    _, _, sigma = theta
    return sigma * np.sqrt(np.maximum(x, 0.0))

model = pfb.models.ContinuousTimeModel(
    drift=drift,
    diffusion=diffusion,
    dt=1.0 / 252,         # daily observations
    subdivisions=4,        # 4 sub-steps per day
    obs_std=1e-4,
)
```

!!! tip
    Increase `subdivisions` for stiff SDEs or when the Euler-Maruyama bias is non-negligible. Typical values: 1 (smooth SDE), 4–16 (moderately stiff), 32+ (stiff).

---

## Model Comparison

| Model | State dim | Observation type | Typical application |
|-------|----------:|------------------|---------------------|
| `StochasticVolatility` | 1 | Continuous | Return volatility tracking |
| `DSGEModel` | 5–30 | Continuous (multivariate) | Macro-econometric analysis |
| `JumpDiffusion` | 1 | Continuous | Asset prices with crashes |
| `RegimeModel` | 1–10 | Any | Business cycles, fault detection |
| `CountModel` | 1–5 | Discrete integer | Disease cases, defect counts |
| `BoundedModel` | 1–5 | Continuous | Probabilities, concentrations |
| `MixtureModel` | 1–10 | Continuous | Multimodal innovations |
| `ContinuousTimeModel` | 1–10 | Continuous (irregular) | Financial term structure |

---

## See Also

- [User Guide: Models](../user-guide/models/index.md) — In-depth model usage
- [User Guide: Stochastic Volatility](../user-guide/models/stochastic-volatility.md) — SV model walkthrough
- [User Guide: DSGE](../user-guide/models/dsge.md) — DSGE model walkthrough
- [User Guide: Custom Models](../user-guide/models/custom-models.md) — Building your own models
- [Tutorials: SV Estimation](../tutorials/sv-estimation.md) — End-to-end SV workflow
- [Tutorials: Regime Switching](../tutorials/regime-switching.md) — Markov-switching tutorial
- [Core API](core.md) — `StateSpaceModel` base class
- [Filters API](filters.md) — Choosing a filter for your model
- [Datasets API](datasets.md) — Built-in datasets for testing
