---
title: "SMC API"
description: "API reference for particlefilterbox.smc — Sequential Monte Carlo samplers, SMC², IBIS, Waste-Free SMC, and adaptive tempering"
---

# SMC API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.smc import SMCSampler, SMCSquared, IBIS, WasteFreeSMC, SMCTempering`
    **Source**: `particlefilterbox/smc/`

## Overview

Sequential Monte Carlo (SMC) samplers generalize particle filters to sampling from arbitrary sequences of probability distributions. Instead of the filtering distribution $p(x_t \mid y_{1:t})$, SMC samplers target a user-specified sequence $\pi_0, \pi_1, \ldots, \pi_T$ — typically tempered posteriors or data-augmented distributions.

$$
\pi_t(\theta) \propto \gamma_t(\theta), \qquad t = 0, 1, \ldots, T
$$

Starting from samples of $\pi_0$ (often the prior), SMC propagates particles through intermediate distributions via MCMC kernels and reweights them to reach $\pi_T$ (the target posterior).

| Class | Target | Use case |
|-------|--------|----------|
| `SMCSampler` | Generic sequence $\{\pi_t\}$ | Custom SMC schedules |
| `SMCSquared` | Joint $(\theta, x_{1:T})$ | Online Bayesian parameter + state inference |
| `IBIS` | Static posterior $p(\theta \mid y_{1:T})$ | Sequential Bayesian updating |
| `WasteFreeSMC` | Static posterior | High-efficiency posterior sampling |
| `SMCTempering` | $\pi_T$ via tempering | Multimodal / stiff posteriors |

---

## SMCSampler

Generic Sequential Monte Carlo sampler for a user-defined sequence of target distributions. Implements the Del Moral, Doucet & Jasra (2006) framework with adaptive resampling and MCMC rejuvenation steps.

At each step $t$, particles $\{\theta^{(i)}\}$ are reweighted from $\pi_{t-1}$ to $\pi_t$, optionally resampled, and moved via an MCMC kernel that leaves $\pi_t$ invariant:

$$
W_t^{(i)} \propto W_{t-1}^{(i)} \cdot \frac{\gamma_t(\theta_{t-1}^{(i)})}{\gamma_{t-1}(\theta_{t-1}^{(i)})}
$$

### Constructor

```python
SMCSampler(
    target: SMCTarget,
    n_particles: int,
    n_steps: int,
    kernel: MCMCKernel,
    schedule: NDArray[np.float64] | str = "adaptive",
    ess_threshold: float = 0.5,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | `SMCTarget` | *required* | Target sequence object exposing `log_gamma(theta, t)` and `log_prior(theta)` |
| `n_particles` | `int` | *required* | Number of particles $N$ |
| `n_steps` | `int` | *required* | Number of intermediate distributions $T$ |
| `kernel` | `MCMCKernel` | *required* | MCMC kernel leaving $\pi_t$ invariant (e.g., `RandomWalkMH`, `HMC`) |
| `schedule` | `NDArray[np.float64] \| str` | `"adaptive"` | Tempering schedule or `"adaptive"` for ESS-based |
| `ess_threshold` | `float` | `0.5` | Resample when ESS drops below `ess_threshold * n_particles` |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

### Methods

##### `sample()`

Run the full SMC sampler and return particle approximation of the target $\pi_T$.

```python
def sample(self) -> SMCResult
```

**Returns**: `SMCResult` — Particle cloud targeting $\pi_T$, with log-normalizing-constant estimate and diagnostics.

**Raises**: `RuntimeError` if all particles degenerate (ESS collapses to 1) before reaching the target.

##### `normalizing_constant()`

Return the estimated log-normalizing-constant (log marginal likelihood) of the target distribution.

```python
def normalizing_constant(self) -> float
```

**Returns**: `float` — Estimate of $\log Z_T = \log \int \gamma_T(\theta) \, d\theta$.

!!! note
    The log-normalizing-constant is accumulated incrementally across SMC steps using the unbiased estimator of Del Moral et al. (2006).

### Example

```python
import numpy as np
import particlefilterbox as pfb

target = pfb.smc.BayesianTarget(
    log_prior=my_log_prior,
    log_likelihood=my_log_likelihood,
    data=observations,
)
kernel = pfb.smc.RandomWalkMH(step_size=0.1)

sampler = pfb.SMCSampler(
    target=target,
    n_particles=5000,
    n_steps=50,
    kernel=kernel,
    schedule="adaptive",
    rng=np.random.default_rng(42),
)

result = sampler.sample()
log_Z = sampler.normalizing_constant()
print(result.summary())
print(f"log marginal likelihood: {log_Z:.2f}")
```

---

## SMCSquared

SMC² (Chopin, Jacob & Papaspiliopoulos, 2013) performs joint Bayesian inference over parameters $\theta$ and latent states $x_{1:T}$ in state-space models. It runs an outer SMC over $\theta$-particles, where each outer particle carries an inner particle filter for $x_{1:T}$.

$$
p(\theta, x_{1:T} \mid y_{1:T}) \propto p(\theta) \, p(y_{1:T} \mid \theta) \, p(x_{1:T} \mid \theta, y_{1:T})
$$

The marginal likelihood $p(y_{1:T} \mid \theta)$ is unbiasedly estimated by the inner particle filter at each outer particle.

### Constructor

```python
SMCSquared(
    model: ParticleFilterModel,
    n_theta: int,
    n_x: int,
    ess_threshold: float = 0.5,
    priors: dict[str, Distribution] | None = None,
    mcmc_kernel: MCMCKernel | None = None,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model with parameter-dependent densities |
| `n_theta` | `int` | *required* | Number of outer parameter particles $N_\theta$ |
| `n_x` | `int` | *required* | Number of inner state particles $N_x$ |
| `ess_threshold` | `float` | `0.5` | Outer-ESS threshold for resample-move |
| `priors` | `dict[str, Distribution] \| None` | `None` | Prior distributions for each parameter |
| `mcmc_kernel` | `MCMCKernel \| None` | `None` | MCMC kernel for $\theta$-rejuvenation (default: adaptive RWM) |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

### Methods

##### `filter()`

Run SMC² online: for each new observation, update all inner filters and reweight outer $\theta$-particles.

```python
def filter(
    self,
    observations: NDArray[np.float64],
) -> SMC2Result
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observations` | `NDArray[np.float64]` | *required* | Observations, shape `(T,)` or `(T, k_obs)` |

**Returns**: `SMC2Result` — Posterior samples of $\theta$, filtered state trajectories, and log-marginal-likelihood estimate.

**Raises**: `RuntimeError` if inner filters consistently degenerate (signals model-data mismatch or too few $N_x$).

### Example

```python
import particlefilterbox as pfb
from scipy import stats

model = pfb.models.StochasticVolatility(variant='basic')

priors = {
    "mu":        stats.norm(0.0, 1.0),
    "phi":       stats.uniform(-1.0, 2.0),
    "sigma_eta": stats.gamma(2.0, scale=0.5),
}

smc2 = pfb.SMCSquared(
    model=model,
    n_theta=1000,
    n_x=500,
    priors=priors,
)

result = smc2.filter(observations)
print(result.parameter_summary())
```

!!! tip
    Choose $N_x$ so that the variance of the log-likelihood estimator at the posterior mode is around 1–2. Too few inner particles → outer chain degeneracy; too many → wasted compute.

---

## IBIS

Iterated Batch Importance Sampling (Chopin, 2002) is a sequential Bayesian updating scheme for **static** parameter models. Starting from prior samples, IBIS processes observations one at a time (or in batches), reweighting and rejuvenating $\theta$-particles via MCMC when ESS drops.

$$
p(\theta \mid y_{1:t}) \propto p(\theta \mid y_{1:t-1}) \cdot p(y_t \mid \theta, y_{1:t-1})
$$

Unlike SMC², IBIS assumes the likelihood $p(y_{1:t} \mid \theta)$ is available in closed form (no latent states).

### Constructor

```python
IBIS(
    model: StaticModel,
    n_particles: int,
    mcmc_steps: int = 5,
    ess_threshold: float = 0.5,
    priors: dict[str, Distribution] | None = None,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `StaticModel` | *required* | Model with tractable log-likelihood `log_likelihood(theta, y)` |
| `n_particles` | `int` | *required* | Number of $\theta$-particles |
| `mcmc_steps` | `int` | `5` | MCMC rejuvenation steps after each resample |
| `ess_threshold` | `float` | `0.5` | ESS threshold for triggering rejuvenation |
| `priors` | `dict[str, Distribution] \| None` | `None` | Prior distributions |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

### Methods

##### `filter()`

Run IBIS over the observation sequence, updating the posterior sequentially.

```python
def filter(
    self,
    observations: NDArray[np.float64],
) -> IBISResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observations` | `NDArray[np.float64]` | *required* | Observations, shape `(T,)` or `(T, k)` |

**Returns**: `IBISResult` — Posterior $\theta$-samples at each $t$, log-marginal-likelihood, rejuvenation history.

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StaticRegression(features=X)
ibis = pfb.IBIS(
    model=model,
    n_particles=2000,
    mcmc_steps=10,
)
result = ibis.filter(y)
print(result.summary())
```

---

## WasteFreeSMC

Waste-Free SMC (Dau & Chopin, 2022) improves the efficiency of standard SMC samplers by keeping **all** intermediate MCMC states as particles, rather than discarding them. Given $n_{\text{mcmc}}$ MCMC steps per SMC step, each parent particle produces $n_{\text{mcmc}}$ children that all contribute to the next target.

For the same compute budget, Waste-Free SMC typically reduces Monte Carlo variance by 2–4× compared to standard SMC.

### Constructor

```python
WasteFreeSMC(
    target: SMCTarget,
    n_particles: int,
    n_mcmc_steps: int,
    kernel: MCMCKernel,
    schedule: NDArray[np.float64] | str = "adaptive",
    ess_threshold: float = 0.5,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | `SMCTarget` | *required* | Target sequence |
| `n_particles` | `int` | *required* | Number of parent particles $M$ (total particles = $M \cdot n_{\text{mcmc}}$) |
| `n_mcmc_steps` | `int` | *required* | MCMC steps per SMC iteration (children per parent) |
| `kernel` | `MCMCKernel` | *required* | MCMC kernel |
| `schedule` | `NDArray[np.float64] \| str` | `"adaptive"` | Tempering schedule |
| `ess_threshold` | `float` | `0.5` | ESS threshold |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

### Methods

##### `sample()`

```python
def sample(self) -> SMCResult
```

**Returns**: `SMCResult` — Posterior samples with log-normalizing-constant estimate.

### Example

```python
import particlefilterbox as pfb

target = pfb.smc.BayesianTarget(
    log_prior=log_prior,
    log_likelihood=log_likelihood,
    data=observations,
)

sampler = pfb.WasteFreeSMC(
    target=target,
    n_particles=500,
    n_mcmc_steps=20,
    kernel=pfb.smc.RandomWalkMH(step_size=0.1),
)

result = sampler.sample()
print(f"Effective total particles: {500 * 20}")
print(result.summary())
```

!!! tip
    For fixed total compute $N_{\text{total}} = M \cdot n_{\text{mcmc}}$, Waste-Free SMC with $n_{\text{mcmc}} \in [10, 50]$ typically outperforms standard SMC ($n_{\text{mcmc}} = 1$).

---

## SMCTempering

Adaptive tempering SMC sampler that automatically selects the tempering schedule to maintain a target ESS at each step. Particularly effective for multimodal or stiff posteriors where a geometric schedule is inadequate.

The tempered target at step $t$ is:

$$
\pi_t(\theta) \propto p(\theta) \, L(\theta)^{\beta_t}, \qquad 0 = \beta_0 < \beta_1 < \cdots < \beta_T = 1
$$

where $L(\theta)$ is the likelihood and $\beta_t$ is chosen adaptively to satisfy $\text{ESS}(W_t) = \alpha \cdot N$.

### Constructor

```python
SMCTempering(
    target: SMCTarget,
    n_particles: int,
    ess_target: float = 0.9,
    kernel: MCMCKernel | None = None,
    mcmc_steps: int = 5,
    max_steps: int = 500,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | `SMCTarget` | *required* | Target with `log_prior()` and `log_likelihood()` |
| `n_particles` | `int` | *required* | Number of particles |
| `ess_target` | `float` | `0.9` | Target ESS fraction (0–1) for adaptive schedule |
| `kernel` | `MCMCKernel \| None` | `None` | MCMC kernel (default: adaptive RWM) |
| `mcmc_steps` | `int` | `5` | MCMC steps after each reweighting |
| `max_steps` | `int` | `500` | Maximum number of tempering steps |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `temperature_schedule` | `NDArray[np.float64]` | Schedule $\{\beta_t\}$ selected adaptively, shape `(n_steps,)` |

### Methods

##### `sample()`

```python
def sample(self) -> SMCResult
```

**Returns**: `SMCResult` — Posterior samples and log-marginal-likelihood estimate.

### Example

```python
import particlefilterbox as pfb

target = pfb.smc.BayesianTarget(
    log_prior=log_prior,
    log_likelihood=log_likelihood,
    data=observations,
)

sampler = pfb.SMCTempering(
    target=target,
    n_particles=3000,
    ess_target=0.9,
)

result = sampler.sample()
print(f"Adaptive schedule: {sampler.temperature_schedule}")
print(f"Number of steps: {len(sampler.temperature_schedule)}")
print(result.summary())
```

!!! note
    `ess_target` closer to 1.0 produces a finer schedule (more SMC steps, more compute) but smoother transitions. Values in `[0.7, 0.95]` are typical.

---

## Result Types

### `SMCResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `particles` | `NDArray[np.float64]` | Final particles, shape `(N, d)` |
| `log_weights` | `NDArray[np.float64]` | Log-weights, shape `(N,)` |
| `log_normalizing_constant` | `float` | Estimated $\log Z$ |
| `schedule` | `NDArray[np.float64]` | Tempering schedule used |
| `ess_history` | `NDArray[np.float64]` | ESS at each step |

### `SMC2Result`

Extends `SMCResult` with per-time-step posteriors:

| Attribute | Type | Description |
|-----------|------|-------------|
| `theta_samples` | `NDArray[np.float64]` | $\theta$-samples at final time, shape `(N_theta, d_theta)` |
| `filtered_states` | `NDArray[np.float64]` | Mean filtered state per time, shape `(T, d_x)` |
| `log_marginal_likelihood` | `float` | Estimated $\log p(y_{1:T})$ |

### `IBISResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `theta_samples` | `NDArray[np.float64]` | Posterior $\theta$-samples, shape `(N, d_theta)` |
| `log_weights` | `NDArray[np.float64]` | Log-weights, shape `(N,)` |
| `log_marginal_likelihood` | `float` | $\log p(y_{1:T})$ estimate |
| `rejuvenation_times` | `list[int]` | Time steps at which MCMC rejuvenation was triggered |

---

## See Also

- [User Guide: SMC Samplers](../user-guide/smc/index.md) — In-depth usage guide
- [User Guide: SMC²](../user-guide/smc/smc2.md) — SMC² walkthrough
- [User Guide: IBIS](../user-guide/smc/ibis.md) — IBIS walkthrough
- [Tutorials: Waste-Free SMC](../tutorials/waste-free-smc.md) — Step-by-step tutorial
- [Theory: Sequential Monte Carlo](../theory/smc-theory.md) — Mathematical foundations
- [PMCMC API](pmcmc.md) — Particle MCMC methods (complementary approach)
- [Core API](core.md) — Base classes and configurations
- [Diagnostics API](diagnostics.md) — Convergence and marginal-likelihood diagnostics
