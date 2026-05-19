---
title: "PMCMC API"
description: "API reference for particlefilterbox.pmcmc — Particle Marginal Metropolis-Hastings, Particle Gibbs, PGAS, Conditional SMC, and MCMCChain"
---

# PMCMC API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.pmcmc import PMMH, ParticleGibbs, PGAS, ConditionalSMC, MCMCChain`
    **Source**: `particlefilterbox/pmcmc/`

## Overview

Particle Markov chain Monte Carlo (PMCMC) methods (Andrieu, Doucet & Holenstein, 2010) combine MCMC and particle filters to sample from the joint posterior over parameters $\theta$ and latent state trajectories $x_{1:T}$ in state-space models:

$$
p(\theta, x_{1:T} \mid y_{1:T}) \propto p(\theta) \, p(x_{1:T}, y_{1:T} \mid \theta)
$$

PMCMC methods use a particle filter as a proposal mechanism inside a Metropolis-Hastings or Gibbs sampler. The key theoretical property is that the **marginal** chain over $\theta$ has the correct posterior as its invariant distribution, even when the particle filter uses a finite number of particles.

| Method | What it samples | Best for |
|--------|----------------|----------|
| `PMMH` | $\theta$ only (marginal) | Low-dimensional $\theta$, expensive PF |
| `ParticleGibbs` | $(\theta, x_{1:T})$ jointly | Moderate-to-high dimensional $\theta$ |
| `PGAS` | $(\theta, x_{1:T})$ with ancestor sampling | Same as PG, better mixing |
| `ConditionalSMC` | $x_{1:T}$ conditional on $\theta$ | Inner kernel for PG/PGAS |

---

## PMMH

Particle Marginal Metropolis-Hastings (Andrieu, Doucet & Holenstein, 2010). Uses a random-walk proposal on $\theta$ and accepts/rejects based on the **particle-filter estimate** of the marginal likelihood $\hat{p}(y_{1:T} \mid \theta)$, which is an unbiased estimator.

The acceptance ratio is:

$$
\alpha(\theta, \theta') = \min\left(1, \, \frac{\hat{p}(y_{1:T} \mid \theta') \, p(\theta') \, q(\theta \mid \theta')}{\hat{p}(y_{1:T} \mid \theta) \, p(\theta) \, q(\theta' \mid \theta)}\right)
$$

### Constructor

```python
PMMH(
    model: ParticleFilterModel,
    n_particles: int,
    n_iterations: int,
    proposal: MCMCProposal,
    priors: dict[str, Distribution],
    burnin: int = 0,
    filter_class: type[BaseParticleFilter] = BootstrapPF,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | Parameter-dependent state-space model |
| `n_particles` | `int` | *required* | Number of particles $N$ for the inner filter |
| `n_iterations` | `int` | *required* | Number of MCMC iterations (post-burn-in) |
| `proposal` | `MCMCProposal` | *required* | Proposal distribution (e.g., `RandomWalkProposal`, `AdaptiveRWM`) |
| `priors` | `dict[str, Distribution]` | *required* | Prior distributions for parameters |
| `burnin` | `int` | `0` | Burn-in iterations to discard |
| `filter_class` | `type[BaseParticleFilter]` | `BootstrapPF` | Particle filter class for likelihood estimation |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `acceptance_rate` | `float` | Overall acceptance rate (after sampling) |

### Methods

##### `sample()`

Run the PMMH sampler and return the posterior chain over $\theta$.

```python
def sample(
    self,
    observations: NDArray[np.float64],
) -> MCMCChain
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observations` | `NDArray[np.float64]` | *required* | Observations, shape `(T,)` or `(T, k_obs)` |

**Returns**: `MCMCChain` — Parameter samples, log-posteriors, and acceptance diagnostics.

**Raises**: `RuntimeError` if the particle filter consistently degenerates (all weights zero).

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

proposal = pfb.pmcmc.AdaptiveRWM(
    initial_cov=0.01 * np.eye(3),
    adapt_until=2000,
)

pmmh = pfb.PMMH(
    model=model,
    n_particles=500,
    n_iterations=10_000,
    proposal=proposal,
    priors=priors,
    burnin=2_000,
)

chain = pmmh.sample(observations)
print(f"Acceptance rate: {pmmh.acceptance_rate:.2%}")
print(chain.summary())
```

!!! tip
    Choose $N$ so that the variance of $\log \hat{p}(y_{1:T} \mid \theta)$ at the posterior mode is around $1.0$. This minimizes asymptotic variance of the resulting chain (Pitt, Silva, Giordani & Kohn, 2012).

---

## ParticleGibbs

Particle Gibbs (Andrieu, Doucet & Holenstein, 2010) alternates between:

1. Sampling the trajectory $x_{1:T}$ conditional on $\theta$ using **Conditional SMC** (CSMC), which always retains a reference trajectory.
2. Sampling $\theta$ conditional on $x_{1:T}$ using standard MCMC (typically Gibbs or Metropolis).

$$
\begin{aligned}
x_{1:T} &\sim \text{CSMC}(\cdot \mid \theta, y_{1:T}, x_{1:T}^{\text{ref}}) \\
\theta &\sim p(\theta \mid x_{1:T}, y_{1:T})
\end{aligned}
$$

### Constructor

```python
ParticleGibbs(
    model: ParticleFilterModel,
    n_particles: int,
    n_iterations: int,
    priors: dict[str, Distribution],
    theta_sampler: Callable | None = None,
    burnin: int = 0,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `n_particles` | `int` | *required* | Number of particles in CSMC |
| `n_iterations` | `int` | *required* | MCMC iterations |
| `priors` | `dict[str, Distribution]` | *required* | Prior distributions |
| `theta_sampler` | `Callable \| None` | `None` | Custom $\theta$-step (default: model-specific Gibbs if available) |
| `burnin` | `int` | `0` | Burn-in iterations |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

### Methods

##### `sample()`

```python
def sample(
    self,
    observations: NDArray[np.float64],
) -> MCMCChain
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observations` | `NDArray[np.float64]` | *required* | Observations |

**Returns**: `MCMCChain` — Joint samples of $(\theta, x_{1:T})$.

### Example

```python
import particlefilterbox as pfb

model = pfb.models.LinearGaussian(dim=2)

pg = pfb.ParticleGibbs(
    model=model,
    n_particles=100,
    n_iterations=5000,
    priors=priors,
    burnin=1000,
)

chain = pg.sample(observations)
print(chain.summary())

# Trajectory samples available via chain.trajectories
traj_samples = chain.trajectories   # shape (n_iterations, T, d_x)
```

!!! warning
    Plain Particle Gibbs suffers from **path degeneracy** — early states in $x_{1:T}$ barely change between iterations. For models with $T > 100$, use **PGAS** instead.

---

## PGAS

Particle Gibbs with Ancestor Sampling (Lindsten, Jordan & Schön, 2014). Extends Particle Gibbs by adding an **ancestor sampling** step at each time, which dramatically improves mixing on early states and breaks the path-degeneracy problem.

Each time step, a new ancestor for the reference trajectory is drawn with probability:

$$
\mathbb{P}(a_t = i) \propto W_t^{(i)} \cdot f(x_{t+1}^{\text{ref}} \mid x_t^{(i)})
$$

### Constructor

```python
PGAS(
    model: ParticleFilterModel,
    n_particles: int,
    n_iterations: int,
    priors: dict[str, Distribution],
    theta_sampler: Callable | None = None,
    burnin: int = 0,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model (must implement `log_transition()` for ancestor sampling) |
| `n_particles` | `int` | *required* | Number of particles in CSMC-AS |
| `n_iterations` | `int` | *required* | MCMC iterations |
| `priors` | `dict[str, Distribution]` | *required* | Prior distributions |
| `theta_sampler` | `Callable \| None` | `None` | Custom $\theta$-step |
| `burnin` | `int` | `0` | Burn-in iterations |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

### Methods

##### `sample()`

```python
def sample(
    self,
    observations: NDArray[np.float64],
) -> MCMCChain
```

**Returns**: `MCMCChain` — Joint samples with improved mixing over PG.

### Example

```python
import particlefilterbox as pfb

model = pfb.models.StochasticVolatility(variant='basic')

pgas = pfb.PGAS(
    model=model,
    n_particles=50,
    n_iterations=10_000,
    priors=priors,
    burnin=2000,
)

chain = pgas.sample(observations)
print(chain.summary())
```

!!! tip
    PGAS typically requires **far fewer particles** than plain Particle Gibbs (50–100 is often sufficient vs. 1000+) because ancestor sampling compensates for trajectory degeneracy.

---

## ConditionalSMC

Conditional Sequential Monte Carlo — the inner kernel used by Particle Gibbs and PGAS. Runs a standard particle filter but **forces** one particle at each time to equal a given reference trajectory. Used as a building block; rarely called directly.

### Constructor

```python
ConditionalSMC(
    model: ParticleFilterModel,
    n_particles: int,
    reference_trajectory: NDArray[np.float64],
    ancestor_sampling: bool = False,
    rng: np.random.Generator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel` | *required* | State-space model |
| `n_particles` | `int` | *required* | Number of particles (including reference) |
| `reference_trajectory` | `NDArray[np.float64]` | *required* | Conditioning trajectory, shape `(T, d_x)` |
| `ancestor_sampling` | `bool` | `False` | Enable ancestor sampling (CSMC-AS, used by PGAS) |
| `rng` | `np.random.Generator \| None` | `None` | Random number generator |

### Methods

##### `filter()`

```python
def filter(
    self,
    observations: NDArray[np.float64],
) -> FilterResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observations` | `NDArray[np.float64]` | *required* | Observations |

**Returns**: `FilterResult` — Particle cloud history with reference trajectory guaranteed to appear at index 0 at each time.

### Example

```python
import particlefilterbox as pfb

model = pfb.models.LinearGaussian(dim=2)
x_ref = previous_iteration_trajectory  # shape (T, 2)

csmc = pfb.ConditionalSMC(
    model=model,
    n_particles=100,
    reference_trajectory=x_ref,
    ancestor_sampling=True,
)

result = csmc.filter(observations)
# Sample a new trajectory from the conditional posterior
new_trajectory = result.sample_trajectory(rng)
```

---

## MCMCChain

Result container for PMCMC methods. Provides post-processing utilities, summary statistics, and conversion to `pandas.DataFrame` for analysis with standard MCMC diagnostic packages (e.g., ArviZ).

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `samples` | `dict[str, NDArray[np.float64]]` | Parameter samples per name, shape `(n_iterations, ...)` |
| `log_posteriors` | `NDArray[np.float64]` | Log-posterior values per iteration, shape `(n_iterations,)` |
| `acceptance_rates` | `dict[str, float]` | Per-parameter (or overall) acceptance rates |
| `ess` | `dict[str, float]` | Effective sample size per parameter |
| `trajectories` | `NDArray[np.float64] \| None` | State trajectories for PG/PGAS, shape `(n_iterations, T, d_x)` or `None` |
| `n_iterations` | `int` | Total iterations after burn-in |

### Methods

##### `thin()`

Return a thinned chain keeping every $k$-th sample.

```python
def thin(self, k: int) -> MCMCChain
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | *required* | Thinning interval (keep every $k$-th sample) |

**Returns**: `MCMCChain` — New chain with thinned samples.

---

##### `burn()`

Discard the first $n$ iterations (additional burn-in on top of sampler burn-in).

```python
def burn(self, n: int) -> MCMCChain
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | *required* | Number of initial iterations to discard |

**Returns**: `MCMCChain` — Chain with first $n$ samples removed.

---

##### `summary()`

Generate a summary table with posterior mean, std, quantiles, ESS, and R-hat (if multi-chain).

```python
def summary(
    self,
    quantiles: tuple[float, ...] = (0.025, 0.5, 0.975),
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantiles` | `tuple[float, ...]` | `(0.025, 0.5, 0.975)` | Quantiles to report |

**Returns**: `pd.DataFrame` — Parameter-by-parameter summary.

---

##### `to_dataframe()`

Convert parameter samples to long-format pandas DataFrame (compatible with ArviZ / seaborn).

```python
def to_dataframe(self) -> pd.DataFrame
```

**Returns**: `pd.DataFrame` with columns `iteration`, `parameter`, `value`.

### Example

```python
chain = pmmh.sample(observations)

# Post-processing
thinned = chain.burn(1000).thin(5)

# Summary table
print(thinned.summary(quantiles=(0.05, 0.5, 0.95)))

# Export for ArviZ
df = thinned.to_dataframe()

# Per-parameter diagnostics
for name, ess in thinned.ess.items():
    print(f"{name}: ESS = {ess:.0f}")
```

---

## See Also

- [User Guide: PMCMC](../user-guide/pmcmc/index.md) — PMCMC methods overview
- [User Guide: PMMH](../user-guide/pmcmc/pmmh.md) — PMMH walkthrough
- [User Guide: Particle Gibbs](../user-guide/pmcmc/particle-gibbs.md) — PG and PGAS walkthrough
- [Tutorials: PMMH for SV](../tutorials/pmmh-sv.md) — Full PMMH tutorial
- [Tutorials: PGAS](../tutorials/pgas.md) — PGAS with ancestor sampling
- [Theory: PMCMC](../theory/pmcmc-theory.md) — Mathematical foundations
- [SMC API](smc.md) — SMC² as alternative for joint $(\theta, x)$ inference
- [Diagnostics API](diagnostics.md) — MCMC convergence diagnostics (`MCMCConvergence`, `MixingDiagnostic`)
- [Visualization API](visualization.md) — Trace plots and posterior plots
