---
title: "Changelog"
description: "particlefilterbox version history — all releases with key changes, migration notes, and breaking changes."
---

# Changelog

All notable changes to **particlefilterbox** are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Sections**: Added, Changed, Deprecated, Removed, Fixed, Security, Performance.

Links to published releases are available on the [GitHub Releases page](https://github.com/nodesecon/particlefilterbox/releases).

---

## [Unreleased]

### Added

- Placeholder for work in progress. New entries should be added here under the appropriate sub-section and graduated to a dated release block on tag.

### Changed

- _No entries yet._

### Fixed

- _No entries yet._

---

## [0.1.0] — 2026-04-22

### Summary

**Initial Release — Core Particle Filtering and SMC Framework**

The first public release of `particlefilterbox` establishes the foundation for
particle filtering, smoothing, Sequential Monte Carlo, and Particle MCMC in
Python. The release is validated against closed-form Kalman solutions on
linear-Gaussian benchmarks and against `kalmanbox` for the Rao-Blackwellized
and Unscented particle filters.

### Added

#### Core framework

- `ParticleCloud` — container for particles and log-weights with ESS tracking,
  log-sum-exp normalization, and defensive copies.
- `FilterResult`, `SmootherResult`, `PMCMCResult`, `SMCResult` — structured
  result containers with `.summary()`, serialization, and metadata.
- `StateSpaceModel` — abstract base class exposing `transition`,
  `observation`, `initial_state`, and optional `log_density` hooks.
- Consistent seeding API: every stochastic routine accepts `seed: int | None`
  or a `numpy.random.Generator`.

#### Resampling schemes

- `multinomial`, `systematic`, `stratified`, `residual`, `residual_systematic`.
- Vectorized implementations with Numba acceleration for `systematic` and
  `stratified` (opt-in via the `accel` extra).

#### Particle filters (`particlefilterbox.filters`)

- `BootstrapParticleFilter` — Gordon, Salmond & Smith (1993).
- `SIRFilter` — Sequential Importance Resampling with adaptive ESS trigger.
- `AuxiliaryParticleFilter` — Pitt & Shephard (1999) with lookahead weights.
- `RaoBlackwellizedParticleFilter` — marginalized linear-Gaussian substructure
  via `kalmanbox`.
- `UnscentedParticleFilter` — van der Merwe et al. (2000); UKF proposal.
- `RegularizedParticleFilter` — kernel smoothing after resampling.
- `EnsembleParticleFilter` — ensemble-square-root proposal.
- `GuidedParticleFilter` — user-supplied proposal density.
- `LocallyOptimalParticleFilter` — locally optimal proposal for conditionally
  Gaussian observation models.

#### Smoothers (`particlefilterbox.smoothers`)

- `FFBSmoother` — Forward Filtering Backward Smoothing, Kitagawa (1996).
- `FFBSimulator` — Forward Filtering Backward Simulation, Godsill et al. (2004).
- `TwoFilterSmoother` — Briers, Doucet & Maskell (2010).
- `FixedLagSmoother` — truncated-lookback smoother for online settings.

#### SMC samplers (`particlefilterbox.smc`)

- `SMCSampler` — Del Moral, Doucet & Jasra (2006) tempered sequence.
- `SMCSquared` — Chopin, Jacob & Papaspiliopoulos (2013) for static
  parameter inference alongside a filtering problem.
- `IBIS` — Iterated Batch Importance Sampling, Chopin (2002).
- `WasteFreeSMC` — Dau & Chopin (2022) waste-free variant.
- `TemperingScheduler` — adaptive temperature selection based on conditional
  ESS.
- `MCMCMove` — Metropolis, Random-Walk, Gaussian, and Independence proposals
  for rejuvenation.

#### PMCMC (`particlefilterbox.pmcmc`)

- `PMMH` — Particle Marginal Metropolis-Hastings, Andrieu, Doucet & Holenstein
  (2010).
- `ParticleGibbs` — Particle Gibbs with conditional SMC update.
- `PGAS` — Particle Gibbs with Ancestor Sampling, Lindsten et al. (2014).
- `ConditionalSMC` — building block for PG and PGAS.
- `SMC2Online` — online variant with particle rejuvenation.
- Diagnostics: acceptance rates, IACT, effective sample size of the PMCMC
  chain.

#### Pre-built state-space models (`particlefilterbox.models`)

- `StochasticVolatility` — log-variance AR(1) with Gaussian observations.
- `JumpDiffusion` — Merton-style jumps with Gaussian diffusion.
- `NonlinearRegimeModel` — regime-switching nonlinear transitions.
- `CountStateSpace` — Poisson / Negative Binomial observations on a latent
  Gaussian state.
- `BoundedStates` — logistic / logit transformations for bounded support.
- `MixtureModel` — finite mixture of Gaussian sub-models.
- `ContinuousTime` — Euler-Maruyama discretization of SDEs.
- `DSGE` — log-linearized DSGE wrapper built on `kalmanbox` for the
  measurement equation.

#### Diagnostics (`particlefilterbox.diagnostics`)

- `ESSDiagnostic` — effective sample size time series and summary.
- `WeightDiagnostic` — weight distribution, coefficient of variation, top-k
  mass.
- `ConvergenceDiagnostic` — Monte Carlo error across particle counts.
- `DegeneracyDiagnostic` — unique-ancestor counts, genealogy depth.
- `FilterComparison` — side-by-side metrics across multiple filters.
- `KalmanValidation` — compare against the Kalman benchmark when available.
- `MCMCConvergence`, `AcceptanceRate`, `Mixing` — PMCMC convergence
  diagnostics.
- `PredictiveChecks` — one-step-ahead residual and PIT checks.
- `MarginalLikelihood` — log-evidence estimates with standard errors.

#### Acceleration (`particlefilterbox.acceleration`)

- Optional Numba JIT for resampling and weight normalization.
- CuPy/JAX GPU backends for large-particle workflows (experimental).
- Thread-parallel batching across independent filter runs.
- Adaptive-N strategies that grow or shrink `N` based on ESS.

#### Visualization (`particlefilterbox.visualization`)

- Particle plots, weight histograms, filtered/smoothed state trajectories.
- PMCMC trace plots and posterior summaries.
- Convergence plots across particle counts.
- Three themes: `"default"`, `"paper"`, `"dark"`.

#### Reporting and experiment framework

- `Report` — HTML and Markdown reports bundling filters, diagnostics, and
  plots.
- `Experiment` — one-liner workflow running a filter, attaching diagnostics,
  and producing a report.

#### Datasets (`particlefilterbox.datasets`)

- `load_linear_gaussian` — simulated benchmark with analytical Kalman
  reference.
- `load_sv` — S&P 500 daily returns for stochastic volatility demos.
- `load_nile` — Nile river annual flow (local-level benchmark).
- `load_lorenz63` — chaotic dynamics benchmark.
- `load_dsge_example` — small-scale DSGE illustration.

#### CLI (`particlefilterbox.cli`)

- `pfbox filter <config.yaml>` — run a filter from a configuration file.
- `pfbox smc <config.yaml>` — run an SMC sampler.
- `pfbox pmcmc <config.yaml>` — run a PMCMC chain.
- `pfbox report <result.npz>` — generate an HTML report from saved output.

#### Documentation

- MkDocs Material site with Getting Started, User Guide, Theory, Diagnostics,
  Acceleration, Tutorials, Visualization, API Reference, FAQ, Benchmarks, and
  Contributing sections.
- MathJax for mathematical derivations.
- `mkdocstrings` for auto-generated API reference.

### Performance

| Benchmark (T=500, N=10 000) | Time (Python) | Time (Numba) | Speedup |
|---|---|---|---|
| Bootstrap PF — Linear Gaussian | 1.8 s | 0.21 s | ~8.5× |
| Systematic resampling | 310 µs | 38 µs | ~8.2× |
| SV Bootstrap PF | 2.4 s | 0.29 s | ~8.3× |
| PMMH (10 000 iters, T=200, N=500) | 42 s | 6.1 s | ~6.9× |

!!! note "Benchmark environment"
    Intel i7-12700K, 32 GB RAM, Python 3.11, NumPy 1.26, Numba 0.59.
    See the [Benchmarks](../benchmarks/index.md) section for the full methodology.

### Validation

Validated against `kalmanbox` on linear-Gaussian special cases:

| Filter | Metric | Tolerance | Result |
|---|---|---|---|
| Bootstrap PF (N = 100 000) | Filtered mean vs. Kalman | ±1e-2 | max diff = 6.8e-3 |
| SIR Filter (N = 100 000) | Filtered mean vs. Kalman | ±1e-2 | max diff = 7.1e-3 |
| Rao-Blackwellized PF | Filtered mean vs. Kalman | ±1e-6 | max diff = 4.2e-7 |
| Unscented PF | Filtered mean vs. Kalman | ±1e-3 | max diff = 8.9e-4 |
| FFBSi smoother | Smoothed mean vs. RTS | ±1e-2 | max diff = 9.3e-3 |

Cross-library comparisons against `particles` (Chopin) and `pyfilter` on the
stochastic volatility benchmark agree to within Monte Carlo error.

### Known Limitations

- GPU backends (`cupy`, `jax`) are experimental; expect API changes in 0.2.
- `WasteFreeSMC` is a minimal implementation — the adaptive variant is
  planned for 0.8.
- `DSGE` currently requires a linearized model; full nonlinear DSGE filtering
  is a 0.5 target.

---

## Versioning Policy

`particlefilterbox` uses [Semantic Versioning](https://semver.org/):

| Component | When incremented |
|---|---|
| **Major** (X.0.0) | Incompatible public-API changes |
| **Minor** (0.X.0) | New features, backward compatible |
| **Patch** (0.0.X) | Bug fixes, backward compatible |

During the `0.y.z` development series, minor versions may include breaking
changes when justified; every such change will be called out in this
changelog under **Changed** or **Removed**, with a migration note.

---

## Migration Notes

### Pre-0.1.0 → 0.1.0

This is the first public release — no migration needed.

---

## See Also

- [Contributing Guide](contributing.md) — How to contribute
- [Roadmap](roadmap.md) — Planned features
- [GitHub Releases](https://github.com/nodesecon/particlefilterbox/releases) — Binary artifacts and release notes
- [API Reference](../api/index.md) — Full API documentation
