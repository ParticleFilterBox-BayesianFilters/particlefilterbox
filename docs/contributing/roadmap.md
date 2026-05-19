---
title: "Roadmap"
description: "particlefilterbox development roadmap — planned features, priorities, and release schedule through v1.0 and beyond."
---

# Roadmap

`particlefilterbox` aims to be the most comprehensive Python library for particle filtering, Sequential Monte Carlo, and Particle MCMC on nonlinear state-space models. This document lays out the planned trajectory from the current `v0.1` release through a production-ready `v1.0` and sketches the long-term research agenda beyond it.

!!! info "Current status — v0.1.0"
    The initial release (April 2026) ships the core filters (Bootstrap, SIR, Auxiliary), the main smoothers, the base resampling schemes, and the experiment/reporting framework. All estimators have a validated Kalman benchmark via [`kalmanbox`](https://github.com/nodesecon/kalmanbox).

---

## Release Matrix

| Version | Target | Theme | Status |
|---|---|---|---|
| **v0.1** | 2026-Q2 | Core filters, smoothers, basic SMC | ✅ Released |
| **v0.2** | 2026-Q3 | Rao-Blackwellization, UPF, kalmanbox integration | 🚧 In progress |
| **v0.3** | 2026-Q4 | SMC methods (SMCSampler, SMC², IBIS) | 📋 Planned |
| **v0.4** | 2027-Q1 | PMCMC (PMMH, PG, PGAS) | 📋 Planned |
| **v0.5** | 2027-Q2 | Pre-built models (SV, DSGE, jump-diffusion, …) | 📋 Planned |
| **v0.6** | 2027-Q3 | Acceleration (Numba, GPU) | 📋 Planned |
| **v0.7** | 2027-Q4 | Diagnostics and visualization completeness | 📋 Planned |
| **v0.8** | 2028-Q1 | Waste-Free SMC, advanced samplers | 📋 Planned |
| **v1.0** | 2028-Q2 | Production-ready, API stable | 🎯 Target |

---

## v0.1 — Core Filters (Released)

**Released April 2026.** Establishes the foundation.

- Core containers: `ParticleCloud`, `FilterResult`, `StateSpaceModel`.
- Resampling: multinomial, systematic, stratified, residual, residual-systematic.
- Filters: Bootstrap, SIR, Auxiliary.
- Smoothers: FFBSm, FFBSi, Two-Filter, Fixed-Lag.
- Experiment framework and HTML/Markdown reports.
- CLI (`pfbox`) for config-driven runs.
- Documentation: Getting Started, User Guide, API Reference, Theory (core chapters).

---

## v0.2 — Rao-Blackwellization and UPF

**Target: 2026-Q3.**

Deepens integration with `kalmanbox` and expands the filter catalog with
algorithms that exploit conditional linear-Gaussian structure.

### Planned features

- **Full Rao-Blackwellized PF** — marginalize the conditionally linear
  sub-state via Kalman filtering, sampling only the nonlinear component.
- **Unscented Particle Filter** — van der Merwe et al. (2000) with the UKF
  proposal; unscented Rauch–Tung–Striebel smoother for RB-smoothing.
- **Regularized PF** — Musso, Oudjane & Le Gland (2001) kernel regularization.
- **Ensemble PF** — ensemble Kalman proposal for high-dimensional states.
- **Guided / locally optimal proposals** — closed-form proposals for
  Gaussian-observation models and conditionally Gaussian states.
- **kalmanbox linearization helpers** — automatic linearization of differentiable
  transitions via `jax.jacfwd` when JAX is available.
- **Filter comparison harness** — `FilterComparison` runs N filters on the
  same data and emits side-by-side metrics.

### Breaking-change notes (planned)

- `ParticleFilter.step()` signature will accept an optional
  `auxiliary_info: dict | None` argument for lookahead state — additive, no
  existing call sites break.

---

## v0.3 — SMC Methods

**Target: 2026-Q4.**

Full SMC sampler machinery for static Bayesian inference and for problems
where both states and parameters need to be sampled from a sequence of
distributions.

### Planned features

- **`SMCSampler`** — Del Moral, Doucet & Jasra (2006) with geometric,
  data-tempering, and adaptive tempering schedules.
- **`SMC²`** — Chopin, Jacob & Papaspiliopoulos (2013) for joint
  state-parameter inference.
- **`IBIS`** — Chopin (2002) iterated batch importance sampling with
  MCMC rejuvenation.
- **Adaptive tempering** — bisection on the conditional ESS criterion.
- **Rejuvenation moves** — RW Metropolis, MALA, preconditioned
  Crank–Nicolson, HMC (via an optional backend).
- **Evidence / marginal likelihood** — principled log-evidence estimator with
  jackknife standard errors.

---

## v0.4 — PMCMC

**Target: 2027-Q1.**

Full particle MCMC family for static parameter inference on state-space
models.

### Planned features

- **`PMMH`** — Particle Marginal Metropolis–Hastings, Andrieu, Doucet &
  Holenstein (2010).
- **Particle Gibbs (`ParticleGibbs`)** — conditional SMC update of the
  trajectory given parameters.
- **PG-AS** — Particle Gibbs with Ancestor Sampling, Lindsten et al. (2014).
- **PG-BS** — Particle Gibbs with Backward Simulation.
- **Adaptive PMCMC** — Roberts–Rosenthal (2009) adaptive proposal covariance.
- **Online SMC²** for streaming parameter posteriors.
- **Chain-level diagnostics** — Gelman–Rubin on multiple PMCMC chains,
  integrated autocorrelation time, effective sample size of the chain.
- **Tuning guide** — recipes for choosing `N` (particles) vs. MCMC iterations
  based on the Pitt et al. (2012) / Doucet et al. (2015) variance criteria.

---

## v0.5 — Pre-built Models

**Target: 2027-Q2.**

A curated catalog of ready-to-use state-space models frequently encountered
in finance, macroeconomics, epidemiology, and engineering.

### Planned features

- **Stochastic Volatility family**
    - Log-variance AR(1) with Gaussian innovations
    - SV with leverage (Omori, Chib, Shephard, Nakajima 2007)
    - SV with jumps in returns and volatility
    - Multifactor SV
- **DSGE** — fully nonlinear filtering via particle methods; log-linearized
  fallback delegating to `kalmanbox`.
- **Jump-Diffusion** — Merton, Kou, and self-exciting Hawkes intensity
  variants.
- **Regime-switching** — Hamilton (1989), Markov-switching AR, MS-GARCH.
- **Count state-space** — Poisson, Negative Binomial, ZIP observations.
- **Bounded state-space** — logit / probit / logistic transforms.
- **Mixture models** — finite mixtures of Gaussians on states.
- **Continuous-time SDEs** — Euler–Maruyama, Milstein, and stochastic Runge–Kutta
  discretizations.
- **Epidemiological models** — SIR, SEIR, SEIRD with time-varying `R_t`.

---

## v0.6 — Acceleration

**Target: 2027-Q3.**

First-class performance primitives.

### Planned features

- **Numba kernels** — JIT-compiled resampling, weight normalization, and
  proposal sampling for common model families.
- **GPU backends** — CuPy and JAX implementations of the core SMC loop; batch
  across particles on the GPU.
- **Parallelization** — thread-level parallelism across independent filter
  runs (model comparison, cross-validation).
- **Adaptive N** — automatic particle-count adjustment based on ESS.
- **Benchmarks** — continuous performance regression tests tracked in CI.
- **Memory efficiency** — streaming genealogy storage for long filter runs.

!!! note "Target"
    A 10× end-to-end speedup over pure-NumPy for the SV-2000 benchmark
    (`T = 2000`, `N = 10 000`) is the gating criterion.

---

## v0.7 — Diagnostics and Visualization

**Target: 2027-Q4.**

Everything a practitioner needs to trust a particle-filter run.

### Planned features

- **Full diagnostics suite**
    - ESS time series, weight-CV, top-k mass, unique-ancestor counts
    - Kalman validation for any LG special case
    - One-step-ahead residual PIT, Rosenblatt-transformed diagnostics
    - Marginal likelihood standard errors (bootstrap and jackknife)
    - PMCMC: IACT, Gelman–Rubin, Geweke, Heidelberger–Welch
- **Visualization overhaul**
    - Particle fan charts with credible ribbons
    - Genealogy plots (Lancelot, Kantas et al. 2015 style)
    - PMCMC trace and corner plots
    - Model-specific plots (SV volatility paths, regime probabilities)
    - Three themes: `default`, `paper`, `dark`
- **Reporting**
    - HTML reports with interactive Plotly figures
    - LaTeX exporter for publication-ready tables
    - Snapshot testing for visual regression

---

## v0.8 — Waste-Free SMC and Advanced Features

**Target: 2028-Q1.**

### Planned features

- **Waste-Free SMC** — Dau & Chopin (2022) full implementation with
  multi-step rejuvenation.
- **Nested SMC** — Naesseth, Lindsten & Schön (2015).
- **Divide-and-conquer SMC** — Lindsten et al. (2017) for tree-structured
  models.
- **Island particle filter** — Vergé et al. (2015) for distributed runs.
- **Adaptive proposals**
    - Empirical-moment Gaussian proposals
    - Normalizing-flow proposals (opt-in via `nflows` or `jax.numpyro`)
- **Twisted particle filters** — Guarniero, Johansen & Lee (2017).
- **Online learning** — particle learning (Lopes, Carvalho, Johannes,
  Polson 2011) for sequential parameter estimation.

---

## v1.0 — Production Release

**Target: 2028-Q2.**

### Gating criteria

- Public API frozen and documented — any breaking change requires a v2.
- Reference implementations for every major textbook example (Chopin &
  Papaspiliopoulos 2020; Doucet, de Freitas & Gordon 2001; Särkkä &
  Svensson 2023).
- Benchmarks against `particles`, `pyfilter`, `bayesloop`, and R's `smcUDF`
  published and reproducible from `benchmarks/`.
- Continuous-integration matrix covers Python 3.11, 3.12, 3.13 on Linux,
  macOS, and Windows.
- 95 %+ line coverage; 100 % of the public API has docstring examples.
- Documentation translated to at least one additional language (Portuguese).

### v1.0 manifesto

- **Correct** — every filter validates against the Kalman closed form when
  reducible.
- **Fast** — Numba or GPU parity with specialized implementations.
- **Pedagogical** — theory pages rigorous enough to cite; examples runnable
  in a notebook.
- **Composable** — clear separation of model, proposal, resampling, and
  diagnostics.

---

## Long-Term Vision (Post-1.0)

Exploratory research directions that will graduate into minor releases as they mature.

### Differentiable particle filtering

- **Differentiable resampling** — Corenflos, Thornton, Deligiannidis &
  Doucet (2021) OT-based resampling for gradient-based learning.
- **End-to-end learning** — train the transition and observation networks
  jointly with the filter via reparameterized SMC.
- **Score-matching proposals** — learn proposal distributions without
  closed-form densities.

### Normalizing-flow integration

- **Flow-based proposals** — replace Gaussian proposals with invertible
  flows for heavy-tailed posteriors.
- **Amortized inference** — train a flow to emulate the filter across a
  family of observations.

### Neural-network-augmented filters

- **Learned proposals** — amortized proposal networks (Gu, Ghahramani &
  Turner 2015).
- **Particle filter recurrent neural networks** — Karkus, Hsu & Lee
  (2018) for sequence modeling.
- **Neural state-space models** — Krishnan, Shalit & Sontag (2017).

### Online learning

- **Particle learning** for streaming parameter updates without re-running
  the filter from scratch.
- **Adaptive resampling schedules** — learn when to resample from data.
- **Concept-drift detection** on long filter runs.

### Distributed computing

- **Dask integration** — out-of-core particle storage for very long
  series.
- **Ray support** — distributed PMCMC chains and multi-model experiments.
- **Cloud-native workflows** — AWS / GCP / Azure turnkey deployments.

### Cross-platform tools

- **R interoperability** — export models to/from `particles` / `nimble`.
- **Stan / PyMC bridges** — share priors and model definitions.
- **Dashboard** — Streamlit/Panel app for interactive SMC exploration.

---

## Documentation Roadmap

### Completed

- [x] Phase 1 — MkDocs Material infrastructure, site scaffolding.
- [x] Phase 2 — Getting Started and User Guide core pages.
- [x] Phase 3 — Theory pages for SMC, particle filters, smoothing.
- [x] Phase 4 — Diagnostics, Acceleration, Tutorials.
- [x] Phase 5 — API Reference, FAQ, Benchmarks, Contributing.

### In Progress

- [ ] Phase 6 — Expansion: community-contributed tutorials and case studies.

### Planned

- [ ] Maintenance: regular updates as features ship.
- [ ] Translations: Portuguese, Spanish, Chinese.
- [ ] Versioned documentation via `mike` (one site per minor release).

---

## How to Influence the Roadmap

### Feature Requests

Open a [GitHub Issue](https://github.com/nodesecon/particlefilterbox/issues) with the `[Feature]` label. Include:

1. **Use case** — what problem does it solve?
2. **Description** — what should the feature do?
3. **References** — academic papers or existing implementations.
4. **Priority justification** — why is this important for SMC users?

### Community Voting

React with a thumbs-up on existing feature-request issues to signal demand. Issues with more community interest are prioritized higher.

### Contributions

The fastest way to get a feature is to implement it yourself! See the [Contributing Guide](contributing.md) for templates and the PR process. Maintainers mentor first-time contributors on methodological PRs.

### Sponsorship

For organizations that need specific features on a timeline, we welcome sponsorship discussions. Contact the team via [GitHub Discussions](https://github.com/nodesecon/particlefilterbox/discussions).

---

## Release Schedule and Support Policy

### Cadence

| Version | Cadence | Content |
|---|---|---|
| **Major** (X.0.0) | As needed | Breaking API changes |
| **Minor** (0.X.0) | ~1 per quarter | New features, backward compatible |
| **Patch** (0.0.X) | As needed | Bug fixes, documentation updates |

### Release Process

1. Feature freeze one week before release.
2. Release candidate published for community testing.
3. Final release after validation.
4. Changelog and migration notes updated before the tag.

### Support Policy

- **Current minor version**: full support (bug fixes, security patches, new features).
- **Previous minor version**: security patches for 6 months.
- **Older versions**: community support only.

Once **v1.0** ships, the previous **major** version receives security patches for 12 months.

---

## See Also

- [Contributing Guide](contributing.md) — How to contribute code and documentation
- [Changelog](changelog.md) — Version history
- [Code of Conduct](code-of-conduct.md) — Community standards
- [API Reference](../api/index.md) — Full API documentation
