# Changelog

## v0.1.1 (2026-06-10)

Bug-fix release. The first release shipped a non-functional CLI and a
broken README example; this release makes them work and fixes failing tests.

### Fixed

- **CLI**: every `pfbox` command (`filter`, `estimate`, `compare`,
  `simulate`) referenced modules and classes that did not exist
  (`models.sv`/`SVModel`, `models.local_level`, `models.linear_gaussian`,
  `filters.apf`, `BootstrapFilter`) and swallowed errors while exiting 0.
  Commands are now wired to the real API (`StochasticVolatility`,
  `BootstrapPF`/`AuxiliaryPF`) and exit non-zero on failure.
- **`pfbox estimate`** now performs real PMMH parameter estimation for the
  stochastic volatility model (posterior means/std and acceptance rate).
- **README** Quick Start used the non-callable `systematic` module instead of
  `systematic_resample`; corrected so the example runs.
- **`LocallyOptimalPF`**: fixed a shape error in the predictive
  log-likelihood that crashed filtering with vectorized models.
- **Tests**: corrected an incorrect `SIR is BootstrapPF` identity assertion
  and replaced no-op CLI test assertions (`exit_code in (0, 1)`) with real
  behavioural checks.
- **Stochastic volatility (leverage)**: documented that the prior transition
  used for filtering does not encode the `rho` leverage correlation (it is a
  contemporaneous innovation coupling handled in `simulate`).
- Removed nonexistent filters/models (EKF/UKF wording, Local Level, Linear
  Gaussian) from the changelog and cleaned up `ruff` lint in the package.

## v0.1.0 (2026-03-17)

Initial release.

### Features

- Particle filters: Bootstrap (SIR), Auxiliary, Guided, Locally Optimal,
  Regularized, Ensemble, Rao-Blackwellized, and Unscented (the latter two
  require the optional `kalmanbox` dependency)
- Resampling: multinomial, systematic, stratified, residual, adaptive,
  killing, and optimal transport
- Smoothers: FFBSm, FFBSi, Two-Filter, and Fixed-Lag
- SMC samplers: SMC Sampler, Tempering, SMC^2, IBIS, Waste-Free SMC
- PMCMC: PMMH, Particle Gibbs, PGAS, Conditional SMC, online SMC^2
- State-space models: Stochastic Volatility, Jump Diffusion, Continuous Time,
  Count State Space, Bounded States, Nonlinear Regime, Mixture, DSGE
- Diagnostics: ESS monitor, weight analysis, convergence study,
  degeneracy detection, model comparison, PMCMC diagnostics
- Visualization with 4 themes (nodesecon, minimal, paper, dark)
- HTML/LaTeX/Markdown report generation
- CLI tool: pfbox filter/estimate/compare/simulate
- Bundled simulated datasets (finance, macro, epidemic)
- PFExperiment reproducible experiment framework
