# Complete Workflow Examples

End-to-end notebooks that integrate model specification, filtering,
parameter estimation, diagnostics, forecasting, and reporting into a
single coherent pipeline. These examples represent the workflow a
researcher would follow in practice when using `particlefilterbox`.

## Notebooks

1. `notebooks/01_sv_full_workflow.ipynb` - Full stochastic volatility
   analysis pipeline applied to SP500 returns.
2. `notebooks/02_parameter_estimation_workflow.ipynb` - Parameter
   estimation comparison across SMC sampler, PMMH, and PGAS on data
   with known ground-truth parameters.

## Purpose

These notebooks demonstrate the complete workflow a researcher would
follow when using `particlefilterbox` for practical analysis. They
are intentionally longer than the topic-focused examples in
`examples/stochastic_volatility/`, `examples/pmcmc/`, and
`examples/smc/` - they chain those pieces into a single pipeline so a
reader can see how the components fit together from raw data to final
report.

## Pipelines

### Workflow 1 - SV Full Workflow (SP500)

1. Load data (`data/sp500_returns.csv`).
2. Exploratory analysis (returns plot, ACF of returns and squared
   returns, distributional summaries).
3. Specify the stochastic volatility model.
4. Filter the latent log-volatility with the Bootstrap particle
   filter.
5. Estimate static parameters via PMMH (or PGAS).
6. Convergence and efficiency diagnostics (ESS, trace plots,
   acceptance rate, Gelman-Rubin where applicable).
7. Out-of-sample volatility forecasting.
8. Final report with summary tables and figures.

### Workflow 2 - Parameter Estimation Workflow (simulated SV)

1. Load data (`data/simulated_sv.csv`) where the true parameters are
   known.
2. Estimate via SMC sampler (tempering).
3. Estimate via PMMH.
4. Estimate via PGAS.
5. Compare posterior distributions and recovery of the true
   parameters.
6. Model comparison: plain SV vs SV with leverage (SV-L).
7. Convergence diagnostics across methods.
8. Estimation report.

## Datasets

Both datasets live under `data/` as symlinks to the canonical copies
maintained in the topic-focused examples, so updates propagate
automatically.

- `data/sp500_returns.csv` - SP500-calibrated daily returns
  (symlink to `../stochastic_volatility/data/sp500_returns.csv`).
- `data/simulated_sv.csv` - Simulated stochastic volatility series
  with known parameters
  (symlink to `../bootstrap_sir/data/simulated_sv.csv`).

## Validation

- `R_validation/` - Reproductions of the pipeline in R using
  `stochvol` and `pomp` for cross-validation of filtering and
  parameter estimation.
- `stata_validation/` - Stata reproductions using `sspace` where the
  model admits a linear-Gaussian state-space cast.

## Directory Layout

```
complete_workflow/
├── README.md
├── data/
│   ├── sp500_returns.csv      (symlink)
│   └── simulated_sv.csv       (symlink)
├── notebooks/                 (populated in F9.2+)
├── solutions/                 (reference outputs and saved traces)
├── R_validation/              (R scripts: stochvol, pomp)
└── stata_validation/          (Stata .do files: sspace)
```
