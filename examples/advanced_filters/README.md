# Advanced Particle Filters Examples

## Notebooks

1. `01_auxiliary_pf.ipynb` - Auxiliary Particle Filter (Pitt & Shephard, 1999)
2. `02_rbpf.ipynb` - Rao-Blackwellized Particle Filter
3. `03_unscented_regularized.ipynb` - Unscented PF and Regularized PF

## Datasets

Uses shared datasets from bootstrap_sir/data/:
- `simulated_sv.csv` - Stochastic volatility model
- `simulated_linear_gaussian.csv` - Linear-Gaussian SSM

## Validation

- `R_validation/` - R scripts using pomp and nimbleSMC
- `stata_validation/` - Stata reference (limited, sspace only)
