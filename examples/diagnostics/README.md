# Particle Filter Diagnostics Examples

This directory contains examples and notebooks demonstrating diagnostic tools
for assessing the quality of particle filter approximations. Proper diagnostics
are essential for identifying particle degeneracy, insufficient sample sizes,
and for comparing competing state-space models.

## Notebooks

1. `01_ess_weight_diagnostics.ipynb` - ESS monitoring and weight analysis
2. `02_convergence_model_comparison.ipynb` - Convergence diagnostics and model comparison

## Datasets

- `data/simulated_sv.csv` - Stochastic volatility model (shared from FASE 1 via symlink)

## Key Diagnostics

### Effective Sample Size (ESS)

The ESS measures the number of "effective" particles after importance sampling.
Low ESS indicates weight degeneracy (a few particles dominate the posterior
approximation).

$$
\mathrm{ESS}_t \;=\; \frac{1}{\sum_{i=1}^{N} \bigl(w_{i,t}\bigr)^{2}}
$$

where `w_{i,t}` are the normalized weights at time `t`. ESS ranges from 1
(degenerate) to `N` (uniform weights). Resampling is typically triggered when
`ESS_t < N/2`.

### Weight Entropy

Entropy provides a complementary view of weight concentration:

$$
H_t \;=\; -\sum_{i=1}^{N} w_{i,t}\,\log\bigl(w_{i,t}\bigr)
$$

Maximum entropy `log(N)` corresponds to uniform weights; `H_t = 0` means a
single particle carries all mass.

### Convergence

For a well-specified particle filter, the RMSE of filtered state estimates
should decay as `O(1 / sqrt(N))` with the number of particles:

$$
\mathrm{RMSE}(N) \;\approx\; \frac{C}{\sqrt{N}}
$$

Plotting RMSE against N (log-log scale) should yield a slope of approximately
-0.5.

### Particle Degeneracy

Degeneracy is diagnosed by monitoring:

- ESS trajectory over time
- Maximum weight (should stay well below 1)
- Number of unique ancestors after resampling
- Weight variance / coefficient of variation

### Model Comparison

The particle filter provides an unbiased estimator of the marginal likelihood
`p(y_{1:T} | M)` for model `M`. Log-likelihood values across competing models
can be compared via Bayes factors or information criteria.

## Directory Layout

```
examples/diagnostics/
├── README.md
├── data/                  # Datasets (symlinks to shared data)
├── notebooks/             # Jupyter notebooks (student-facing)
├── solutions/             # Solution notebooks
├── R_validation/          # Cross-validation scripts using pomp
└── stata_validation/      # Cross-validation scripts using sspace
```
