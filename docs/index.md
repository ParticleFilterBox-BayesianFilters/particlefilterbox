# particlefilterbox

**Particle filtering and Sequential Monte Carlo methods in Python.**

particlefilterbox is a comprehensive Python library for state estimation and parameter
inference using particle filters and Sequential Monte Carlo (SMC) methods. Designed for
economists, financial engineers, and applied statisticians.

## Features

- **Particle Filters**: Bootstrap, Auxiliary, Extended Kalman, Unscented
- **Smoothers**: FFBSm, Two-Filter, Fixed-Lag
- **SMC**: SMC Sampler, Tempering, SMC^2, IBIS, Waste-Free SMC
- **PMCMC**: PMMH, Particle Gibbs, Conditional SMC
- **Models**: Stochastic Volatility, Local Level, Linear Gaussian, DSGE
- **Diagnostics**: ESS, weight analysis, convergence, R-hat
- **Visualization**: Comprehensive plotting with themes
- **Reports**: HTML, LaTeX, and Markdown report generation
- **CLI**: Command-line interface for quick analysis
- **Datasets**: Bundled simulated financial and macro datasets

## Quick Start

```python
from particlefilterbox.models.sv import SVModel
from particlefilterbox.filters.bootstrap import BootstrapFilter
from particlefilterbox.visualization import plot_filtered_state, set_theme

# Create model and filter
model = SVModel(mu=0.0, phi=0.97, sigma_eta=0.15)
pf = BootstrapFilter(model=model, n_particles=1000)

# Run filter
results = pf.filter(observations)

# Visualize
set_theme('nodesecon')
fig, ax = plot_filtered_state(results, state_idx=0)
fig.savefig('filtered.png')
```

## Installation

```bash
pip install particlefilterbox

# With visualization support
pip install particlefilterbox[viz]

# With CLI support
pip install particlefilterbox[cli]

# Full installation
pip install particlefilterbox[all]
```

## CLI

```bash
pfbox filter data.csv --model sv --n-particles 1000 --plot
pfbox estimate data.csv --model sv --method pmmh --n-iterations 5000
pfbox compare data.csv --models sv,local_level --n-particles 2000
pfbox simulate --model sv --n-obs 500 --seed 42
```

## References

- Doucet, A. & Johansen, A.M. (2011). A tutorial on particle filtering and smoothing.
- Andrieu, C., Doucet, A. & Holenstein, R. (2010). Particle MCMC methods. JRSS-B.
- Del Moral, P., Doucet, A. & Jasra, A. (2006). Sequential Monte Carlo samplers. JRSS-B.
- Chopin, N. & Papaspiliopoulos, O. (2020). An Introduction to Sequential Monte Carlo.

## License

MIT License - NodeSEcon
