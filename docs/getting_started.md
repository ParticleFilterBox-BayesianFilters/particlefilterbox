# Getting Started

## Installation

### Basic Installation

```bash
pip install particlefilterbox
```

### With Optional Dependencies

```bash
# Visualization (matplotlib)
pip install particlefilterbox[viz]

# Command-line interface (typer)
pip install particlefilterbox[cli]

# Documentation tools
pip install particlefilterbox[docs]

# GPU acceleration (requires CUDA)
pip install particlefilterbox[accel]

# Everything
pip install particlefilterbox[all]
```

### Development Installation

```bash
git clone https://github.com/nodesecon/particlefilterbox.git
cd particlefilterbox
pip install -e ".[dev]"
```

## First Steps

### 1. Load Data

```python
from particlefilterbox.datasets import load_dataset

# Load bundled dataset
sp500 = load_dataset('sp500_returns')
print(sp500.head())

# Or use your own data
import numpy as np
y = np.loadtxt('my_data.csv')
```

### 2. Create a Model

```python
from particlefilterbox.models.sv import SVModel

# Stochastic Volatility model with known parameters
model = SVModel(mu=0.0, phi=0.97, sigma_eta=0.15)
```

### 3. Run a Particle Filter

```python
from particlefilterbox.filters.bootstrap import BootstrapFilter

pf = BootstrapFilter(
    model=model,
    n_particles=1000,
    resampling='systematic',
)
results = pf.filter(sp500['returns'].values)

print(f"Log-likelihood: {results.log_likelihood:.4f}")
```

### 4. Visualize Results

```python
from particlefilterbox.visualization import (
    plot_filtered_state,
    plot_ess_timeline,
    set_theme,
)

set_theme('nodesecon')
fig, ax = plot_filtered_state(results, state_idx=0)
fig.savefig('filtered_volatility.png')

fig, ax = plot_ess_timeline(results)
fig.savefig('ess.png')
```

### 5. Estimate Parameters

```python
from particlefilterbox.pmcmc.pmmh import PMMH

pmmh = PMMH(
    model=SVModel(),
    n_particles=200,
    n_iterations=10000,
)
mcmc_results = pmmh.run(sp500['returns'].values)
print(mcmc_results.summary())
```

### 6. Generate a Report

```python
from particlefilterbox.reports import PFReportTransformer

transformer = PFReportTransformer()
report = transformer.transform(results)
report.to_html('report.html')
report.to_markdown('report.md')
```

## Next Steps

- Read the [User Guide](guide/particle_filters.md) for detailed documentation
- Check [Examples](examples/sv.md) for complete workflows
- See the [API Reference](api/core.md) for all available functions
