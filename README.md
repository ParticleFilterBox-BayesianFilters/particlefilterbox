# particlefilterbox

[![CI](https://github.com/ParticleFilterBox-BayesianFilters/particlefilterbox/actions/workflows/ci.yml/badge.svg)](https://github.com/ParticleFilterBox-BayesianFilters/particlefilterbox/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ParticleFilterBox-BayesianFilters/particlefilterbox/branch/main/graph/badge.svg)](https://codecov.io/gh/ParticleFilterBox-BayesianFilters/particlefilterbox)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://badge.fury.io/py/particlefilterbox.svg)](https://badge.fury.io/py/particlefilterbox)
[![Python versions](https://img.shields.io/pypi/pyversions/particlefilterbox)](https://pypi.org/project/particlefilterbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Development Status](https://img.shields.io/badge/development%20status-alpha-orange)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/particlefilterbox?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/particlefilterbox)
[![Documentation](https://readthedocs.org/projects/particlefilterbox/badge/?version=latest)](https://particlefilterbox.readthedocs.io/)

Particle filtering and Sequential Monte Carlo methods for state estimation.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from particlefilterbox.core import ParticleCloud, PFConfig
from particlefilterbox.resampling import systematic_resample

config = PFConfig(n_particles=1000, resampling='systematic')
config.validate()

cloud = ParticleCloud(n_particles=1000, k_states=1)
cloud.set_uniform_weights()

indices = systematic_resample(cloud.normalized_weights)
cloud.resample(indices)
```

## License

MIT
