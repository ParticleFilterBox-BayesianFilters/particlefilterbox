# particlefilterbox

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
