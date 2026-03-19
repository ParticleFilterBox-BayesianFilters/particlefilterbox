"""Bootstrap Particle Filter (Gordon et al., 1993).

The Bootstrap PF uses the state transition as the proposal distribution:
    q(x_t | x_{t-1}, y_t) = p(x_t | x_{t-1})

This is the simplest particle filter. The weight update reduces to:
    log w_t^(i) += log p(y_t | x_t^(i))
because the transition/proposal terms cancel.

References
----------
- Gordon, N.J., Salmond, D.J. & Smith, A.F.M. (1993). Novel approach to
  nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F,
  140(2), 107-113.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from particlefilterbox.filters.base import BaseParticleFilter

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from particlefilterbox.core.cloud import ParticleCloud
    from particlefilterbox.core.config import PFConfig
    from particlefilterbox.core.model import ParticleFilterModel


class BootstrapPF(BaseParticleFilter):
    """Bootstrap Particle Filter.

    Uses the prior (state transition) as the proposal distribution.

    Parameters
    ----------
    model : ParticleFilterModel
        State-space model with transition and log_observation_likelihood.
    config : PFConfig
        Particle filter configuration.
    """

    def __init__(self, model: ParticleFilterModel, config: PFConfig) -> None:
        super().__init__(model, config)

    def _propagate(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> ParticleCloud:
        """Propagate particles using the prior (transition distribution)."""
        from particlefilterbox.core.cloud import ParticleCloud as _Cloud

        new_particles = self.model.transition(cloud.particles, t, rng)

        new_cloud = _Cloud(cloud.n_particles, cloud.k_states)
        new_cloud.particles = new_particles
        new_cloud.set_log_weights(cloud.log_weights.copy())
        return new_cloud

    def _compute_weights(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        """Compute incremental log-weights using observation likelihood.

        Since proposal = prior, weights are just the observation likelihood.
        """
        return self.model.log_observation_likelihood(cloud.particles, y_t, t)
