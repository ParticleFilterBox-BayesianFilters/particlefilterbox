"""Locally Optimal Particle Filter.

Uses the optimal proposal distribution q*(x_t|x_{t-1},y_t) = p(x_t|x_{t-1},y_t)
to minimize weight variance. Requires the model to provide the optimal proposal
analytically.

References
----------
Doucet, A., Godsill, S. & Andrieu, C. (2000). On sequential Monte Carlo
sampling methods for Bayesian filtering. Statistics and Computing, 10(3), 197-208.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from particlefilterbox._logging import get_logger
from particlefilterbox.filters.base import BaseParticleFilter

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from particlefilterbox.core.cloud import ParticleCloud
    from particlefilterbox.core.config import PFConfig
    from particlefilterbox.core.model import ParticleFilterModel

logger = get_logger(__name__)


class LocallyOptimalPF(BaseParticleFilter):
    """Locally Optimal Particle Filter using the optimal proposal.

    This filter uses the optimal proposal distribution:
        q*(x_t | x_{t-1}^(i), y_t) = p(x_t | x_{t-1}^(i), y_t)

    The weight update simplifies to:
        w_t^(i) = p(y_t | x_{t-1}^(i))  (predictive likelihood)

    The model must implement one of:
    - ``optimal_proposal(particles, observation, t, rng)`` -> new_particles
    - ``optimal_proposal_params(particles, observation, t)`` -> (means, covs)

    Parameters
    ----------
    model : ParticleFilterModel
        State-space model with optimal proposal capability.
    config : PFConfig
        Particle filter configuration.

    Raises
    ------
    ValueError
        If model does not provide optimal_proposal or optimal_proposal_params.
    """

    def __init__(
        self,
        model: ParticleFilterModel,
        config: PFConfig,
    ) -> None:
        super().__init__(model=model, config=config)

        self._has_optimal_proposal = hasattr(model, "optimal_proposal") and callable(
            getattr(model, "optimal_proposal", None)
        )
        self._has_optimal_params = hasattr(model, "optimal_proposal_params") and callable(
            getattr(model, "optimal_proposal_params", None)
        )

        if not self._has_optimal_proposal and not self._has_optimal_params:
            raise ValueError(
                "LocallyOptimalPF requires model to implement "
                "'optimal_proposal(particles, observation, t, rng)' or "
                "'optimal_proposal_params(particles, observation, t)'"
            )

        logger.info(
            "LocallyOptimalPF initialized (optimal_proposal=%s, optimal_params=%s)",
            self._has_optimal_proposal,
            self._has_optimal_params,
        )

    def _sample_optimal_proposal(
        self,
        particles: NDArray[np.float64],
        observation: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample from the optimal proposal distribution.

        Parameters
        ----------
        particles : ndarray of shape (N, k_states)
            Current particles (ancestors).
        observation : ndarray of shape (k_obs,)
            Current observation.
        t : int
            Current time step.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        new_particles : ndarray of shape (N, k_states)
            Particles sampled from optimal proposal.
        """
        if self._has_optimal_proposal:
            return self.model.optimal_proposal(  # type: ignore[attr-defined]
                particles, observation, t, rng
            )

        # Use optimal_proposal_params to get (means, covs) and sample
        means, covs = self.model.optimal_proposal_params(  # type: ignore[attr-defined]
            particles, observation, t
        )

        n_particles = particles.shape[0]
        k_states = particles.shape[1]
        new_particles = np.empty((n_particles, k_states), dtype=np.float64)

        for i in range(n_particles):
            mean_i = means[i].flatten()
            cov_i = covs[i] if covs.ndim == 3 else covs

            if k_states == 1:
                std_i = np.sqrt(float(cov_i.flatten()[0]))
                new_particles[i, 0] = rng.normal(mean_i[0], std_i)
            else:
                new_particles[i] = rng.multivariate_normal(mean_i, cov_i)

        return new_particles

    def _predictive_log_likelihood(
        self,
        particles: NDArray[np.float64],
        observation: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Compute predictive log-likelihood p(y_t | x_{t-1}^(i)).

        For the optimal proposal, the weight is just the predictive likelihood.

        Parameters
        ----------
        particles : ndarray of shape (N, k_states)
            Ancestor particles x_{t-1}.
        observation : ndarray of shape (k_obs,)
            Current observation.
        t : int
            Current time step.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        log_pred : ndarray of shape (N,)
            Log predictive likelihoods.
        """
        n_particles = particles.shape[0]

        if hasattr(self.model, "predictive_log_likelihood"):
            # The model's predictive_log_likelihood is vectorized over particles:
            # it accepts ancestor particles of shape (N, k_states) and returns
            # log p(y_t | x_{t-1}^(i)) of shape (N,). Call it once on the full
            # batch rather than looping particle-by-particle.
            log_pred = np.asarray(
                self.model.predictive_log_likelihood(  # type: ignore[attr-defined]
                    observation, particles, t
                ),
                dtype=np.float64,
            ).reshape(-1)
            if log_pred.shape[0] != n_particles:
                raise ValueError(
                    "predictive_log_likelihood returned shape "
                    f"{log_pred.shape}, expected ({n_particles},)"
                )
            return log_pred

        # Monte Carlo approximation: integrate p(y|x_t) p(x_t|x_{t-1}) dx_t
        n_mc = 50
        log_pred = np.empty(n_particles, dtype=np.float64)
        for i in range(n_particles):
            particle_i = np.tile(particles[i : i + 1], (n_mc, 1))
            x_mc = self.model.transition(particle_i, t, rng)
            log_liks = self.model.log_observation_likelihood(x_mc, observation, t)
            log_pred[i] = float(logsumexp(log_liks) - np.log(n_mc))

        return log_pred

    def _propagate(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> ParticleCloud:
        """Propagate particles using the optimal proposal.

        Samples from q*(x_t | x_{t-1}, y_t) instead of the transition prior.
        Stores ancestor particles for weight computation.
        """
        from particlefilterbox.core.cloud import ParticleCloud as _Cloud

        # Save ancestor particles for predictive likelihood computation
        self._ancestor_particles = cloud.particles.copy()

        new_particles = self._sample_optimal_proposal(cloud.particles, y_t, t, rng)

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
        """Compute incremental log-weights as predictive likelihood.

        For the optimal proposal, weights are p(y_t | x_{t-1}^(i)).
        """
        rng = self._get_rng()
        return self._predictive_log_likelihood(self._ancestor_particles, y_t, t, rng)
