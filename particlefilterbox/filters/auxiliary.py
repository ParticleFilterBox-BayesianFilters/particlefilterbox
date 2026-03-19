"""Auxiliary Particle Filter (Pitt & Shephard, 1999).

Implements pre-selection of particles via approximate likelihood evaluation
at the predicted observation, leading to better particle allocation in
regions of high likelihood.

References
----------
Pitt, M.K. & Shephard, N. (1999). Filtering via Simulation: Auxiliary
Particle Filters. JASA, 94(446), 590-599.
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


class AuxiliaryPF(BaseParticleFilter):
    """Auxiliary Particle Filter with first-stage pre-selection.

    The APF introduces a pre-selection step that evaluates an approximate
    likelihood at the predicted mean of each particle's transition. This
    allows the filter to preferentially propagate particles that are likely
    to explain the current observation.

    Parameters
    ----------
    model : ParticleFilterModel
        State-space model. Optionally implements ``transition_mean(particles, t)``
        for computing the mean of the transition distribution without noise.
        If not provided, the transition is called with zero noise as approximation.
    config : PFConfig
        Particle filter configuration.

    Notes
    -----
    The algorithm proceeds in two stages:

    1. **First stage (pre-selection)**: Compute approximate weights
       lambda_i = p(y_t | mu_t^(i)) where mu_t^(i) is the transition mean.
       Resample indices proportional to w_{t-1}^(i) * lambda_i.

    2. **Second stage (propagation + correction)**: Propagate particles
       through the transition model using resampled ancestors. Correct
       weights: w_t^(i) = p(y_t | x_t^(i)) / lambda_{k^(i)}.

    Examples
    --------
    >>> from particlefilterbox.filters import AuxiliaryPF
    >>> apf = AuxiliaryPF(model=my_model, config=config)
    >>> result = apf.filter(observations)
    """

    def __init__(
        self,
        model: ParticleFilterModel,
        config: PFConfig,
    ) -> None:
        super().__init__(model=model, config=config)
        self._has_transition_mean = hasattr(model, "transition_mean") and callable(
            getattr(model, "transition_mean", None)
        )
        logger.info(
            "AuxiliaryPF initialized (transition_mean available: %s)",
            self._has_transition_mean,
        )

    def _transition_mean(
        self,
        particles: NDArray[np.floating],
        t: int,
    ) -> NDArray[np.floating]:
        """Compute the mean of the transition distribution for each particle.

        Parameters
        ----------
        particles : ndarray of shape (N, k_states)
            Current particle positions.
        t : int
            Current time step.

        Returns
        -------
        mu : ndarray of shape (N, k_states)
            Predicted means for each particle.
        """
        if self._has_transition_mean:
            return self.model.transition_mean(particles, t)  # type: ignore[attr-defined]

        # Fallback: call transition with a fixed-seed RNG to get a
        # deterministic-ish approximation of the mean.
        rng = np.random.default_rng(0)
        return self.model.transition(particles, t, rng)

    def _first_stage_weights(
        self,
        log_weights: NDArray[np.floating],
        particles: NDArray[np.floating],
        observation: NDArray[np.floating],
        t: int,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute first-stage pre-selection weights.

        Parameters
        ----------
        log_weights : ndarray of shape (N,)
            Current log weights.
        particles : ndarray of shape (N, k_states)
            Current particle positions.
        observation : ndarray of shape (k_obs,)
            Current observation.
        t : int
            Current time step.

        Returns
        -------
        first_stage_log_weights : ndarray of shape (N,)
            Combined log weights for pre-selection (log w + log lambda).
        log_lambdas : ndarray of shape (N,)
            Log approximate likelihoods at transition means.
        """
        # Compute transition means
        mu = self._transition_mean(particles, t)

        # Evaluate approximate likelihood at transition means
        # log_observation_likelihood expects (particles, y_t, t) -> (N,)
        log_lambdas = self.model.log_observation_likelihood(mu, observation, t)

        # First-stage weights: w_{t-1} * lambda
        first_stage_log_weights = log_weights + log_lambdas

        return first_stage_log_weights, log_lambdas

    def _propagate(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> ParticleCloud:
        """APF propagation: pre-select then propagate.

        This overrides the base class to implement the full APF step
        (pre-selection + propagation). The weight correction is done
        in _compute_weights.
        """
        from particlefilterbox.core.cloud import ParticleCloud as _Cloud
        from particlefilterbox.resampling import systematic_resample

        particles = cloud.particles
        log_weights = cloud.log_weights

        # === FIRST STAGE: Pre-selection ===
        first_stage_log_weights, log_lambdas = self._first_stage_weights(
            log_weights, particles, y_t, t
        )

        # Normalize first-stage weights for resampling
        norm_first = np.exp(first_stage_log_weights - logsumexp(first_stage_log_weights))
        # Clip for numerical safety
        norm_first = np.clip(norm_first, 0.0, None)
        norm_first /= norm_first.sum()

        # Resample ancestor indices based on first-stage weights
        ancestor_indices = systematic_resample(norm_first, rng=rng)

        # === SECOND STAGE: Propagate selected particles ===
        selected_particles = particles[ancestor_indices]
        new_particles = self.model.transition(selected_particles, t, rng)

        # Build new cloud and store lambdas for weight correction
        new_cloud = _Cloud(cloud.n_particles, cloud.k_states)
        new_cloud.particles = new_particles
        new_cloud.set_uniform_weights()

        # Store ancestor lambdas for weight correction in _compute_weights
        self._selected_log_lambdas = log_lambdas[ancestor_indices]

        return new_cloud

    def _compute_weights(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        """Compute corrected APF weights: p(y|x_t) / lambda_{ancestor}.

        The incremental weights for APF are the true likelihood divided
        by the approximate likelihood used in pre-selection.
        """
        # True log-likelihood at propagated positions
        log_lik = self.model.log_observation_likelihood(cloud.particles, y_t, t)

        # Correction: subtract the approximate likelihood used in pre-selection
        return log_lik - self._selected_log_lambdas
