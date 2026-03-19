"""Sequential Importance Resampling (SIR) Particle Filter.

The SIR filter supports a general proposal distribution q(x_t | x_{t-1}, y_t).
The weight update includes the importance correction:

    log w_t += log p(y_t | x_t) + log p(x_t | x_{t-1}) - log q(x_t | x_{t-1}, y_t)

If the model does not provide a custom proposal, SIR falls back to the
Bootstrap PF behavior (proposal = prior, correction terms cancel).

References
----------
- Doucet, A., Godsill, S. & Andrieu, C. (2000). On sequential Monte Carlo
  sampling methods for Bayesian filtering.
- Arulampalam, M.S. et al. (2002). A tutorial on particle filters for
  online nonlinear/non-Gaussian Bayesian tracking.
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


class SIR(BaseParticleFilter):
    """Sequential Importance Resampling particle filter.

    Supports a general proposal distribution. Falls back to Bootstrap PF
    if the model does not provide a custom proposal.

    Parameters
    ----------
    model : ParticleFilterModel
        State-space model. May optionally implement:
        - proposal_sample(particles, y_t, t, rng): sample from proposal
        - log_proposal_density(x_curr, x_prev, y_t, t): log-density of proposal
    config : PFConfig
        Particle filter configuration.

    Notes
    -----
    If the model has ``proposal_sample`` and ``log_proposal_density`` methods,
    SIR uses the custom proposal. Otherwise, it falls back to using the
    transition as the proposal (Bootstrap PF behavior).

    Examples
    --------
    >>> from particlefilterbox.filters.sir import SIR
    >>> pf = SIR(model_with_proposal, config)
    >>> results = pf.filter(observations)
    """

    def __init__(self, model: ParticleFilterModel, config: PFConfig) -> None:
        super().__init__(model, config)
        self._has_custom_proposal = self._check_custom_proposal()
        self._x_prev: NDArray[np.float64] | None = None

    def _check_custom_proposal(self) -> bool:
        """Check if the model provides a custom proposal distribution.

        Returns
        -------
        bool
            True if the model has both proposal_sample and log_proposal_density.
        """
        has_sample = hasattr(self.model, "proposal_sample") and callable(
            getattr(self.model, "proposal_sample", None)
        )
        has_density = hasattr(self.model, "log_proposal_density") and callable(
            getattr(self.model, "log_proposal_density", None)
        )
        return has_sample and has_density

    @property
    def uses_custom_proposal(self) -> bool:
        """Whether the SIR filter is using a custom proposal."""
        return self._has_custom_proposal

    def _propagate(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> ParticleCloud:
        """Propagate particles through the proposal distribution.

        If the model has a custom proposal, use it. Otherwise, fall back
        to the prior (transition).

        Parameters
        ----------
        cloud : ParticleCloud
            Current particle cloud.
        y_t : NDArray[np.float64]
            Observation at time t.
        t : int
            Time index.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        ParticleCloud
            Cloud with propagated particles, same weights.
        """
        from particlefilterbox.core.cloud import ParticleCloud as _Cloud

        # Store previous particles for weight computation
        self._x_prev = cloud.particles.copy()

        is_missing = bool(np.any(np.isnan(y_t)))

        if self._has_custom_proposal and not is_missing:
            # Use custom proposal: x_t ~ q(x_t | x_{t-1}, y_t)
            new_particles = self.model.proposal_sample(  # type: ignore[attr-defined]
                cloud.particles, y_t, t, rng=rng
            )
        else:
            # Fall back to prior: x_t ~ p(x_t | x_{t-1})
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
        """Compute incremental log-weights with importance correction.

        With custom proposal:
            log w += log p(y|x) + log p(x|x_prev) - log q(x|x_prev, y)

        Without custom proposal (Bootstrap fallback):
            log w += log p(y|x)

        Parameters
        ----------
        cloud : ParticleCloud
            Current particle cloud (already propagated).
        y_t : NDArray[np.float64]
            Observation at time t.
        t : int
            Time index.

        Returns
        -------
        NDArray[np.float64]
            Incremental log-weights of shape (N,).
        """
        # Observation likelihood: log p(y_t | x_t)
        log_obs = self.model.log_observation_likelihood(cloud.particles, y_t, t)

        if self._has_custom_proposal:
            assert self._x_prev is not None
            # Transition density: log p(x_t | x_{t-1})
            log_trans = self.model.log_transition_density(
                cloud.particles, self._x_prev, t
            )
            # Proposal density: log q(x_t | x_{t-1}, y_t)
            log_prop = self.model.log_proposal_density(  # type: ignore[attr-defined]
                cloud.particles, self._x_prev, y_t, t
            )
            # Importance weight correction
            return log_obs + log_trans - log_prop
        else:
            # Bootstrap fallback: transition/proposal cancel
            return log_obs
