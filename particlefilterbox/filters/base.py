"""Base particle filter with generic filtering loop.

This module implements the BaseParticleFilter abstract class that provides
the common filtering loop shared by all particle filters:
    initialize -> propagate -> weight -> normalize -> resample

References
----------
- Gordon, N.J., Salmond, D.J. & Smith, A.F.M. (1993).
- Doucet, A., Godsill, S. & Andrieu, C. (2000).
- Arulampalam, M.S. et al. (2002).
- Chopin, N. & Papaspiliopoulos, O. (2020). Cap. 10.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from particlefilterbox._logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from particlefilterbox.core.cloud import ParticleCloud
    from particlefilterbox.core.config import PFConfig
    from particlefilterbox.core.model import ParticleFilterModel

logger = get_logger(__name__)


class BaseParticleFilter(ABC):
    """Abstract base class for particle filters.

    Implements the generic filtering loop:
        initialize -> propagate -> weight -> normalize -> resample

    Subclasses must implement:
        - _propagate(cloud, y_t, t, rng): propagate particles via proposal
        - _compute_weights(cloud, y_t, t): compute incremental log-weights

    Parameters
    ----------
    model : ParticleFilterModel
        The state-space model to filter.
    config : PFConfig
        Configuration for the particle filter (n_particles, resampling, etc.).
    """

    def __init__(self, model: ParticleFilterModel, config: PFConfig) -> None:
        self.model = model
        self.config = config
        self.n_particles = config.n_particles
        self._rng: np.random.Generator | None = None

    def _get_rng(self) -> np.random.Generator:
        """Get or create the random number generator."""
        if self._rng is None:
            seed = self.config.seed if hasattr(self.config, "seed") else None
            self._rng = np.random.default_rng(seed)
        return self._rng

    def initialize(self, rng: np.random.Generator) -> ParticleCloud:
        """Initialize particle cloud from the prior p(x_0).

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        ParticleCloud
            Initial particle cloud with uniform weights.
        """
        from particlefilterbox.core.cloud import ParticleCloud

        state_dim = self.model.k_states
        cloud = ParticleCloud(self.n_particles, state_dim)

        # Sample initial particles from the model's prior
        particles = self.model.initial_distribution(self.n_particles, rng)
        if particles.ndim == 1:
            particles = particles[:, np.newaxis]

        cloud.particles = particles
        cloud.set_uniform_weights()
        return cloud

    def filter(
        self,
        endog: NDArray[np.float64],
        mask: NDArray[np.bool_] | None = None,
    ) -> ParticleFilterResults:
        """Run the full particle filter on observed data.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observed data of shape (T,) or (T, obs_dim).
        mask : NDArray[np.bool_] | None
            Optional boolean mask of shape (T,). True indicates missing data.

        Returns
        -------
        ParticleFilterResults
            Complete filtering results.
        """
        rng = self._get_rng()

        if endog.ndim == 1:
            endog = endog[:, np.newaxis]

        n_obs, _obs_dim = endog.shape
        state_dim = self.model.k_states

        missing_mask = mask if mask is not None else np.any(np.isnan(endog), axis=1)

        filtered_means = np.zeros((n_obs, state_dim))
        filtered_covs = np.zeros((n_obs, state_dim, state_dim))
        log_likelihoods = np.zeros(n_obs)
        ess_history = np.zeros(n_obs)
        resampled = np.zeros(n_obs, dtype=bool)

        cloud = self.initialize(rng)

        for t in range(n_obs):
            y_t = endog[t]
            is_missing = bool(missing_mask[t])

            cloud, ll_t, did_resample = self._filter_step_internal(cloud, y_t, t, is_missing, rng)

            filtered_means[t] = self._compute_mean(cloud)
            filtered_covs[t] = self._compute_cov(cloud, filtered_means[t])
            log_likelihoods[t] = ll_t
            ess_history[t] = self._compute_ess(cloud)
            resampled[t] = did_resample

        total_ll = float(np.sum(log_likelihoods))

        return ParticleFilterResults(
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            log_likelihood=total_ll,
            log_likelihoods=log_likelihoods,
            ess_history=ess_history,
            resampled=resampled,
            n_particles=self.n_particles,
            final_cloud=cloud,
        )

    def filter_step(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
    ) -> tuple[ParticleCloud, float]:
        """Run a single step of the particle filter (online mode).

        Parameters
        ----------
        cloud : ParticleCloud
            Current particle cloud.
        y_t : NDArray[np.float64]
            Observation at time t.
        t : int
            Time index.

        Returns
        -------
        tuple[ParticleCloud, float]
            Updated particle cloud and incremental log-likelihood.
        """
        rng = self._get_rng()
        if y_t.ndim == 0:
            y_t = y_t.reshape(1)
        is_missing = self._handle_missing(y_t)
        cloud, ll_t, _ = self._filter_step_internal(cloud, y_t, t, is_missing, rng)
        return cloud, ll_t

    def _filter_step_internal(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
        is_missing: bool,
        rng: np.random.Generator,
    ) -> tuple[ParticleCloud, float, bool]:
        """Internal single step of the particle filter."""
        from particlefilterbox.core.cloud import ParticleCloud as _Cloud

        cloud = self._propagate(cloud, y_t, t, rng)

        if is_missing:
            return cloud, 0.0, False

        log_w_incremental = self._compute_weights(cloud, y_t, t)
        new_log_weights = cloud.log_weights + log_w_incremental
        lse_new: float = float(np.asarray(logsumexp(new_log_weights)))
        lse_old: float = float(np.asarray(logsumexp(cloud.log_weights)))
        ll_t = lse_new - lse_old

        new_log_weights = new_log_weights - lse_new

        new_cloud = _Cloud(cloud.n_particles, cloud.k_states)
        new_cloud.particles = cloud.particles.copy()
        new_cloud.set_log_weights(new_log_weights)

        new_cloud, did_resample = self._maybe_resample(new_cloud, rng)

        return new_cloud, ll_t, did_resample

    @abstractmethod
    def _propagate(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> ParticleCloud:
        """Propagate particles through the proposal distribution."""
        ...

    @abstractmethod
    def _compute_weights(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        """Compute incremental log-weights of shape (N,)."""
        ...

    def _maybe_resample(
        self,
        cloud: ParticleCloud,
        rng: np.random.Generator,
    ) -> tuple[ParticleCloud, bool]:
        """Resample particles if ESS falls below threshold."""
        from particlefilterbox.resampling import systematic_resample

        ess = self._compute_ess(cloud)
        threshold = self.config.ess_threshold * self.n_particles

        if ess < threshold:
            weights = cloud.normalized_weights
            indices = systematic_resample(weights, rng=rng)
            cloud.resample(indices)
            return cloud, True

        return cloud, False

    def _handle_missing(self, y_t: NDArray[np.float64]) -> bool:
        """Check if observation is missing (all NaN)."""
        return bool(np.all(np.isnan(y_t)))

    def _compute_ess(self, cloud: ParticleCloud) -> float:
        """Compute Effective Sample Size."""
        return cloud.ess

    def _compute_mean(self, cloud: ParticleCloud) -> NDArray[np.float64]:
        """Compute weighted mean of particles."""
        return cloud.weighted_mean()

    def _compute_cov(
        self,
        cloud: ParticleCloud,
        mean: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute weighted covariance of particles."""
        return cloud.weighted_cov()


class ParticleFilterResults:
    """Results from running a particle filter.

    Attributes
    ----------
    filtered_means : NDArray[np.float64]
        Array of shape (T, state_dim) with filtered state means.
    filtered_covs : NDArray[np.float64]
        Array of shape (T, state_dim, state_dim) with filtered state covariances.
    log_likelihood : float
        Total log-likelihood.
    log_likelihoods : NDArray[np.float64]
        Array of shape (T,) with incremental log-likelihoods.
    ess_history : NDArray[np.float64]
        Array of shape (T,) with ESS at each step.
    resampled : NDArray[np.bool_]
        Array of shape (T,) indicating if resampling occurred.
    n_particles : int
        Number of particles used.
    final_cloud : ParticleCloud | None
        The final particle cloud after filtering.
    """

    def __init__(
        self,
        filtered_means: NDArray[np.float64],
        filtered_covs: NDArray[np.float64],
        log_likelihood: float,
        log_likelihoods: NDArray[np.float64],
        ess_history: NDArray[np.float64],
        resampled: NDArray[np.bool_],
        n_particles: int,
        final_cloud: ParticleCloud | None = None,
    ) -> None:
        self.filtered_means = filtered_means
        self.filtered_covs = filtered_covs
        self.log_likelihood = log_likelihood
        self.log_likelihoods = log_likelihoods
        self.ess_history = ess_history
        self.resampled = resampled
        self.n_particles = n_particles
        self.final_cloud = final_cloud
