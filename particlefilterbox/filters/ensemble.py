"""Ensemble Particle Filter.

Hybrid approach combining ensemble Kalman updates with particle filtering.
Uses perturbed observations to update the particle ensemble via a
Kalman-like gain computed from ensemble statistics.

References
----------
Chopin, N. & Papaspiliopoulos, O. (2020). An Introduction to Sequential
Monte Carlo. Springer. Cap. 11.

Evensen, G. (2003). The Ensemble Kalman Filter: theoretical formulation
and practical implementation. Ocean Dynamics, 53, 343-367.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from particlefilterbox._logging import get_logger
from particlefilterbox.filters.base import BaseParticleFilter, ParticleFilterResults

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from particlefilterbox.core.cloud import ParticleCloud
    from particlefilterbox.core.config import PFConfig
    from particlefilterbox.core.model import ParticleFilterModel

logger = get_logger(__name__)


class EnsemblePF(BaseParticleFilter):
    """Ensemble Particle Filter with Kalman-like ensemble update.

    Combines particle filter propagation with ensemble Kalman updates.
    Instead of standard importance weighting and resampling, uses a
    Kalman gain computed from ensemble statistics to update particles.

    Parameters
    ----------
    model : ParticleFilterModel
        State-space model.
    config : PFConfig
        Particle filter configuration.
    ensemble_fraction : float
        Fraction of steps using ensemble update vs standard PF
        (default 1.0 = all ensemble).

    Examples
    --------
    >>> epf = EnsemblePF(model=model, config=config)
    >>> result = epf.filter(observations)
    """

    def __init__(
        self,
        model: ParticleFilterModel,
        config: PFConfig,
        ensemble_fraction: float = 1.0,
    ) -> None:
        super().__init__(model=model, config=config)
        self._ensemble_fraction = ensemble_fraction

        logger.info("EnsemblePF initialized (ensemble_fraction=%.2f)", ensemble_fraction)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _predict_observations(
        self,
        particles: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        """Compute predicted observations for each particle.

        Parameters
        ----------
        particles : ndarray of shape (N, k_states)
            Current particles.
        t : int
            Time step.

        Returns
        -------
        y_pred : ndarray of shape (N, k_obs)
            Predicted observations.
        """
        n_particles = particles.shape[0]
        k_obs = self.model.k_obs

        if hasattr(self.model, "observation_function"):
            obs_fn = self.model.observation_function  # type: ignore[attr-defined]
            if callable(obs_fn):
                y_pred = np.empty((n_particles, k_obs), dtype=np.float64)
                for i in range(n_particles):
                    y_pred[i] = np.atleast_1d(obs_fn(particles[i], t))[:k_obs]
                return y_pred

        if hasattr(self.model, "observation_mean"):
            return np.atleast_2d(
                self.model.observation_mean(particles, t)  # type: ignore[attr-defined]
            )

        # Default: identity observation
        return particles[:, :k_obs]

    def _get_obs_noise_cov(self, t: int) -> NDArray[np.float64]:
        """Get observation noise covariance."""
        if hasattr(self.model, "R"):
            r_attr = self.model.R  # type: ignore[attr-defined]
            if callable(r_attr):
                return np.atleast_2d(r_attr(t))
        if hasattr(self.model, "observation_noise_cov"):
            return np.atleast_2d(
                self.model.observation_noise_cov(t)  # type: ignore[attr-defined]
            )
        if hasattr(self.model, "sigma_y"):
            k_obs = self.model.k_obs
            return np.eye(k_obs) * self.model.sigma_y**2  # type: ignore[attr-defined]
        return np.eye(self.model.k_obs) * 1.0

    # ------------------------------------------------------------------
    # Ensemble Kalman mechanics
    # ------------------------------------------------------------------

    def _compute_kalman_gain(
        self,
        particles: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        obs_cov: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute ensemble Kalman gain.

        K = Cov(x, y_pred) * (Cov(y_pred, y_pred) + R)^{-1}

        Parameters
        ----------
        particles : ndarray of shape (N, k_states)
            Predicted particles.
        y_pred : ndarray of shape (N, k_obs)
            Predicted observations.
        obs_cov : ndarray of shape (k_obs, k_obs)
            Observation noise covariance.

        Returns
        -------
        gain : ndarray of shape (k_states, k_obs)
            Kalman gain.
        """
        n = particles.shape[0]

        x_mean = np.mean(particles, axis=0)
        y_mean = np.mean(y_pred, axis=0)

        dx = particles - x_mean
        dy = y_pred - y_mean

        # Cross-covariance Cov(x, y)
        cov_xy = (dx.T @ dy) / (n - 1)

        # Observation covariance Cov(y, y) + R
        cov_yy = (dy.T @ dy) / (n - 1) + obs_cov

        try:
            cov_yy_inv = np.linalg.inv(cov_yy)
        except np.linalg.LinAlgError:
            cov_yy_inv = np.linalg.pinv(cov_yy)

        return cov_xy @ cov_yy_inv

    def _ensemble_update(
        self,
        particles: NDArray[np.float64],
        observation: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        """Perform ensemble Kalman update on particles.

        Parameters
        ----------
        particles : ndarray of shape (N, k_states)
            Predicted particles (after transition).
        observation : ndarray of shape (k_obs,)
            Current observation.
        t : int
            Time step.

        Returns
        -------
        updated : ndarray of shape (N, k_states)
            Updated particles.
        """
        n_particles = particles.shape[0]
        k_obs = observation.shape[0]

        y_pred = self._predict_observations(particles, t)
        obs_cov = self._get_obs_noise_cov(t)
        gain = self._compute_kalman_gain(particles, y_pred, obs_cov)

        # Perturbed observations (Burgers et al, 1998)
        rng = self._get_rng()
        try:
            obs_perturbations = rng.multivariate_normal(np.zeros(k_obs), obs_cov, size=n_particles)
        except np.linalg.LinAlgError:
            obs_cov_jit = obs_cov + np.eye(k_obs) * 1e-8
            obs_perturbations = rng.multivariate_normal(
                np.zeros(k_obs), obs_cov_jit, size=n_particles
            )

        updated = np.empty_like(particles)
        for i in range(n_particles):
            innovation = observation.flatten() + obs_perturbations[i] - y_pred[i]
            updated[i] = particles[i] + gain @ innovation

        return updated

    # ------------------------------------------------------------------
    # BaseParticleFilter interface
    # ------------------------------------------------------------------

    def _propagate(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> ParticleCloud:
        """Propagate particles using transition then ensemble update."""
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
        """Compute incremental log-weights (standard bootstrap fallback)."""
        return self.model.log_observation_likelihood(cloud.particles, y_t, t)

    def filter(
        self,
        endog: NDArray[np.float64],
        mask: NDArray[np.bool_] | None = None,
    ) -> ParticleFilterResults:
        """Run the Ensemble Particle Filter.

        Parameters
        ----------
        endog : ndarray of shape (T,) or (T, k_obs)
            Observation sequence.
        mask : ndarray of shape (T,) or None
            Optional missing-data mask.

        Returns
        -------
        ParticleFilterResults
            Filtering results.
        """
        from particlefilterbox.core.cloud import ParticleCloud as _Cloud

        rng = self._get_rng()

        if endog.ndim == 1:
            endog = endog[:, np.newaxis]

        n_steps, _obs_dim = endog.shape
        k_states = self.model.k_states
        n_particles = self.n_particles

        if mask is None:
            mask = np.any(np.isnan(endog), axis=1)

        cloud = self.initialize(rng)

        filtered_means = np.zeros((n_steps, k_states))
        filtered_covs = np.zeros((n_steps, k_states, k_states))
        log_likelihoods = np.zeros(n_steps)
        ess_history = np.zeros(n_steps)
        resampled = np.zeros(n_steps, dtype=bool)

        for t in range(n_steps):
            y_t = endog[t]
            is_missing = bool(mask[t])

            # Propagate through transition
            predicted = self.model.transition(cloud.particles, t, rng)

            if is_missing:
                new_cloud = _Cloud(n_particles, k_states)
                new_cloud.particles = predicted
                new_cloud.set_log_weights(cloud.log_weights.copy())
                cloud = new_cloud
                filtered_means[t] = cloud.weighted_mean()
                filtered_covs[t] = cloud.weighted_cov()
                ess_history[t] = cloud.ess
                continue

            # Decide ensemble vs standard update
            use_ensemble = rng.random() < self._ensemble_fraction

            if use_ensemble:
                # Ensemble Kalman update
                new_particles = self._ensemble_update(predicted, y_t, t)

                new_cloud = _Cloud(n_particles, k_states)
                new_cloud.particles = new_particles
                new_cloud.set_uniform_weights()
                cloud = new_cloud

                # Approximate log-likelihood from ensemble spread
                y_pred = self._predict_observations(new_particles, t)
                y_mean = np.mean(y_pred, axis=0)
                y_diff = y_t.flatten() - y_mean
                obs_cov = self._get_obs_noise_cov(t)
                y_var = np.var(y_pred, axis=0) + np.diag(obs_cov)
                log_likelihoods[t] = float(
                    -0.5 * np.sum(y_diff**2 / y_var) - 0.5 * np.sum(np.log(2 * np.pi * y_var))
                )
            else:
                # Standard bootstrap update
                new_cloud = _Cloud(n_particles, k_states)
                new_cloud.particles = predicted
                new_cloud.set_log_weights(cloud.log_weights.copy())

                log_w_inc = self.model.log_observation_likelihood(predicted, y_t, t)
                new_lw = new_cloud.log_weights + log_w_inc
                ll_t = float(logsumexp(new_lw) - logsumexp(new_cloud.log_weights))
                new_lw = new_lw - logsumexp(new_lw)
                new_cloud.set_log_weights(new_lw)
                log_likelihoods[t] = ll_t

                new_cloud, did_resample = self._maybe_resample(new_cloud, rng)
                resampled[t] = did_resample
                cloud = new_cloud

            filtered_means[t] = cloud.weighted_mean()
            filtered_covs[t] = cloud.weighted_cov()
            ess_history[t] = cloud.ess

            # Resample if ESS low (ensemble mode)
            if use_ensemble:
                ess = cloud.ess
                if ess < self.config.ess_threshold * n_particles:
                    from particlefilterbox.resampling import systematic_resample

                    indices = systematic_resample(cloud.normalized_weights, rng=rng)
                    cloud.resample(indices)
                    resampled[t] = True

        total_ll = float(np.sum(log_likelihoods))

        return ParticleFilterResults(
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            log_likelihood=total_ll,
            log_likelihoods=log_likelihoods,
            ess_history=ess_history,
            resampled=resampled,
            n_particles=n_particles,
            final_cloud=cloud,
        )
