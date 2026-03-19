"""Unscented Particle Filter (UPF).

Uses the Unscented Kalman Filter as a proposal distribution for each
particle, yielding more informed proposals than the bootstrap prior.

References
----------
van der Merwe, R., Doucet, A., de Freitas, N. & Wan, E. (2001).
The Unscented Particle Filter. Advances in Neural Information
Processing Systems, 13.

Chopin, N. & Papaspiliopoulos, O. (2020). An Introduction to Sequential
Monte Carlo. Springer. Cap. 12.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# CRITICAL: kalmanbox integration
from kalmanbox.filters import UnscentedKalmanFilter
from numpy.typing import NDArray

from particlefilterbox._logging import get_logger
from particlefilterbox.filters.base import BaseParticleFilter, ParticleFilterResults
from particlefilterbox.resampling import systematic_resample
from particlefilterbox.utils.log_ops import log_sum_exp, normalize_log_weights

if TYPE_CHECKING:
    from particlefilterbox.core.cloud import ParticleCloud
    from particlefilterbox.core.config import PFConfig
    from particlefilterbox.core.model import ParticleFilterModel

logger = get_logger(__name__)


class UnscentedPF(BaseParticleFilter):
    """Unscented Particle Filter using UKF as proposal.

    Each particle carries UKF statistics (mean, covariance) that are
    used to generate proposals informed by both the dynamics and the
    current observation.

    Weight update:
        w_t^(i) = p(y_t | x_t^(i)) * p(x_t^(i) | x_{t-1}^(i)) / q(x_t^(i))

    where q(x_t^(i)) = N(x_t^(i); m_UKF^(i), P_UKF^(i)).

    Parameters
    ----------
    model : ParticleFilterModel
        State-space model. Must provide:
        - ``transition(particles, t, rng)``
        - ``log_observation_likelihood(particles, y_t, t)``
        - ``transition_function(x, t) -> x_pred`` (deterministic transition)
        - ``observation_function(x, t) -> y_pred`` (deterministic observation)
        - ``Q(t)`` or ``process_noise_cov(t)`` -- process noise covariance
        - ``R(t)`` or ``observation_noise_cov(t)`` -- observation noise covariance
    config : PFConfig
        Particle filter configuration.
    alpha : float
        UKF scaling parameter alpha (default 1e-3).
    beta : float
        UKF parameter beta (default 2.0 for Gaussian).
    kappa : float
        UKF parameter kappa (default 0.0).

    Examples
    --------
    >>> from particlefilterbox.filters import UnscentedPF
    >>> upf = UnscentedPF(model=my_model, config=config)
    >>> result = upf.filter(observations)
    """

    def __init__(
        self,
        model: ParticleFilterModel,
        config: PFConfig,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        super().__init__(model=model, config=config)

        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

        # UKF instance from kalmanbox
        self._ukf = UnscentedKalmanFilter(
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )

        # Extract model functions
        self._has_transition_fn = hasattr(model, "transition_function") and callable(
            model.transition_function
        )
        self._has_observation_fn = hasattr(model, "observation_function") and callable(
            model.observation_function
        )

        logger.info(
            "UnscentedPF initialized (alpha=%.4f, beta=%.1f, kappa=%.1f)",
            alpha,
            beta,
            kappa,
        )

    # --- Abstract method stubs (not used; filter() is overridden) ---

    def _propagate(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> ParticleCloud:
        """Not used -- UPF overrides filter() entirely."""
        raise NotImplementedError("UPF uses its own filter loop.")

    def _compute_weights(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        """Not used -- UPF overrides filter() entirely."""
        raise NotImplementedError("UPF uses its own filter loop.")

    # --- UPF-specific methods ---

    def _get_process_cov(self, t: int) -> NDArray[np.float64]:
        """Get process noise covariance at time t."""
        if hasattr(self.model, "Q") and callable(self.model.Q):
            return np.atleast_2d(self.model.Q(t))  # type: ignore[attr-defined]
        elif hasattr(self.model, "process_noise_cov"):
            return np.atleast_2d(self.model.process_noise_cov(t))  # type: ignore[attr-defined]
        elif hasattr(self.model, "sigma_x"):
            k = self.model.k_states
            return np.eye(k) * self.model.sigma_x**2  # type: ignore[attr-defined]
        else:
            return np.eye(self.model.k_states) * 0.1

    def _get_obs_cov(self, t: int) -> NDArray[np.float64]:
        """Get observation noise covariance at time t."""
        if hasattr(self.model, "R_obs") and callable(self.model.R_obs):
            return np.atleast_2d(self.model.R_obs(t))  # type: ignore[attr-defined]
        elif hasattr(self.model, "observation_noise_cov"):
            return np.atleast_2d(self.model.observation_noise_cov(t))  # type: ignore[attr-defined]
        elif hasattr(self.model, "sigma_y"):
            k = self.model.k_obs if hasattr(self.model, "k_obs") else 1
            return np.eye(k) * self.model.sigma_y**2  # type: ignore[attr-defined]
        else:
            k = self.model.k_obs if hasattr(self.model, "k_obs") else 1
            return np.eye(k) * 0.1

    def _transition_fn(self, x: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Deterministic transition function."""
        if self._has_transition_fn:
            return np.atleast_1d(
                self.model.transition_function(x, t)  # type: ignore[attr-defined]
            )
        elif hasattr(self.model, "transition_mean"):
            return np.atleast_1d(
                self.model.transition_mean(x.reshape(1, -1), t).flatten()  # type: ignore[attr-defined]
            )
        else:
            rng_det = np.random.default_rng(0)
            result = self.model.transition(x.reshape(1, -1), t, rng_det)
            return result.flatten()

    def _observation_fn(self, x: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Deterministic observation function."""
        if self._has_observation_fn:
            return np.atleast_1d(
                self.model.observation_function(x, t)  # type: ignore[attr-defined]
            )
        elif hasattr(self.model, "observation_mean"):
            return np.atleast_1d(
                self.model.observation_mean(x.reshape(1, -1), t).flatten()  # type: ignore[attr-defined]
            )
        else:
            k_obs = self.model.k_obs if hasattr(self.model, "k_obs") else x.shape[0]
            return x[:k_obs]

    def _ukf_predict_update(
        self,
        m_prior: NDArray[np.float64],
        p_prior: NDArray[np.float64],
        observation: NDArray[np.float64],
        t: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run UKF predict + update for one particle.

        Parameters
        ----------
        m_prior : ndarray of shape (k_states,)
            Prior UKF mean (from previous particle position).
        p_prior : ndarray of shape (k_states, k_states)
            Prior UKF covariance.
        observation : ndarray of shape (k_obs,)
            Current observation.
        t : int
            Time step.

        Returns
        -------
        m_post : ndarray of shape (k_states,)
            Posterior UKF mean.
        P_post : ndarray of shape (k_states, k_states)
            Posterior UKF covariance.
        """
        k = self.model.k_states
        q_cov = self._get_process_cov(t)
        r_cov = self._get_obs_cov(t)

        # === UKF Predict ===
        lam = self._alpha**2 * (k + self._kappa) - k
        n_sigma = 2 * k + 1

        # Weights
        w_m = np.zeros(n_sigma)
        w_c = np.zeros(n_sigma)
        w_m[0] = lam / (k + lam)
        w_c[0] = lam / (k + lam) + (1 - self._alpha**2 + self._beta)
        for i in range(1, n_sigma):
            w_m[i] = 1.0 / (2 * (k + lam))
            w_c[i] = 1.0 / (2 * (k + lam))

        # Sigma points
        try:
            sqrt_p = np.linalg.cholesky((k + lam) * p_prior)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(p_prior)
            eigvals = np.maximum(eigvals, 1e-10)
            sqrt_p = eigvecs @ np.diag(np.sqrt((k + lam) * eigvals))

        sigma_pts = np.zeros((n_sigma, k))
        sigma_pts[0] = m_prior
        for i in range(k):
            sigma_pts[i + 1] = m_prior + sqrt_p[:, i]
            sigma_pts[k + i + 1] = m_prior - sqrt_p[:, i]

        # Transform through transition
        sigma_pred = np.zeros_like(sigma_pts)
        for i in range(n_sigma):
            sigma_pred[i] = self._transition_fn(sigma_pts[i], t)

        # Predicted mean and covariance
        m_pred = np.zeros(k)
        for i in range(n_sigma):
            m_pred += w_m[i] * sigma_pred[i]

        p_pred = q_cov.copy()
        for i in range(n_sigma):
            diff = sigma_pred[i] - m_pred
            p_pred += w_c[i] * np.outer(diff, diff)

        # === UKF Update ===
        try:
            sqrt_p_pred = np.linalg.cholesky((k + lam) * p_pred)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(p_pred)
            eigvals = np.maximum(eigvals, 1e-10)
            sqrt_p_pred = eigvecs @ np.diag(np.sqrt((k + lam) * eigvals))

        sigma_pts_upd = np.zeros((n_sigma, k))
        sigma_pts_upd[0] = m_pred
        for i in range(k):
            sigma_pts_upd[i + 1] = m_pred + sqrt_p_pred[:, i]
            sigma_pts_upd[k + i + 1] = m_pred - sqrt_p_pred[:, i]

        # Transform through observation
        k_obs = observation.shape[0]
        sigma_obs = np.zeros((n_sigma, k_obs))
        for i in range(n_sigma):
            sigma_obs[i] = self._observation_fn(sigma_pts_upd[i], t)

        # Predicted observation
        y_pred = np.zeros(k_obs)
        for i in range(n_sigma):
            y_pred += w_m[i] * sigma_obs[i]

        # Innovation covariance
        s_inn = r_cov.copy()
        for i in range(n_sigma):
            diff_y = sigma_obs[i] - y_pred
            s_inn += w_c[i] * np.outer(diff_y, diff_y)

        # Cross-covariance
        pxy = np.zeros((k, k_obs))
        for i in range(n_sigma):
            diff_x = sigma_pts_upd[i] - m_pred
            diff_y = sigma_obs[i] - y_pred
            pxy += w_c[i] * np.outer(diff_x, diff_y)

        # Kalman gain
        try:
            s_inv = np.linalg.inv(s_inn)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s_inn)

        kg = pxy @ s_inv

        # Update
        innovation = observation.flatten() - y_pred
        m_post = m_pred + kg @ innovation
        p_post = p_pred - kg @ s_inn @ kg.T

        # Ensure symmetry and positive definiteness
        p_post = 0.5 * (p_post + p_post.T)
        eigvals_post = np.linalg.eigvalsh(p_post)
        if np.any(eigvals_post < 0):
            p_post += np.eye(k) * (abs(eigvals_post.min()) + 1e-8)

        return m_post, p_post

    def _log_transition_density(
        self,
        x_new: NDArray[np.float64],
        x_old: NDArray[np.float64],
        t: int,
    ) -> float:
        """Compute log p(x_new | x_old) under the transition model."""
        q_cov = self._get_process_cov(t)
        mu = self._transition_fn(x_old, t)
        diff = x_new - mu

        k = len(diff)
        sign, log_det = np.linalg.slogdet(q_cov)
        if sign <= 0:
            return -1e10

        try:
            q_inv = np.linalg.inv(q_cov)
        except np.linalg.LinAlgError:
            q_inv = np.linalg.pinv(q_cov)

        log_p = float(-0.5 * k * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * diff @ q_inv @ diff)
        return log_p

    def _log_proposal_density(
        self,
        x: NDArray[np.float64],
        m_ukf: NDArray[np.float64],
        p_ukf: NDArray[np.float64],
    ) -> float:
        """Compute log q(x) = log N(x; m_UKF, P_UKF)."""
        k = len(x)
        diff = x - m_ukf

        sign, log_det = np.linalg.slogdet(p_ukf)
        if sign <= 0:
            return -1e10

        try:
            p_inv = np.linalg.inv(p_ukf)
        except np.linalg.LinAlgError:
            p_inv = np.linalg.pinv(p_ukf)

        log_q = float(-0.5 * k * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * diff @ p_inv @ diff)
        return log_q

    def _resample_with_ukf(
        self,
        particles: NDArray[np.float64],
        ukf_means: NDArray[np.float64],
        ukf_covs: NDArray[np.float64],
        normalized_weights: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Resample particles along with their UKF states.

        Parameters
        ----------
        particles : ndarray of shape (N, k_states)
        ukf_means : ndarray of shape (N, k_states)
        ukf_covs : ndarray of shape (N, k_states, k_states)
        normalized_weights : ndarray of shape (N,)

        Returns
        -------
        new_particles, new_means, new_covs
        """
        rng = self._get_rng()
        indices = systematic_resample(normalized_weights, rng=rng)

        new_particles = particles[indices].copy()
        new_means = ukf_means[indices].copy()
        new_covs = ukf_covs[indices].copy()

        return new_particles, new_means, new_covs

    def filter(
        self,
        endog: NDArray[np.float64],
        mask: NDArray[np.bool_] | None = None,
    ) -> ParticleFilterResults:
        """Run the Unscented Particle Filter.

        Parameters
        ----------
        endog : ndarray of shape (T,) or (T, k_obs)
            Observation sequence.
        mask : ndarray of shape (T,) or None
            Optional boolean mask. True indicates missing data.

        Returns
        -------
        result : ParticleFilterResults
            Filtering results.
        """
        rng = self._get_rng()

        observations = np.atleast_2d(endog) if endog.ndim > 1 else endog[:, np.newaxis]
        n_obs = observations.shape[0]
        n_particles = self.n_particles
        k_states = self.model.k_states

        if mask is None:
            mask = np.any(np.isnan(observations), axis=1)

        # Initialize particles
        particles = self.model.initial_distribution(n_particles, rng)
        if particles.ndim == 1:
            particles = particles.reshape(n_particles, k_states)

        # Initialize UKF states for each particle
        ukf_means = particles.copy()  # (N, k_states)
        ukf_covs = np.tile(
            np.eye(k_states, dtype=np.float64) * 1.0,
            (n_particles, 1, 1),
        )  # (N, k_states, k_states)

        log_weights = np.full(n_particles, -np.log(n_particles))

        # Storage
        filtered_means = np.empty((n_obs, k_states), dtype=np.float64)
        filtered_covs = np.empty((n_obs, k_states, k_states), dtype=np.float64)
        log_likelihoods = np.empty(n_obs, dtype=np.float64)
        ess_history = np.empty(n_obs, dtype=np.float64)
        resampled = np.zeros(n_obs, dtype=bool)

        for t in range(n_obs):
            obs_t = observations[t]
            is_missing = bool(mask[t])

            if is_missing:
                # Skip weight update for missing observations
                new_particles = self.model.transition(particles, t, rng)
                if new_particles.ndim == 1:
                    new_particles = new_particles.reshape(n_particles, k_states)
                particles = new_particles
                filtered_means[t] = 0.0
                filtered_covs[t] = 0.0
                log_likelihoods[t] = 0.0
                ess_history[t] = float(n_particles)
                continue

            new_particles = np.empty_like(particles)
            new_ukf_means = np.empty_like(ukf_means)
            new_ukf_covs = np.empty_like(ukf_covs)
            log_weight_updates = np.empty(n_particles, dtype=np.float64)

            for i in range(n_particles):
                # Step 1: UKF predict + update to get proposal
                m_ukf, p_ukf = self._ukf_predict_update(
                    m_prior=ukf_means[i],
                    p_prior=ukf_covs[i],
                    observation=obs_t,
                    t=t,
                )

                # Step 2: Sample from UKF proposal
                try:
                    x_new = rng.multivariate_normal(m_ukf, p_ukf)
                except np.linalg.LinAlgError:
                    p_jittered = p_ukf + np.eye(k_states) * 1e-6
                    x_new = rng.multivariate_normal(m_ukf, p_jittered)

                new_particles[i] = x_new
                new_ukf_means[i] = m_ukf
                new_ukf_covs[i] = p_ukf

                # Step 3: Compute importance weight
                # w = p(y|x_new) * p(x_new|x_old) / q(x_new; m_ukf, P_ukf)
                log_lik = float(
                    self.model.log_observation_likelihood(x_new.reshape(1, -1), obs_t, t)[0]
                )
                log_trans = self._log_transition_density(x_new, particles[i], t)
                log_prop = self._log_proposal_density(x_new, m_ukf, p_ukf)

                log_weight_updates[i] = log_lik + log_trans - log_prop

            # Update weights
            log_weights = log_weights + log_weight_updates

            # Log-likelihood increment
            log_likelihoods[t] = log_sum_exp(log_weights) - np.log(n_particles)

            # Normalize
            normalized_weights = normalize_log_weights(log_weights)

            # ESS
            ess = 1.0 / np.sum(normalized_weights**2)

            # Compute mean and covariance
            mean = np.average(new_particles, weights=normalized_weights, axis=0)
            diff = new_particles - mean
            cov = np.einsum("i,ij,ik->jk", normalized_weights, diff, diff)

            # Store
            filtered_means[t] = mean
            filtered_covs[t] = cov
            ess_history[t] = ess

            # Update state
            particles = new_particles
            ukf_means = new_ukf_means
            ukf_covs = new_ukf_covs

            # Resampling (carrying UKF states)
            if ess < self.config.ess_threshold * n_particles:
                particles, ukf_means, ukf_covs = self._resample_with_ukf(
                    particles,
                    ukf_means,
                    ukf_covs,
                    normalized_weights,
                )
                log_weights = np.full(n_particles, -np.log(n_particles))
                resampled[t] = True

            logger.debug("t=%d: ESS=%.1f, loglik=%.4f", t, ess, log_likelihoods[t])

        total_ll = float(np.sum(log_likelihoods))

        return ParticleFilterResults(
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            log_likelihood=total_ll,
            log_likelihoods=log_likelihoods,
            ess_history=ess_history,
            resampled=resampled,
            n_particles=n_particles,
            final_cloud=None,
        )
