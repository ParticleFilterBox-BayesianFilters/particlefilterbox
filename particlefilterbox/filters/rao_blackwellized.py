"""Rao-Blackwellized Particle Filter (Marginalized Particle Filter).

Exploits conditional linear-Gaussian substructure in mixed models by
maintaining Kalman filter statistics for the linear component and
particles for the nonlinear component. Uses kalmanbox for Kalman
filter computations.

References
----------
Schon, T., Gustafsson, F. & Nordlund, P.J. (2005). Marginalized Particle
Filters for Mixed Linear/Nonlinear State-Space Models. IEEE Transactions
on Signal Processing, 53(7), 2279-2289.

Doucet, A., Godsill, S. & Andrieu, C. (2000). On sequential Monte Carlo
sampling methods for Bayesian filtering. Statistics and Computing, 10(3).

Chopin, N. & Papaspiliopoulos, O. (2020). An Introduction to Sequential
Monte Carlo. Springer. Cap. 11.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from kalmanbox.core import StateSpaceRepresentation

# CRITICAL: kalmanbox integration
from kalmanbox.filters import KalmanFilter
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


class RaoBlackwellizedPF(BaseParticleFilter):
    """Rao-Blackwellized Particle Filter for mixed linear/nonlinear models.

    Each particle maintains:
    - x_nl: nonlinear state component (sampled via particles)
    - m: Kalman filter mean for linear state (updated analytically)
    - P: Kalman filter covariance for linear state (updated analytically)

    The model must implement:
    - ``has_linear_substate() -> True``
    - ``k_nonlinear: int`` -- dimension of nonlinear state
    - ``k_linear: int`` -- dimension of linear state
    - ``linear_ssm(x_nonlinear) -> StateSpaceRepresentation``
    - ``transition_nonlinear(x_nl, t, rng) -> x_nl_new``

    Parameters
    ----------
    model : ParticleFilterModel
        Mixed linear/nonlinear state-space model.
    config : PFConfig
        Particle filter configuration.

    Raises
    ------
    ValueError
        If model does not support linear substate decomposition.

    Notes
    -----
    CRITICAL: RBPF with 500 particles should match or exceed Bootstrap PF
    with 5000 particles on mixed linear/nonlinear models, due to the
    variance reduction from analytical marginalization.

    Examples
    --------
    >>> from particlefilterbox.filters import RaoBlackwellizedPF
    >>> rbpf = RaoBlackwellizedPF(model=mixed_model, config=config)
    >>> result = rbpf.filter(observations)
    """

    def __init__(
        self,
        model: ParticleFilterModel,
        config: PFConfig,
    ) -> None:
        super().__init__(model=model, config=config)

        # Validate model interface
        if not (
            hasattr(model, "has_linear_substate")
            and callable(model.has_linear_substate)
            and model.has_linear_substate()
        ):
            raise ValueError(
                "RaoBlackwellizedPF requires model.has_linear_substate() == True. "
                "Model must support mixed linear/nonlinear decomposition."
            )

        if not hasattr(model, "k_nonlinear") or not hasattr(model, "k_linear"):
            raise ValueError("Model must define k_nonlinear and k_linear attributes.")

        if not hasattr(model, "linear_ssm") or not callable(model.linear_ssm):
            raise ValueError(
                "Model must implement linear_ssm(x_nonlinear) -> StateSpaceRepresentation."
            )

        if not hasattr(model, "transition_nonlinear") or not callable(model.transition_nonlinear):
            raise ValueError("Model must implement transition_nonlinear(x_nl, t, rng) -> x_nl_new.")

        self._k_nonlinear: int = model.k_nonlinear  # type: ignore[attr-defined]
        self._k_linear: int = model.k_linear  # type: ignore[attr-defined]

        # Kalman filter instance (reused for each particle)
        self._kf = KalmanFilter()

        logger.info(
            "RaoBlackwellizedPF initialized (k_nonlinear=%d, k_linear=%d)",
            self._k_nonlinear,
            self._k_linear,
        )

    # --- Abstract method stubs (not used; filter() is overridden) ---

    def _propagate(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> ParticleCloud:
        """Not used -- RBPF overrides filter() entirely."""
        raise NotImplementedError("RBPF uses its own filter loop.")

    def _compute_weights(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        """Not used -- RBPF overrides filter() entirely."""
        raise NotImplementedError("RBPF uses its own filter loop.")

    # --- RBPF-specific methods ---

    def _init_kalman_states(
        self,
        n_particles: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Initialize Kalman filter states for all particles.

        Parameters
        ----------
        n_particles : int
            Number of particles.

        Returns
        -------
        kalman_means : ndarray of shape (N, k_linear)
            Initial Kalman means (zeros).
        kalman_covs : ndarray of shape (N, k_linear, k_linear)
            Initial Kalman covariances (identity * large_value).
        """
        k_lin = self._k_linear
        kalman_means = np.zeros((n_particles, k_lin), dtype=np.float64)
        kalman_covs = np.tile(
            np.eye(k_lin, dtype=np.float64) * 10.0,
            (n_particles, 1, 1),
        )

        # If model provides initial linear state distribution
        if hasattr(self.model, "initial_linear_mean"):
            m0 = np.asarray(
                self.model.initial_linear_mean()  # type: ignore[attr-defined]
            )
            kalman_means[:] = m0
        if hasattr(self.model, "initial_linear_cov"):
            p0 = np.asarray(
                self.model.initial_linear_cov()  # type: ignore[attr-defined]
            )
            kalman_covs[:] = p0

        return kalman_means, kalman_covs

    def _kalman_update(
        self,
        x_nl: NDArray[np.float64],
        m_prior: NDArray[np.float64],
        p_prior: NDArray[np.float64],
        observation: NDArray[np.float64],
        t: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Run Kalman predict + update for one particle.

        Uses kalmanbox KalmanFilter static methods with SSR matrices
        obtained from the model's ``linear_ssm(x_nl)`` method.

        Parameters
        ----------
        x_nl : ndarray of shape (k_nonlinear,)
            Nonlinear state for this particle.
        m_prior : ndarray of shape (k_linear,)
            Prior Kalman mean.
        p_prior : ndarray of shape (k_linear, k_linear)
            Prior Kalman covariance.
        observation : ndarray of shape (k_obs,)
            Current observation.
        t : int
            Time step.

        Returns
        -------
        m_post : ndarray of shape (k_linear,)
            Posterior Kalman mean.
        p_post : ndarray of shape (k_linear, k_linear)
            Posterior Kalman covariance.
        log_lik : float
            Log marginal likelihood p(y_t | x_nl, m_{t-1}, P_{t-1}).
        """
        # Get linear SSM conditioned on nonlinear state
        ssm: StateSpaceRepresentation = self.model.linear_ssm(x_nl)  # type: ignore[attr-defined]

        # === Kalman Predict (using kalmanbox) ===
        m_pred, p_pred = KalmanFilter.predict_step(
            a=m_prior,
            P=p_prior,
            T=ssm.T,
            R=ssm.R,
            Q=ssm.Q,
            c=ssm.c,
        )

        # === Kalman Update (using kalmanbox) ===
        m_post, p_post, _v, _f, _k, log_lik = KalmanFilter.update_step(
            a_pred=m_pred,
            P_pred=p_pred,
            y=observation.flatten(),
            Z=ssm.Z,
            H=ssm.H,
            d=ssm.d,
        )

        return m_post, p_post, float(log_lik)

    def _resample_with_kalman(
        self,
        particles_nl: NDArray[np.float64],
        kalman_means: NDArray[np.float64],
        kalman_covs: NDArray[np.float64],
        normalized_weights: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Resample particles along with their Kalman states.

        Parameters
        ----------
        particles_nl : ndarray of shape (N, k_nonlinear)
            Nonlinear particles.
        kalman_means : ndarray of shape (N, k_linear)
            Kalman means.
        kalman_covs : ndarray of shape (N, k_linear, k_linear)
            Kalman covariances.
        normalized_weights : ndarray of shape (N,)
            Normalized weights.

        Returns
        -------
        new_particles : ndarray of shape (N, k_nonlinear)
            Resampled nonlinear particles.
        new_means : ndarray of shape (N, k_linear)
            Resampled Kalman means.
        new_covs : ndarray of shape (N, k_linear, k_linear)
            Resampled Kalman covariances.
        """
        rng = self._get_rng()
        indices = systematic_resample(normalized_weights, rng=rng)

        new_particles = particles_nl[indices].copy()
        new_means = kalman_means[indices].copy()
        new_covs = kalman_covs[indices].copy()

        return new_particles, new_means, new_covs

    def filter(
        self,
        endog: NDArray[np.float64],
        mask: NDArray[np.bool_] | None = None,
    ) -> ParticleFilterResults:
        """Run the Rao-Blackwellized Particle Filter.

        Parameters
        ----------
        endog : ndarray of shape (T,) or (T, k_obs)
            Observation sequence.
        mask : ndarray of shape (T,) or None
            Optional boolean mask. True indicates missing data.

        Returns
        -------
        result : ParticleFilterResults
            Filtering results. The filtered_means contain the combined
            nonlinear + linear state estimates.
        """
        rng = self._get_rng()

        observations = np.atleast_2d(endog) if endog.ndim > 1 else endog[:, np.newaxis]
        n_obs = observations.shape[0]
        n_particles = self.n_particles
        k_nl = self._k_nonlinear
        k_lin = self._k_linear
        k_total = k_nl + k_lin

        if mask is None:
            mask = np.any(np.isnan(observations), axis=1)

        # Initialize nonlinear particles
        if hasattr(self.model, "initial_nonlinear_distribution"):
            particles_nl = self.model.initial_nonlinear_distribution(  # type: ignore[attr-defined]
                n_particles, rng
            )
        else:
            particles_nl = self.model.initial_distribution(n_particles, rng)
            if particles_nl.ndim == 1:
                particles_nl = particles_nl.reshape(n_particles, -1)
            particles_nl = particles_nl[:, :k_nl]

        if particles_nl.ndim == 1:
            particles_nl = particles_nl.reshape(n_particles, k_nl)

        # Initialize Kalman states
        kalman_means, kalman_covs = self._init_kalman_states(n_particles)

        # Uniform log-weights
        log_weights = np.full(n_particles, -np.log(n_particles))

        # Storage
        filtered_means = np.empty((n_obs, k_total), dtype=np.float64)
        filtered_covs = np.empty((n_obs, k_total, k_total), dtype=np.float64)
        log_likelihoods = np.empty(n_obs, dtype=np.float64)
        ess_history = np.empty(n_obs, dtype=np.float64)
        resampled = np.zeros(n_obs, dtype=bool)

        for t in range(n_obs):
            obs_t = observations[t]
            is_missing = bool(mask[t])

            # Step 1: Propagate nonlinear particles
            new_particles_nl = self.model.transition_nonlinear(  # type: ignore[attr-defined]
                particles_nl, t, rng
            )
            if new_particles_nl.ndim == 1:
                new_particles_nl = new_particles_nl.reshape(n_particles, k_nl)

            if is_missing:
                # Skip weight update for missing observations
                filtered_means[t] = 0.0
                filtered_covs[t] = 0.0
                log_likelihoods[t] = 0.0
                ess_history[t] = float(n_particles)
                particles_nl = new_particles_nl
                continue

            # Step 2: Kalman predict + update for each particle
            new_kalman_means = np.empty_like(kalman_means)
            new_kalman_covs = np.empty_like(kalman_covs)
            log_liks = np.empty(n_particles, dtype=np.float64)

            for i in range(n_particles):
                m_post, p_post, log_lik = self._kalman_update(
                    x_nl=new_particles_nl[i],
                    m_prior=kalman_means[i],
                    p_prior=kalman_covs[i],
                    observation=obs_t,
                    t=t,
                )
                new_kalman_means[i] = m_post
                new_kalman_covs[i] = p_post
                log_liks[i] = log_lik

            # Step 3: Update weights
            log_weights = log_weights + log_liks

            # Log-likelihood increment
            log_likelihoods[t] = log_sum_exp(log_weights) - np.log(n_particles)

            # Normalize
            normalized_weights = normalize_log_weights(log_weights)

            # ESS
            ess = 1.0 / np.sum(normalized_weights**2)

            # Combine nonlinear + linear for state estimate
            combined_particles = np.hstack([new_particles_nl, new_kalman_means])

            # Weighted mean
            mean_nl = np.average(new_particles_nl, weights=normalized_weights, axis=0)
            mean_lin = np.average(new_kalman_means, weights=normalized_weights, axis=0)
            mean = np.concatenate([mean_nl, mean_lin])

            # Covariance (approximate)
            diff = combined_particles - mean
            cov = np.einsum("i,ij,ik->jk", normalized_weights, diff, diff)
            # Add average Kalman covariance for linear part
            avg_p = np.average(
                new_kalman_covs,
                weights=normalized_weights,
                axis=0,
            )
            cov[k_nl:, k_nl:] += avg_p

            # Store
            filtered_means[t] = mean
            filtered_covs[t] = cov
            ess_history[t] = ess

            # Update state
            particles_nl = new_particles_nl
            kalman_means = new_kalman_means
            kalman_covs = new_kalman_covs

            # Resampling (carrying Kalman states)
            if ess < self.config.ess_threshold * n_particles:
                particles_nl, kalman_means, kalman_covs = self._resample_with_kalman(
                    particles_nl,
                    kalman_means,
                    kalman_covs,
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
