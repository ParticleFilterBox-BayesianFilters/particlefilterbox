"""Guided Particle Filter.

Uses observation information to guide the proposal distribution,
improving particle placement in regions of high likelihood.

Supports three guidance modes:
- linearization: EKF-like update via local linearization
- mode_finding: find posterior mode via optimization
- nudging: simple shift toward observation

References
----------
Doucet, A., Godsill, S. & Andrieu, C. (2000). On sequential Monte Carlo
sampling methods for Bayesian filtering. Statistics and Computing, 10(3).

Chopin, N. & Papaspiliopoulos, O. (2020). An Introduction to Sequential
Monte Carlo. Springer. Cap. 12.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import logsumexp

from particlefilterbox._logging import get_logger
from particlefilterbox.filters.base import BaseParticleFilter, ParticleFilterResults

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from particlefilterbox.core.cloud import ParticleCloud
    from particlefilterbox.core.config import PFConfig
    from particlefilterbox.core.model import ParticleFilterModel

logger = get_logger(__name__)


class GuidedPF(BaseParticleFilter):
    """Guided Particle Filter with observation-informed proposals.

    Modifies the bootstrap proposal to account for the current observation,
    moving particles toward regions of high likelihood.

    Parameters
    ----------
    model : ParticleFilterModel
        State-space model.
    config : PFConfig
        Particle filter configuration.
    guide_mode : str
        Guidance mode: ``'linearization'``, ``'mode_finding'``, or
        ``'nudging'``.
    nudge_factor : float
        Strength of nudging toward observation (for ``'nudging'`` mode).
        0.0 = no nudge (bootstrap), 1.0 = full nudge. Default 0.5.
    linearization_noise_scale : float
        Scale factor for proposal noise in linearization mode. Default 1.0.

    Examples
    --------
    >>> gpf = GuidedPF(model=model, config=config, guide_mode='linearization')
    >>> result = gpf.filter(observations)
    """

    SUPPORTED_MODES = ("linearization", "mode_finding", "nudging")

    def __init__(
        self,
        model: ParticleFilterModel,
        config: PFConfig,
        guide_mode: Literal["linearization", "mode_finding", "nudging"] = "linearization",
        nudge_factor: float = 0.5,
        linearization_noise_scale: float = 1.0,
    ) -> None:
        super().__init__(model=model, config=config)

        if guide_mode not in self.SUPPORTED_MODES:
            msg = f"Unsupported guide_mode '{guide_mode}'. Choose from {self.SUPPORTED_MODES}"
            raise ValueError(msg)

        self._guide_mode = guide_mode
        self._nudge_factor = nudge_factor
        self._lin_noise_scale = linearization_noise_scale

        logger.info(
            "GuidedPF initialized (mode=%s, nudge=%.2f)",
            guide_mode,
            nudge_factor,
        )

    @property
    def guide_mode(self) -> str:
        """Current guidance mode."""
        return self._guide_mode

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _get_process_cov(self, t: int) -> NDArray[np.float64]:
        """Get process noise covariance."""
        if hasattr(self.model, "Q"):
            q_attr = self.model.Q  # type: ignore[attr-defined]
            if callable(q_attr):
                return np.atleast_2d(q_attr(t))
        if hasattr(self.model, "process_noise_cov"):
            return np.atleast_2d(
                self.model.process_noise_cov(t)  # type: ignore[attr-defined]
            )
        if hasattr(self.model, "sigma_x"):
            k = self.model.k_states
            return np.eye(k) * self.model.sigma_x**2  # type: ignore[attr-defined]
        return np.eye(self.model.k_states) * 0.1

    def _get_obs_cov(self, t: int) -> NDArray[np.float64]:
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
            k = self.model.k_obs
            return np.eye(k) * self.model.sigma_y**2  # type: ignore[attr-defined]
        return np.eye(self.model.k_obs) * 1.0

    def _transition_fn(self, x: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Deterministic transition (mean)."""
        if hasattr(self.model, "transition_function"):
            return np.atleast_1d(
                self.model.transition_function(x, t)  # type: ignore[attr-defined]
            )
        if hasattr(self.model, "transition_mean"):
            return self.model.transition_mean(  # type: ignore[attr-defined]
                x.reshape(1, -1), t
            ).flatten()
        # Fallback: use transition with a fixed rng (deterministic seed)
        rng_det = np.random.default_rng(0)
        return self.model.transition(x.reshape(1, -1), t, rng_det).flatten()

    def _observation_fn(self, x: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Deterministic observation function."""
        if hasattr(self.model, "observation_function"):
            return np.atleast_1d(
                self.model.observation_function(x, t)  # type: ignore[attr-defined]
            )
        if hasattr(self.model, "observation_mean"):
            return self.model.observation_mean(  # type: ignore[attr-defined]
                x.reshape(1, -1), t
            ).flatten()
        k_obs = self.model.k_obs
        return x[:k_obs]

    def _jacobian_h(self, x: NDArray[np.float64], t: int, eps: float = 1e-5) -> NDArray[np.float64]:
        """Numerical Jacobian of observation function.

        Parameters
        ----------
        x : ndarray of shape (k_states,)
            State at which to evaluate Jacobian.
        t : int
            Time step.
        eps : float
            Finite difference step size.

        Returns
        -------
        jac : ndarray of shape (k_obs, k_states)
            Jacobian matrix.
        """
        h0 = self._observation_fn(x, t)
        k_obs = h0.shape[0]
        k_states = x.shape[0]

        jac = np.empty((k_obs, k_states), dtype=np.float64)
        for j in range(k_states):
            x_plus = x.copy()
            x_plus[j] += eps
            jac[:, j] = (self._observation_fn(x_plus, t) - h0) / eps

        return jac

    # ------------------------------------------------------------------
    # Guidance modes
    # ------------------------------------------------------------------

    def _guide_linearization(
        self,
        x_pred: NDArray[np.float64],
        observation: NDArray[np.float64],
        t: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Guide proposal via local linearization (EKF-like).

        Returns
        -------
        mean : ndarray of shape (k_states,)
        cov : ndarray of shape (k_states, k_states)
        """
        q_cov = self._get_process_cov(t)
        r_cov = self._get_obs_cov(t)

        jac = self._jacobian_h(x_pred, t)
        y_pred = self._observation_fn(x_pred, t)

        s_mat = jac @ q_cov @ jac.T + r_cov
        try:
            s_inv = np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s_mat)

        gain = q_cov @ jac.T @ s_inv
        innovation = observation.flatten() - y_pred

        mean = x_pred + gain @ innovation
        cov = (np.eye(self.model.k_states) - gain @ jac) @ q_cov
        cov *= self._lin_noise_scale**2

        # Ensure positive definite
        cov = 0.5 * (cov + cov.T)
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals < 0):
            cov += np.eye(self.model.k_states) * (abs(float(eigvals.min())) + 1e-8)

        return mean, cov

    def _guide_mode_finding(
        self,
        x_pred: NDArray[np.float64],
        observation: NDArray[np.float64],
        t: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Guide proposal via posterior mode finding.

        Returns
        -------
        mean : ndarray of shape (k_states,)
        cov : ndarray of shape (k_states, k_states)
        """
        q_cov = self._get_process_cov(t)
        r_cov = self._get_obs_cov(t)

        try:
            q_inv = np.linalg.inv(q_cov)
        except np.linalg.LinAlgError:
            q_inv = np.linalg.pinv(q_cov)

        try:
            r_inv = np.linalg.inv(r_cov)
        except np.linalg.LinAlgError:
            r_inv = np.linalg.pinv(r_cov)

        def neg_log_posterior(x: NDArray[np.float64]) -> float:
            diff_prior = x - x_pred
            prior_term = 0.5 * float(diff_prior @ q_inv @ diff_prior)
            y_pred = self._observation_fn(x, t)
            diff_lik = observation.flatten() - y_pred
            lik_term = 0.5 * float(diff_lik @ r_inv @ diff_lik)
            return prior_term + lik_term

        result = minimize(
            neg_log_posterior,
            x_pred,
            method="L-BFGS-B",
            options={"maxiter": 50},
        )
        mode: NDArray[np.float64] = result.x

        # Laplace approximation
        jac = self._jacobian_h(mode, t)
        precision = q_inv + jac.T @ r_inv @ jac
        try:
            cov = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(precision)

        cov = 0.5 * (cov + cov.T)
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals < 0):
            cov += np.eye(self.model.k_states) * (abs(float(eigvals.min())) + 1e-8)

        return mode, cov

    def _guide_nudging(
        self,
        x_pred: NDArray[np.float64],
        observation: NDArray[np.float64],
        t: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Guide proposal via simple nudging toward observation.

        Returns
        -------
        mean : ndarray of shape (k_states,)
        cov : ndarray of shape (k_states, k_states)
        """
        q_cov = self._get_process_cov(t)

        y_pred = self._observation_fn(x_pred, t)
        innovation = observation.flatten() - y_pred

        jac = self._jacobian_h(x_pred, t)
        try:
            jac_pinv = np.linalg.pinv(jac)
        except np.linalg.LinAlgError:
            jac_pinv = np.zeros((self.model.k_states, observation.shape[0]))

        nudge = self._nudge_factor * jac_pinv @ innovation
        mean = x_pred + nudge

        return mean, q_cov

    def _guide_proposal(
        self,
        x_pred: NDArray[np.float64],
        observation: NDArray[np.float64],
        t: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Generate guided proposal parameters.

        Parameters
        ----------
        x_pred : ndarray of shape (k_states,)
            Predicted state from transition.
        observation : ndarray of shape (k_obs,)
            Current observation.
        t : int
            Time step.

        Returns
        -------
        mean : ndarray of shape (k_states,)
        cov : ndarray of shape (k_states, k_states)
        """
        if self._guide_mode == "linearization":
            return self._guide_linearization(x_pred, observation, t)
        if self._guide_mode == "mode_finding":
            return self._guide_mode_finding(x_pred, observation, t)
        if self._guide_mode == "nudging":
            return self._guide_nudging(x_pred, observation, t)
        msg = f"Unknown guide mode: {self._guide_mode}"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Density helpers
    # ------------------------------------------------------------------

    def _log_proposal_density(
        self,
        x: NDArray[np.float64],
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> float:
        """Log density of guided proposal N(x; mean, cov)."""
        k = len(x)
        diff = x - mean
        sign, log_det = np.linalg.slogdet(cov)
        if sign <= 0:
            return -1e10
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        return float(-0.5 * k * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * diff @ cov_inv @ diff)

    def _log_transition_density(
        self,
        x_new: NDArray[np.float64],
        x_old: NDArray[np.float64],
        t: int,
    ) -> float:
        """Log transition density p(x_new | x_old)."""
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
        return float(-0.5 * k * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * diff @ q_inv @ diff)

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
        """Propagate particles using guided proposal."""
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
        """Compute incremental log-weights (bootstrap fallback)."""
        return self.model.log_observation_likelihood(cloud.particles, y_t, t)

    def filter(
        self,
        endog: NDArray[np.float64],
        mask: NDArray[np.bool_] | None = None,
    ) -> ParticleFilterResults:
        """Run the Guided Particle Filter.

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
        resampled_arr = np.zeros(n_steps, dtype=bool)

        for t in range(n_steps):
            y_t = endog[t]
            is_missing = bool(mask[t])

            if is_missing:
                predicted = self.model.transition(cloud.particles, t, rng)
                new_cloud = _Cloud(n_particles, k_states)
                new_cloud.particles = predicted
                new_cloud.set_log_weights(cloud.log_weights.copy())
                cloud = new_cloud
                filtered_means[t] = cloud.weighted_mean()
                filtered_covs[t] = cloud.weighted_cov()
                ess_history[t] = cloud.ess
                continue

            # Guided proposal step
            new_particles = np.empty((n_particles, k_states), dtype=np.float64)
            log_weight_updates = np.empty(n_particles, dtype=np.float64)
            old_particles = cloud.particles

            for i in range(n_particles):
                x_pred = self._transition_fn(old_particles[i], t)
                prop_mean, prop_cov = self._guide_proposal(x_pred, y_t, t)

                try:
                    x_new = rng.multivariate_normal(prop_mean, prop_cov)
                except np.linalg.LinAlgError:
                    prop_cov_jit = prop_cov + np.eye(k_states) * 1e-6
                    x_new = rng.multivariate_normal(prop_mean, prop_cov_jit)

                new_particles[i] = x_new

                # Importance weight: w = p(y|x) * p(x|x_old) / q(x)
                log_lik = float(
                    self.model.log_observation_likelihood(x_new.reshape(1, -1), y_t, t)[0]
                )
                log_trans = self._log_transition_density(x_new, old_particles[i], t)
                log_prop = self._log_proposal_density(x_new, prop_mean, prop_cov)

                log_weight_updates[i] = log_lik + log_trans - log_prop

            new_log_weights = cloud.log_weights + log_weight_updates

            # Log-likelihood increment
            ll_t = float(logsumexp(new_log_weights) - logsumexp(cloud.log_weights))
            new_log_weights_norm = new_log_weights - logsumexp(new_log_weights)

            new_cloud = _Cloud(n_particles, k_states)
            new_cloud.particles = new_particles
            new_cloud.set_log_weights(new_log_weights_norm)
            cloud = new_cloud

            log_likelihoods[t] = ll_t
            filtered_means[t] = cloud.weighted_mean()
            filtered_covs[t] = cloud.weighted_cov()
            ess_history[t] = cloud.ess

            # Resample if ESS low
            cloud, did_resample = self._maybe_resample(cloud, rng)
            resampled_arr[t] = did_resample

        total_ll = float(np.sum(log_likelihoods))

        return ParticleFilterResults(
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            log_likelihood=total_ll,
            log_likelihoods=log_likelihoods,
            ess_history=ess_history,
            resampled=resampled_arr,
            n_particles=n_particles,
            final_cloud=cloud,
        )
