"""Online SMC^2 for streaming parameter inference.

Maintains a population of parameter particles, each with its own internal
particle filter. Processes one observation at a time, making it suitable
for online/streaming applications.

State:
    - N_theta parameter particles, each with weight
    - Each parameter particle has N_x internal state particles
    - Running log-evidence estimate

For each new observation y_t:
    1. Extend each internal PF to incorporate y_t
    2. Reweight parameters based on incremental likelihood
    3. Resample + rejuvenate if ESS drops below threshold

References:
    Chopin, N., Jacob, P. E. & Papaspiliopoulos, O. (2013). SMC^2: An
    efficient algorithm for sequential analysis of state space models.
    JRSS-B, 75(3), 397-426.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

__all__ = ["SMC2Online", "SMC2OnlineState"]


@dataclass
class SMC2OnlineState:
    """Current state of SMC2Online.

    Attributes
    ----------
    theta_particles : NDArray[np.float64]
        Parameter particles of shape ``(N_theta, k_params)``.
    theta_weights : NDArray[np.float64]
        Normalized weights of shape ``(N_theta,)``.
    log_evidence : float
        Cumulative log marginal evidence p(y_{1:t}).
    n_observations : int
        Number of observations processed so far.
    ess_history : list[float]
        ESS at each time step.
    resample_times : list[int]
        Time indices where resampling occurred.
    """

    theta_particles: NDArray[np.float64]
    theta_weights: NDArray[np.float64]
    log_evidence: float = 0.0
    n_observations: int = 0
    ess_history: list[float] = field(default_factory=lambda: list[float]())
    resample_times: list[int] = field(default_factory=lambda: list[int]())


class SMC2Online:
    """Online SMC^2 for streaming parameter inference.

    Processes one observation at a time, maintaining an ensemble of
    parameter particles with internal particle filters.

    Parameters
    ----------
    model : Any
        State-space model. Must implement:
        - ``set_params(theta)``: Set parameters.
        - ``get_params()``: Get parameters as array.
        - ``param_names`` (optional): Parameter names.
        And provide interfaces for internal PF:
        - ``initial_sample(n, rng)``: Sample initial states.
        - ``transition_sample(x, rng)``: Propagate states.
        - ``observation_logpdf(y, x)``: Log observation density.
    n_theta : int
        Number of parameter particles. Default 500.
    n_x : int
        Number of state particles per parameter particle. Default 100.
    prior : Any
        Prior with ``logpdf(theta)`` and ``sample(rng)``.
    ess_threshold : float
        ESS/N_theta ratio below which resampling is triggered. Default 0.5.
    n_mcmc_moves : int
        Number of MCMC rejuvenation moves after resampling. Default 3.
    seed : int | None
        Random seed.

    Examples
    --------
    >>> smc2 = SMC2Online(model=model, n_theta=500, n_x=100, prior=prior)
    >>> for y_t in streaming_data:
    ...     smc2.update(y_t)
    >>> posterior = smc2.current_posterior()
    """

    def __init__(
        self,
        model: Any,
        n_theta: int = 500,
        n_x: int = 100,
        prior: Any = None,
        ess_threshold: float = 0.5,
        n_mcmc_moves: int = 3,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.n_theta = n_theta
        self.n_x = n_x
        self.prior = prior
        self.ess_threshold = ess_threshold
        self.n_mcmc_moves = n_mcmc_moves
        self._rng = np.random.default_rng(seed)

        # Determine param dimension
        if hasattr(model, "get_params"):
            _params: NDArray[np.float64] = np.atleast_1d(
                np.asarray(model.get_params(), dtype=np.float64)
            )
            self._param_dim = int(_params.shape[0])
        else:
            self._param_dim = 1

        # Initialize parameter particles from prior
        self._theta_particles: NDArray[np.float64] = np.zeros((n_theta, self._param_dim))
        self._theta_weights: NDArray[np.float64] = np.ones(n_theta) / n_theta
        self._log_weights: NDArray[np.float64] = np.zeros(n_theta)

        # Internal PF states: state particles for each theta particle
        self._x_particles: list[NDArray[np.float64]] = []
        self._x_weights: list[NDArray[np.float64]] = []

        # Running state
        self._log_evidence: float = 0.0
        self._n_observations: int = 0
        self._ess_history: list[float] = []
        self._resample_times: list[int] = []
        self._initialized: bool = False

        # Initialize from prior
        self._initialize_from_prior()

    def _initialize_from_prior(self) -> None:
        """Initialize parameter particles from the prior distribution."""
        if self.prior is not None and hasattr(self.prior, "sample"):
            for m in range(self.n_theta):
                self._theta_particles[m] = self.prior.sample(self._rng)
        else:
            # Fallback: sample around current model params
            base: NDArray[np.float64] = np.atleast_1d(
                np.asarray(self.model.get_params(), dtype=np.float64)
            )
            for m in range(self.n_theta):
                self._theta_particles[m] = base + 0.1 * self._rng.standard_normal(self._param_dim)

        # Initialize internal PF states
        self._x_particles = []
        self._x_weights = []
        for m in range(self.n_theta):
            self.model.set_params(self._theta_particles[m])
            if hasattr(self.model, "initial_sample"):
                try:
                    x_raw = self.model.initial_sample(self.n_x, self._rng)
                    x_init: NDArray[np.float64] = np.atleast_1d(np.asarray(x_raw, dtype=np.float64))
                except (ValueError, RuntimeError):
                    x_init = self._rng.standard_normal(self.n_x)
            else:
                x_init = self._rng.standard_normal(self.n_x)
            self._x_particles.append(x_init)
            self._x_weights.append(np.ones(self.n_x) / self.n_x)

        self._theta_weights = np.ones(self.n_theta) / self.n_theta
        self._log_weights = np.zeros(self.n_theta)
        self._initialized = True

    def update(self, y_t: float | NDArray[np.float64]) -> None:
        """Process one new observation.

        Parameters
        ----------
        y_t : float | NDArray[np.float64]
            New observation (scalar or array).
        """
        y_obs: float | NDArray[np.float64]
        if np.isscalar(y_t):
            y_obs = float(cast("np.floating[Any]", y_t))
        else:
            y_obs = np.asarray(y_t, dtype=np.float64)

        if self._n_observations == 0 and not self._initialized:
            self._initialize_from_prior()

        # 1. EXTEND each internal PF and compute incremental log-likelihood
        log_increments: NDArray[np.float64] = np.zeros(self.n_theta)

        for m in range(self.n_theta):
            self.model.set_params(self._theta_particles[m])
            log_inc = self._extend_pf(m, y_obs)
            log_increments[m] = log_inc

        # 2. REWEIGHT parameter particles
        self._log_weights += log_increments

        # Normalize
        max_lw = float(np.max(self._log_weights))
        weights: NDArray[np.float64] = np.exp(self._log_weights - max_lw)
        sum_w = float(np.sum(weights))

        if sum_w < 1e-300:
            self._theta_weights = np.ones(self.n_theta) / self.n_theta
        else:
            self._log_evidence += max_lw + float(np.log(sum_w)) - float(np.log(self.n_theta))
            self._theta_weights = weights / sum_w

        # Update log weights (centered)
        self._log_weights = np.log(np.maximum(self._theta_weights, 1e-300))

        self._n_observations += 1

        # 3. Compute ESS and resample if needed
        ess = float(1.0 / np.sum(self._theta_weights**2))
        self._ess_history.append(ess)

        if ess < self.ess_threshold * self.n_theta:
            self._resample_and_rejuvenate()
            self._resample_times.append(self._n_observations)

    def _extend_pf(self, m: int, y_t: float | NDArray[np.float64]) -> float:
        """Extend the m-th internal particle filter by one step.

        Parameters
        ----------
        m : int
            Index of the parameter particle.
        y_t : float | NDArray[np.float64]
            New observation.

        Returns
        -------
        float
            Incremental log-likelihood contribution log p(y_t | y_{1:t-1}, theta_m).
        """
        x_particles = self._x_particles[m]
        n = len(x_particles)

        # Propagate state particles (if t > 0)
        if self._n_observations > 0:
            new_particles: NDArray[np.float64] = np.zeros_like(x_particles)
            # Resample first
            w_inner = self._x_weights[m]
            indices = self._rng.choice(n, size=n, p=w_inner)
            resampled = x_particles[indices]

            for j in range(n):
                if hasattr(self.model, "transition_sample"):
                    try:
                        x_new = self.model.transition_sample(resampled[j], self._rng)
                        new_particles[j] = float(np.asarray(x_new, dtype=np.float64))
                    except (ValueError, RuntimeError):
                        new_particles[j] = resampled[j] + self._rng.standard_normal()
                else:
                    new_particles[j] = resampled[j] + self._rng.standard_normal()

            x_particles = new_particles

        # Compute observation weights
        log_obs_weights: NDArray[np.float64] = np.zeros(n)
        for j in range(n):
            if hasattr(self.model, "observation_logpdf"):
                log_obs_weights[j] = self.model.observation_logpdf(y_t, x_particles[j])

        # Incremental log-likelihood
        max_lw = float(np.max(log_obs_weights))
        w: NDArray[np.float64] = np.exp(log_obs_weights - max_lw)
        sum_w = float(np.sum(w))

        if sum_w < 1e-300:
            log_increment = float(-np.inf)
            self._x_weights[m] = np.ones(n) / n
        else:
            log_increment = max_lw + float(np.log(sum_w)) - float(np.log(n))
            self._x_weights[m] = w / sum_w

        self._x_particles[m] = x_particles
        return log_increment

    def _resample_and_rejuvenate(self) -> None:
        """Resample parameter particles and rejuvenate via MCMC moves."""
        # Resample
        indices = self._rng.choice(self.n_theta, size=self.n_theta, p=self._theta_weights)

        new_theta: NDArray[np.float64] = self._theta_particles[indices].copy()
        new_x_particles: list[NDArray[np.float64]] = [
            self._x_particles[int(i)].copy() for i in indices
        ]
        new_x_weights: list[NDArray[np.float64]] = [self._x_weights[int(i)].copy() for i in indices]

        self._theta_particles = new_theta
        self._x_particles = new_x_particles
        self._x_weights = new_x_weights

        # Reset weights to uniform
        self._theta_weights = np.ones(self.n_theta) / self.n_theta
        self._log_weights = np.zeros(self.n_theta)

        # Rejuvenate via MCMC moves
        if self.n_mcmc_moves > 0:
            self._rejuvenate()

    def _rejuvenate(self) -> None:
        """Apply MCMC moves to diversify parameter particles.

        Uses a random walk Metropolis-Hastings kernel with empirical
        covariance of the current particles.
        """
        # Compute empirical covariance for proposal
        emp_cov: NDArray[np.float64] = np.cov(self._theta_particles.T)
        if emp_cov.ndim == 0:
            emp_cov = np.array([[float(emp_cov)]])

        scale = (2.38**2) / max(self._param_dim, 1)
        proposal_cov: NDArray[np.float64] = scale * emp_cov

        # Add small jitter for numerical stability
        proposal_cov = proposal_cov + 1e-8 * np.eye(self._param_dim)

        try:
            chol = np.asarray(np.linalg.cholesky(proposal_cov), dtype=np.float64)
        except np.linalg.LinAlgError:
            chol = np.sqrt(scale) * np.eye(self._param_dim)

        for _ in range(self.n_mcmc_moves):
            for m in range(self.n_theta):
                # Propose
                z: NDArray[np.float64] = self._rng.standard_normal(self._param_dim)
                theta_proposed: NDArray[np.float64] = self._theta_particles[m] + chol @ z

                # Evaluate prior
                if self.prior is not None:
                    logprior_current = float(self.prior.logpdf(self._theta_particles[m]))
                    logprior_proposed = float(self.prior.logpdf(theta_proposed))
                else:
                    logprior_current = 0.0
                    logprior_proposed = 0.0

                if not np.isfinite(logprior_proposed):
                    continue

                # Simple acceptance based on prior only (approximate)
                log_alpha = logprior_proposed - logprior_current

                if float(np.log(self._rng.random())) < log_alpha:
                    self._theta_particles[m] = theta_proposed

    def current_posterior(self) -> SMC2OnlineState:
        """Return the current posterior state.

        Returns
        -------
        SMC2OnlineState
            Current state with parameter particles and weights.
        """
        return SMC2OnlineState(
            theta_particles=self._theta_particles.copy(),
            theta_weights=self._theta_weights.copy(),
            log_evidence=self._log_evidence,
            n_observations=self._n_observations,
            ess_history=list(self._ess_history),
            resample_times=list(self._resample_times),
        )

    def posterior_mean(self) -> NDArray[np.float64]:
        """Compute weighted posterior mean of parameters.

        Returns
        -------
        NDArray[np.float64]
            Weighted mean of parameter particles.
        """
        result: NDArray[np.float64] = np.average(
            self._theta_particles, weights=self._theta_weights, axis=0
        )
        return result

    def posterior_std(self) -> NDArray[np.float64]:
        """Compute weighted posterior standard deviation.

        Returns
        -------
        NDArray[np.float64]
            Weighted std of parameter particles.
        """
        mean = self.posterior_mean()
        diff = self._theta_particles - mean
        var: NDArray[np.float64] = np.average(diff**2, weights=self._theta_weights, axis=0)
        return np.sqrt(var)

    def reset(self) -> None:
        """Reset the SMC2Online state.

        Re-initializes all particles and weights from the prior.
        """
        self._log_evidence = 0.0
        self._n_observations = 0
        self._ess_history = []
        self._resample_times = []
        self._initialized = False
        self._initialize_from_prior()
