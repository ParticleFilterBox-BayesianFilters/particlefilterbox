"""MCMC kernels for particle rejuvenation in SMC methods.

After resampling, particles are duplicated and need diversification.
These MCMC kernels perform Metropolis-Hastings moves that preserve the
current target distribution, preventing particle degeneracy.

References:
    Roberts, G.O. & Rosenthal, J.S. (2009). Examples of adaptive MCMC.
    Journal of Computational and Graphical Statistics, 18(2), 349-367.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from particlefilterbox._logging import get_logger

logger = get_logger("smc.mcmc_moves")


class LogTargetFn(Protocol):
    """Protocol for log-target density functions."""

    def __call__(self, theta: NDArray[np.floating[Any]]) -> float: ...


@dataclass
class MCMCStepResult:
    """Result of a single MCMC step.

    Attributes
    ----------
    theta : NDArray, shape (k,)
        Parameter vector after the step (accepted or rejected).
    accepted : bool
        Whether the proposal was accepted.
    log_target : float
        Log-target density at the returned theta.
    """

    theta: NDArray[np.floating[Any]]
    accepted: bool
    log_target: float


class RandomWalkMH:
    """Random Walk Metropolis-Hastings kernel.

    Proposes new parameters by adding Gaussian noise to current values.
    The proposal is symmetric, so the acceptance ratio simplifies to
    the likelihood ratio.

    Parameters
    ----------
    proposal_cov : NDArray, shape (k, k) or (k,)
        Covariance matrix (or diagonal) of the Gaussian proposal.
        If 1-D, interpreted as diagonal variances.
    adaptive : bool
        If True, adapt the proposal scale based on acceptance rate.
        Default is False.

    Examples
    --------
    >>> import numpy as np
    >>> kernel = RandomWalkMH(proposal_cov=np.eye(2) * 0.1)
    >>> rng = np.random.default_rng(42)
    >>> def log_target(x):
    ...     return -0.5 * np.sum(x**2)
    >>> result = kernel.step(np.zeros(2), log_target, rng)
    """

    def __init__(
        self,
        proposal_cov: NDArray[np.floating[Any]],
        adaptive: bool = False,
    ) -> None:
        proposal_cov = np.asarray(proposal_cov, dtype=np.float64)
        if proposal_cov.ndim == 1:
            self._cov = np.diag(proposal_cov)
        elif proposal_cov.ndim == 2:
            self._cov = proposal_cov.copy()
        else:
            msg = f"proposal_cov must be 1-D or 2-D, got {proposal_cov.ndim}-D"
            raise ValueError(msg)

        self.adaptive = adaptive
        self._scale: float = 1.0
        self._dim = self._cov.shape[0]

        # Pre-compute Cholesky for sampling
        self._chol = np.linalg.cholesky(self._cov)

    @property
    def proposal_cov(self) -> NDArray[np.floating[Any]]:
        """Current proposal covariance (including scale)."""
        return self._scale**2 * self._cov

    def step(
        self,
        theta: NDArray[np.floating[Any]],
        log_target: LogTargetFn,
        rng: np.random.Generator,
        log_target_current: float | None = None,
    ) -> MCMCStepResult:
        """Perform one Metropolis-Hastings step.

        Parameters
        ----------
        theta : NDArray, shape (k,)
            Current parameter vector.
        log_target : callable
            Log-density of the target distribution.
        rng : np.random.Generator
            Random number generator.
        log_target_current : float or None
            Log-target at current theta. If None, will be computed.

        Returns
        -------
        MCMCStepResult
            Result containing new theta, acceptance flag, and log-target.
        """
        theta = np.asarray(theta, dtype=np.float64)

        if log_target_current is None:
            log_target_current = log_target(theta)

        # Propose: theta* = theta + scale * L @ z, z ~ N(0, I)
        z = rng.standard_normal(self._dim)
        theta_prop = theta + self._scale * (self._chol @ z)

        # Evaluate target at proposal
        log_target_prop = log_target(theta_prop)

        # Accept/reject (symmetric proposal -> ratio = target ratio)
        log_alpha = log_target_prop - log_target_current
        log_u = np.log(rng.uniform())

        if log_u < log_alpha:
            return MCMCStepResult(
                theta=theta_prop,
                accepted=True,
                log_target=log_target_prop,
            )
        return MCMCStepResult(
            theta=theta.copy(),
            accepted=False,
            log_target=log_target_current,
        )


class AdaptiveMH:
    """Adaptive Metropolis-Hastings kernel.

    Automatically tunes the proposal scale to achieve a target acceptance
    rate (default 0.234, optimal for high-dimensional targets per
    Roberts & Rosenthal 2001).

    Parameters
    ----------
    dim : int
        Dimensionality of the parameter space.
    initial_cov : NDArray or None
        Initial proposal covariance. If None, uses identity * 2.38^2/dim.
    target_acceptance : float
        Target acceptance rate. Default is 0.234.

    References
    ----------
    Roberts, G.O. & Rosenthal, J.S. (2009). Examples of adaptive MCMC.
    """

    def __init__(
        self,
        dim: int,
        initial_cov: NDArray[np.floating[Any]] | None = None,
        target_acceptance: float = 0.234,
    ) -> None:
        self.dim = dim
        self.target_acceptance = target_acceptance

        if initial_cov is not None:
            self._cov = np.asarray(initial_cov, dtype=np.float64).copy()
        else:
            # Optimal scaling for Gaussian targets: (2.38^2 / d) * I
            self._cov = np.eye(dim) * (2.38**2 / dim)

        self._scale: float = 1.0
        self._n_calls: int = 0
        self._n_accepted: int = 0
        self._chol = np.linalg.cholesky(self._cov)

        # For online covariance estimation
        self._mean: NDArray[np.floating[Any]] = np.zeros(dim, dtype=np.float64)
        self._m2: NDArray[np.floating[Any]] = np.zeros((dim, dim), dtype=np.float64)

    @property
    def acceptance_rate(self) -> float:
        """Current empirical acceptance rate."""
        if self._n_calls == 0:
            return 0.0
        return self._n_accepted / self._n_calls

    def step(
        self,
        theta: NDArray[np.floating[Any]],
        log_target: LogTargetFn,
        rng: np.random.Generator,
        log_target_current: float | None = None,
    ) -> MCMCStepResult:
        """Perform one adaptive MH step.

        Parameters
        ----------
        theta : NDArray, shape (k,)
            Current parameter vector.
        log_target : callable
            Log-density of the target distribution.
        rng : np.random.Generator
            Random number generator.
        log_target_current : float or None
            Log-target at current theta. If None, will be computed.

        Returns
        -------
        MCMCStepResult
            Result with new theta, acceptance flag, and log-target.
        """
        theta = np.asarray(theta, dtype=np.float64)

        if log_target_current is None:
            log_target_current = log_target(theta)

        # Propose
        z = rng.standard_normal(self.dim)
        theta_prop = theta + self._scale * (self._chol @ z)

        log_target_prop = log_target(theta_prop)

        log_alpha = log_target_prop - log_target_current
        log_u = np.log(rng.uniform())

        self._n_calls += 1

        if log_u < log_alpha:
            accepted = True
            theta_out = theta_prop
            log_target_out = log_target_prop
        else:
            accepted = False
            theta_out = theta.copy()
            log_target_out = log_target_current

        if accepted:
            self._n_accepted += 1

        # Adapt after each step
        self.adapt(float(accepted))

        # Update online covariance estimate
        self._update_cov_estimate(theta_out)

        return MCMCStepResult(
            theta=theta_out,
            accepted=accepted,
            log_target=log_target_out,
        )

    def adapt(self, acceptance_rate: float) -> None:
        """Adapt the proposal scale based on acceptance rate.

        Uses the Robbins-Monro stochastic approximation:
        log(scale) += gamma_n * (acceptance - target)

        Parameters
        ----------
        acceptance_rate : float
            Most recent acceptance indicator (0 or 1) or smoothed rate.
        """
        # Robbins-Monro step size: gamma_n = 1/n^0.6
        gamma = max(1.0 / (self._n_calls + 1) ** 0.6, 0.01)
        log_scale = np.log(self._scale)
        log_scale += gamma * (acceptance_rate - self.target_acceptance)
        # Bound the scale to prevent extreme values
        self._scale = float(np.exp(np.clip(log_scale, -5.0, 5.0)))

    def _update_cov_estimate(self, theta: NDArray[np.floating[Any]]) -> None:
        """Update online covariance estimate using Welford's algorithm.

        After enough samples, use the empirical covariance for proposals.

        Parameters
        ----------
        theta : NDArray, shape (k,)
            Most recent sample.
        """
        n = self._n_calls
        delta = theta - self._mean
        self._mean = self._mean + delta / n
        delta2 = theta - self._mean
        self._m2 = self._m2 + np.outer(delta, delta2)

        # Update Cholesky after burn-in (at least 2*dim samples)
        if n > 2 * self.dim and n % self.dim == 0:
            empirical_cov = self._m2 / (n - 1)
            # Add small regularization
            regularized = empirical_cov + 1e-6 * np.eye(self.dim)
            try:
                self._chol = np.linalg.cholesky(regularized)
                self._cov = regularized
            except np.linalg.LinAlgError:
                logger.debug("Cholesky failed, keeping previous covariance")


class IndependentMH:
    """Independent Metropolis-Hastings kernel.

    Proposes from a fixed distribution (not centered on current state).
    Useful when a good approximation to the target is available.

    Parameters
    ----------
    proposal_mean : NDArray, shape (k,)
        Mean of the Gaussian proposal distribution.
    proposal_cov : NDArray, shape (k, k)
        Covariance of the Gaussian proposal distribution.
    """

    def __init__(
        self,
        proposal_mean: NDArray[np.floating[Any]],
        proposal_cov: NDArray[np.floating[Any]],
    ) -> None:
        self._mean = np.asarray(proposal_mean, dtype=np.float64)
        self._cov = np.asarray(proposal_cov, dtype=np.float64)
        self._dim = len(self._mean)
        self._chol = np.linalg.cholesky(self._cov)

        # Pre-compute for log-proposal density
        self._log_det = float(2.0 * np.sum(np.log(np.diag(self._chol))))
        self._cov_inv = np.linalg.inv(self._cov)
        self._log_norm = -0.5 * (self._dim * np.log(2.0 * np.pi) + self._log_det)

    def _log_proposal(self, theta: NDArray[np.floating[Any]]) -> float:
        """Log-density of the proposal distribution.

        Parameters
        ----------
        theta : NDArray, shape (k,)
            Point at which to evaluate.

        Returns
        -------
        float
            Log-density q(theta).
        """
        diff = theta - self._mean
        return float(self._log_norm - 0.5 * diff @ self._cov_inv @ diff)

    def step(
        self,
        theta: NDArray[np.floating[Any]],
        log_target: LogTargetFn,
        rng: np.random.Generator,
        log_target_current: float | None = None,
    ) -> MCMCStepResult:
        """Perform one independent MH step.

        Parameters
        ----------
        theta : NDArray, shape (k,)
            Current parameter vector.
        log_target : callable
            Log-density of the target distribution.
        rng : np.random.Generator
            Random number generator.
        log_target_current : float or None
            Log-target at current theta.

        Returns
        -------
        MCMCStepResult
            Result with new theta, acceptance flag, and log-target.
        """
        theta = np.asarray(theta, dtype=np.float64)

        if log_target_current is None:
            log_target_current = log_target(theta)

        # Propose from fixed distribution
        z = rng.standard_normal(self._dim)
        theta_prop = self._mean + self._chol @ z

        log_target_prop = log_target(theta_prop)

        # Hastings ratio includes proposal ratio (asymmetric)
        log_q_current = self._log_proposal(theta)
        log_q_prop = self._log_proposal(theta_prop)
        log_alpha = log_target_prop - log_target_current + log_q_current - log_q_prop

        log_u = np.log(rng.uniform())

        if log_u < log_alpha:
            return MCMCStepResult(
                theta=theta_prop,
                accepted=True,
                log_target=log_target_prop,
            )
        return MCMCStepResult(
            theta=theta.copy(),
            accepted=False,
            log_target=log_target_current,
        )


class MALAKernel:
    """Metropolis-Adjusted Langevin Algorithm kernel.

    Uses gradient information to make informed proposals. The proposal
    is a Gaussian centered at theta + (step_size/2) * grad(log_target).

    Parameters
    ----------
    step_size : float
        Step size (epsilon) for the Langevin dynamics. Default is 0.1.
    dim : int
        Dimensionality of the parameter space.

    Notes
    -----
    Requires the log-target to be differentiable. Gradient is computed
    via finite differences if not provided analytically.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        dim: int = 1,
    ) -> None:
        if step_size <= 0:
            msg = f"step_size must be positive, got {step_size}"
            raise ValueError(msg)
        self.step_size = step_size
        self.dim = dim

    def _grad_log_target(
        self,
        theta: NDArray[np.floating[Any]],
        log_target: LogTargetFn,
        eps: float = 1e-5,
    ) -> NDArray[np.floating[Any]]:
        """Compute gradient of log-target via central finite differences.

        Parameters
        ----------
        theta : NDArray, shape (k,)
            Point at which to evaluate gradient.
        log_target : callable
            Log-target density function.
        eps : float
            Step size for finite differences.

        Returns
        -------
        NDArray, shape (k,)
            Gradient vector.
        """
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += eps
            theta_minus[i] -= eps
            grad[i] = (log_target(theta_plus) - log_target(theta_minus)) / (2.0 * eps)
        return grad

    def _log_proposal(
        self,
        theta_to: NDArray[np.floating[Any]],
        theta_from: NDArray[np.floating[Any]],
        grad_from: NDArray[np.floating[Any]],
    ) -> float:
        """Log-density of MALA proposal q(theta_to | theta_from).

        q(theta_to | theta_from) = N(theta_to; mu, step_size * I)
        where mu = theta_from + (step_size/2) * grad_log_target(theta_from)

        Parameters
        ----------
        theta_to : NDArray, shape (k,)
            Proposed point.
        theta_from : NDArray, shape (k,)
            Current point.
        grad_from : NDArray, shape (k,)
            Gradient at current point.

        Returns
        -------
        float
            Log-density of the proposal.
        """
        mu = theta_from + 0.5 * self.step_size * grad_from
        diff = theta_to - mu
        log_q = -0.5 * np.sum(diff**2) / self.step_size
        log_q -= 0.5 * len(theta_from) * np.log(2.0 * np.pi * self.step_size)
        return float(log_q)

    def step(
        self,
        theta: NDArray[np.floating[Any]],
        log_target: LogTargetFn,
        rng: np.random.Generator,
        log_target_current: float | None = None,
    ) -> MCMCStepResult:
        """Perform one MALA step.

        Parameters
        ----------
        theta : NDArray, shape (k,)
            Current parameter vector.
        log_target : callable
            Log-density of the target distribution.
        rng : np.random.Generator
            Random number generator.
        log_target_current : float or None
            Log-target at current theta.

        Returns
        -------
        MCMCStepResult
            Result with new theta, acceptance flag, and log-target.
        """
        theta = np.asarray(theta, dtype=np.float64)

        if log_target_current is None:
            log_target_current = log_target(theta)

        # Compute gradient at current position
        grad_current = self._grad_log_target(theta, log_target)

        # Propose: theta* = theta + (eps/2)*grad + sqrt(eps)*z
        mu = theta + 0.5 * self.step_size * grad_current
        noise = rng.standard_normal(self.dim) * np.sqrt(self.step_size)
        theta_prop = mu + noise

        log_target_prop = log_target(theta_prop)

        # Compute gradient at proposal for reverse proposal density
        grad_prop = self._grad_log_target(theta_prop, log_target)

        # Hastings ratio with MALA correction
        log_q_forward = self._log_proposal(theta_prop, theta, grad_current)
        log_q_reverse = self._log_proposal(theta, theta_prop, grad_prop)
        log_alpha = log_target_prop - log_target_current + log_q_reverse - log_q_forward

        log_u = np.log(rng.uniform())

        if log_u < log_alpha:
            return MCMCStepResult(
                theta=theta_prop,
                accepted=True,
                log_target=log_target_prop,
            )
        return MCMCStepResult(
            theta=theta.copy(),
            accepted=False,
            log_target=log_target_current,
        )


def random_walk_mh(
    theta: NDArray[np.floating[Any]],
    log_target: LogTargetFn,
    proposal_cov: NDArray[np.floating[Any]],
    rng: np.random.Generator,
    n_steps: int = 1,
) -> tuple[NDArray[np.floating[Any]], int]:
    """Run a short Random Walk MH chain.

    Convenience function for running multiple RW-MH steps.

    Parameters
    ----------
    theta : NDArray, shape (k,)
        Starting parameter vector.
    log_target : callable
        Log-density of the target distribution.
    proposal_cov : NDArray, shape (k, k) or (k,)
        Proposal covariance (or diagonal).
    rng : np.random.Generator
        Random number generator.
    n_steps : int
        Number of MH steps to run. Default is 1.

    Returns
    -------
    tuple of (theta_final, n_accepted)
        Final parameter vector and number of accepted steps.
    """
    kernel = RandomWalkMH(proposal_cov=proposal_cov)
    current = theta.copy()
    n_accepted = 0
    log_t: float | None = None

    for _ in range(n_steps):
        result = kernel.step(current, log_target, rng, log_target_current=log_t)
        current = result.theta
        log_t = result.log_target
        if result.accepted:
            n_accepted += 1

    return current, n_accepted


def run_mcmc_chain(
    theta_init: NDArray[np.floating[Any]],
    log_target: LogTargetFn,
    kernel: RandomWalkMH | AdaptiveMH | IndependentMH | MALAKernel,
    rng: np.random.Generator,
    n_steps: int = 10,
) -> tuple[NDArray[np.floating[Any]], float]:
    """Run a short MCMC chain using any kernel.

    Parameters
    ----------
    theta_init : NDArray, shape (k,)
        Starting parameter vector.
    log_target : callable
        Log-density of the target distribution.
    kernel : MCMC kernel
        One of RandomWalkMH, AdaptiveMH, IndependentMH, MALAKernel.
    rng : np.random.Generator
        Random number generator.
    n_steps : int
        Number of steps to run.

    Returns
    -------
    tuple of (theta_final, acceptance_rate)
        Final parameter and empirical acceptance rate over the chain.
    """
    current = theta_init.copy()
    n_accepted = 0
    log_t: float | None = None

    for _ in range(n_steps):
        result = kernel.step(current, log_target, rng, log_target_current=log_t)
        current = result.theta
        log_t = result.log_target
        if result.accepted:
            n_accepted += 1

    acceptance_rate = n_accepted / n_steps if n_steps > 0 else 0.0
    return current, acceptance_rate
