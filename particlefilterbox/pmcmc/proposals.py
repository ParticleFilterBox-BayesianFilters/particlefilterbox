"""Proposal distributions for PMCMC methods.

Provides various proposal mechanisms for the Metropolis-Hastings step
in Particle MCMC algorithms.

References:
    Roberts, G. O. & Rosenthal, J. S. (2009). Examples of adaptive MCMC.
    Journal of Computational and Graphical Statistics, 18(2), 349-367.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BaseProposal",
    "GaussianRandomWalk",
    "AdaptiveGaussian",
    "LogNormalProposal",
    "TransformedProposal",
]


class BaseProposal(abc.ABC):
    """Abstract base class for MCMC proposal distributions.

    Parameters
    ----------
    dim : int
        Dimension of the parameter space.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(self, dim: int, seed: int | None = None) -> None:
        self.dim = dim
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def propose(
        self,
        theta_current: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Generate a proposal given the current state.

        Parameters
        ----------
        theta_current : NDArray[np.float64]
            Current parameter vector of shape ``(d,)``.
        rng : np.random.Generator | None
            Random number generator. Uses internal RNG if None.

        Returns
        -------
        NDArray[np.float64]
            Proposed parameter vector of shape ``(d,)``.
        """
        ...

    def log_ratio(
        self,
        theta_proposed: NDArray[np.float64],
        theta_current: NDArray[np.float64],
    ) -> float:
        """Log proposal ratio log q(current|proposed) - log q(proposed|current).

        For symmetric proposals this returns 0.0.

        Parameters
        ----------
        theta_proposed : NDArray[np.float64]
            Proposed parameter vector.
        theta_current : NDArray[np.float64]
            Current parameter vector.

        Returns
        -------
        float
            Log proposal ratio.
        """
        return 0.0

    def adapt(self, theta: NDArray[np.float64], accepted: bool) -> None:  # noqa: B027
        """Update internal state after an MCMC step.

        Default implementation does nothing. Override for adaptive proposals.

        Parameters
        ----------
        theta : NDArray[np.float64]
            Parameter vector after accept/reject.
        accepted : bool
            Whether the proposal was accepted.
        """


class GaussianRandomWalk(BaseProposal):
    """Gaussian random walk proposal with fixed covariance.

    Proposes theta* = theta + epsilon, where epsilon ~ N(0, cov).

    Parameters
    ----------
    dim : int
        Dimension of the parameter space.
    cov : NDArray[np.float64] | float | None
        Covariance matrix of shape ``(d, d)``, or scalar for isotropic.
        If None, uses identity scaled by ``(2.38**2)/d``.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dim: int,
        cov: NDArray[np.float64] | float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim=dim, seed=seed)

        if cov is None:
            scale = (2.38**2) / max(dim, 1)
            self.cov = scale * np.eye(dim)
        elif np.isscalar(cov):
            self.cov = float(cov) * np.eye(dim)  # type: ignore[arg-type]
        else:
            self.cov = np.asarray(cov, dtype=np.float64)

        # Precompute Cholesky for efficient sampling
        self._chol = np.linalg.cholesky(self.cov)

    def propose(
        self,
        theta_current: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Generate Gaussian random walk proposal.

        Parameters
        ----------
        theta_current : NDArray[np.float64]
            Current parameter vector of shape ``(d,)``.
        rng : np.random.Generator | None
            Random number generator.

        Returns
        -------
        NDArray[np.float64]
            Proposed parameter vector.
        """
        if rng is None:
            rng = self._rng
        z = rng.standard_normal(self.dim)
        return theta_current + self._chol @ z


class AdaptiveGaussian(BaseProposal):
    """Adaptive Gaussian proposal with Roberts-Rosenthal scaling.

    Uses the optimal scaling rule: scale = (2.38^2)/d, and adapts the
    covariance matrix based on the empirical covariance of accepted samples.

    The adaptation follows Roberts & Rosenthal (2009):
        Sigma_n = (1-beta)*s_d*Sigma_emp + beta*s_d*(0.1^2/d)*I_d
    where:
        - s_d = (2.38^2)/d is the optimal scaling
        - Sigma_emp is the empirical covariance of the chain so far
        - beta is a small weight (default 0.05) on the identity component
        - The identity component prevents degeneracy

    Parameters
    ----------
    dim : int
        Dimension of the parameter space.
    initial_cov : NDArray[np.float64] | float | None
        Initial covariance matrix. If None, uses ``(2.38^2/d)*I``.
    target_acceptance : float
        Target acceptance rate for adaptation. Default 0.234 (optimal for
        Gaussian targets in high dimensions).
    adaptation_start : int
        Iteration at which to start adapting. Default 100.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dim: int,
        initial_cov: NDArray[np.float64] | float | None = None,
        target_acceptance: float = 0.234,
        adaptation_start: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim=dim, seed=seed)

        self.target_acceptance = target_acceptance
        self.adaptation_start = adaptation_start

        # Roberts-Rosenthal optimal scaling
        self.s_d: float = (2.38**2) / max(dim, 1)
        self.beta: float = 0.05  # Weight on identity component

        # Initialize covariance
        if initial_cov is None:
            self._cov = self.s_d * np.eye(dim)
        elif np.isscalar(initial_cov):
            self._cov = float(initial_cov) * np.eye(dim)  # type: ignore[arg-type]
        else:
            self._cov = np.asarray(initial_cov, dtype=np.float64)

        self._chol = np.linalg.cholesky(self._cov)

        # Running statistics for empirical covariance
        self._n_samples: int = 0
        self._mean: NDArray[np.float64] = np.zeros(dim)
        self._cov_sum: NDArray[np.float64] = np.zeros((dim, dim))
        self._samples: list[NDArray[np.float64]] = []

        # Log-scale adaptation for global scaling (Andrieu & Roberts 2009)
        self._log_scale: float = 0.0
        self._n_accepted: int = 0
        self._n_total: int = 0

    @property
    def cov(self) -> NDArray[np.float64]:
        """Current proposal covariance matrix."""
        return self._cov.copy()

    def propose(
        self,
        theta_current: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Generate adaptive Gaussian proposal.

        Parameters
        ----------
        theta_current : NDArray[np.float64]
            Current parameter vector of shape ``(d,)``.
        rng : np.random.Generator | None
            Random number generator.

        Returns
        -------
        NDArray[np.float64]
            Proposed parameter vector.
        """
        if rng is None:
            rng = self._rng

        scale = np.exp(self._log_scale)
        z = rng.standard_normal(self.dim)
        return theta_current + scale * (self._chol @ z)

    def adapt(self, theta: NDArray[np.float64], accepted: bool) -> None:
        """Update proposal based on MCMC step outcome.

        Adapts both the covariance matrix (using empirical covariance) and
        the global scale (to track target acceptance rate).

        Parameters
        ----------
        theta : NDArray[np.float64]
            Current parameter vector (after accept/reject decision).
        accepted : bool
            Whether the proposal was accepted.
        """
        self._n_total += 1
        if accepted:
            self._n_accepted += 1

        # Update running mean and covariance
        self._n_samples += 1
        self._samples.append(theta.copy())

        # Welford's online algorithm for mean
        delta = theta - self._mean
        self._mean = self._mean + delta / self._n_samples
        delta2 = theta - self._mean
        self._cov_sum = self._cov_sum + np.outer(delta, delta2)

        # Adapt covariance after adaptation_start iterations
        if self._n_samples >= self.adaptation_start:
            self._update_covariance()

        # Adapt global scale using Robbins-Monro
        # gamma_n = 1/n^0.6 (diminishing adaptation)
        gamma_n = 1.0 / (self._n_total**0.6)
        current_rate = 1.0 if accepted else 0.0
        self._log_scale += gamma_n * (current_rate - self.target_acceptance)

    def _update_covariance(self) -> None:
        """Update proposal covariance using empirical covariance."""
        if self._n_samples < 2:
            return

        # Empirical covariance
        emp_cov = self._cov_sum / (self._n_samples - 1)

        # Roberts-Rosenthal mixture:
        # (1-beta)*s_d*Sigma_emp + beta*s_d*(0.1^2/d)*I_d
        identity_scale = (0.1**2) / max(self.dim, 1)
        self._cov = (
            1 - self.beta
        ) * self.s_d * emp_cov + self.beta * self.s_d * identity_scale * np.eye(self.dim)

        # Ensure positive definiteness and recompute Cholesky
        try:
            self._chol = np.linalg.cholesky(self._cov)
        except np.linalg.LinAlgError:
            # Fallback: add jitter
            jitter = 1e-6 * np.eye(self.dim)
            self._cov = self._cov + jitter
            self._chol = np.linalg.cholesky(self._cov)


class LogNormalProposal(BaseProposal):
    """Log-normal proposal for strictly positive parameters.

    Proposes in log-space: log(theta*) = log(theta) + epsilon,
    where epsilon ~ N(0, cov). This ensures theta* > 0.

    The proposal ratio accounts for the Jacobian of the log transform.

    Parameters
    ----------
    dim : int
        Dimension of the parameter space.
    cov : NDArray[np.float64] | float | None
        Covariance in log-space. If None, uses ``(2.38^2/d)*I``.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dim: int,
        cov: NDArray[np.float64] | float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim=dim, seed=seed)

        if cov is None:
            scale = (2.38**2) / max(dim, 1)
            self._log_cov = scale * np.eye(dim)
        elif np.isscalar(cov):
            self._log_cov = float(cov) * np.eye(dim)  # type: ignore[arg-type]
        else:
            self._log_cov = np.asarray(cov, dtype=np.float64)

        self._log_chol = np.linalg.cholesky(self._log_cov)

    def propose(
        self,
        theta_current: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Generate log-normal proposal.

        Parameters
        ----------
        theta_current : NDArray[np.float64]
            Current parameter vector (must be positive).
        rng : np.random.Generator | None
            Random number generator.

        Returns
        -------
        NDArray[np.float64]
            Proposed parameter vector (positive).
        """
        if rng is None:
            rng = self._rng

        log_theta = np.log(np.maximum(theta_current, 1e-300))
        z = rng.standard_normal(self.dim)
        log_proposed = log_theta + self._log_chol @ z
        return np.exp(log_proposed)

    def log_ratio(
        self,
        theta_proposed: NDArray[np.float64],
        theta_current: NDArray[np.float64],
    ) -> float:
        """Log proposal ratio with Jacobian correction.

        For log-normal proposal:
            log q(current|proposed) - log q(proposed|current)
            = sum(log(proposed)) - sum(log(current))

        This is the Jacobian correction for the log transform.

        Parameters
        ----------
        theta_proposed : NDArray[np.float64]
            Proposed parameter vector.
        theta_current : NDArray[np.float64]
            Current parameter vector.

        Returns
        -------
        float
            Log proposal ratio.
        """
        # Jacobian: d/d(theta) log(theta) = 1/theta
        # log |J(proposed->current)| - log |J(current->proposed)|
        log_ratio = float(
            np.sum(np.log(np.maximum(theta_proposed, 1e-300)))
            - np.sum(np.log(np.maximum(theta_current, 1e-300)))
        )
        return log_ratio


class TransformedProposal(BaseProposal):
    """Proposal in a transformed (unconstrained) space.

    Applies element-wise transformations to map constrained parameters to
    unconstrained space, proposes there with a Gaussian random walk, and
    maps back. Automatically handles the Jacobian correction.

    Parameters
    ----------
    dim : int
        Dimension of the parameter space.
    transforms : list[dict[str, Any]]
        List of transform specifications, one per dimension. Each dict has:
        - ``'type'``: One of ``'none'``, ``'log'``, ``'logit'``.
        - ``'lower'``: Lower bound (for logit).
        - ``'upper'``: Upper bound (for logit).
    cov : NDArray[np.float64] | float | None
        Covariance in transformed space.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dim: int,
        transforms: list[dict[str, Any]],
        cov: NDArray[np.float64] | float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(dim=dim, seed=seed)
        self.transforms = transforms

        if cov is None:
            scale = (2.38**2) / max(dim, 1)
            self._cov = scale * np.eye(dim)
        elif np.isscalar(cov):
            self._cov = float(cov) * np.eye(dim)  # type: ignore[arg-type]
        else:
            self._cov = np.asarray(cov, dtype=np.float64)

        self._chol = np.linalg.cholesky(self._cov)

    def _to_unconstrained(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map from constrained to unconstrained space."""
        phi = np.empty_like(theta)
        for i, tr in enumerate(self.transforms):
            t = tr.get("type", "none")
            if t == "none":
                phi[i] = theta[i]
            elif t == "log":
                phi[i] = np.log(max(theta[i], 1e-300))
            elif t == "logit":
                lo = tr.get("lower", 0.0)
                hi = tr.get("upper", 1.0)
                x = (theta[i] - lo) / (hi - lo)
                x = np.clip(x, 1e-10, 1 - 1e-10)
                phi[i] = np.log(x / (1 - x))
            else:
                phi[i] = theta[i]
        return phi

    def _to_constrained(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map from unconstrained to constrained space."""
        theta = np.empty_like(phi)
        for i, tr in enumerate(self.transforms):
            t = tr.get("type", "none")
            if t == "none":
                theta[i] = phi[i]
            elif t == "log":
                theta[i] = np.exp(phi[i])
            elif t == "logit":
                lo = tr.get("lower", 0.0)
                hi = tr.get("upper", 1.0)
                s = 1.0 / (1.0 + np.exp(-phi[i]))
                theta[i] = lo + (hi - lo) * s
            else:
                theta[i] = phi[i]
        return theta

    def _log_jacobian(self, theta: NDArray[np.float64]) -> float:
        """Compute log |d(unconstrained)/d(constrained)|."""
        log_jac = 0.0
        for i, tr in enumerate(self.transforms):
            t = tr.get("type", "none")
            if t == "log":
                log_jac -= np.log(max(theta[i], 1e-300))
            elif t == "logit":
                lo = tr.get("lower", 0.0)
                hi = tr.get("upper", 1.0)
                x = (theta[i] - lo) / (hi - lo)
                x = np.clip(x, 1e-10, 1 - 1e-10)
                log_jac -= np.log(hi - lo) + np.log(x) + np.log(1 - x)
        return float(log_jac)

    def propose(
        self,
        theta_current: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Generate proposal in transformed space.

        Parameters
        ----------
        theta_current : NDArray[np.float64]
            Current parameter vector.
        rng : np.random.Generator | None
            Random number generator.

        Returns
        -------
        NDArray[np.float64]
            Proposed parameter vector in constrained space.
        """
        if rng is None:
            rng = self._rng

        phi_current = self._to_unconstrained(theta_current)
        z = rng.standard_normal(self.dim)
        phi_proposed = phi_current + self._chol @ z
        return self._to_constrained(phi_proposed)

    def log_ratio(
        self,
        theta_proposed: NDArray[np.float64],
        theta_current: NDArray[np.float64],
    ) -> float:
        """Log proposal ratio with Jacobian for the transformation.

        Parameters
        ----------
        theta_proposed : NDArray[np.float64]
            Proposed parameter vector.
        theta_current : NDArray[np.float64]
            Current parameter vector.

        Returns
        -------
        float
            Log proposal ratio including Jacobian correction.
        """
        # Jacobian correction: log |J(proposed)| - log |J(current)|
        # Since we propose in unconstrained space and evaluate in constrained,
        # the correction is log |d(constrained)/d(unconstrained)|
        log_jac_proposed = -self._log_jacobian(theta_proposed)
        log_jac_current = -self._log_jacobian(theta_current)
        return log_jac_proposed - log_jac_current
