"""Base class for Sequential Monte Carlo methods.

All SMC methods in particlefilterbox inherit from BaseSMC, which provides
common functionality: particle management, resampling, ESS computation,
and log-evidence accumulation.

References:
    Del Moral, P., Doucet, A. & Jasra, A. (2006). Sequential Monte Carlo
    samplers. JRSS-B, 68(3), 411-436.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from particlefilterbox._logging import get_logger
from particlefilterbox.utils.log_ops import log_mean_exp, log_sum_exp

if TYPE_CHECKING:
    from particlefilterbox.smc.results import SMCResults

logger = get_logger("smc.base")


class BaseSMC(abc.ABC):
    """Abstract base class for all SMC methods.

    Provides shared infrastructure for particle-based sequential methods
    including resampling, ESS computation, and log-evidence tracking.

    Parameters
    ----------
    n_particles : int
        Number of particles (samples) to use.
    resampling_method : str
        Resampling algorithm to use. One of 'systematic', 'multinomial',
        'stratified', 'residual'. Default is 'systematic'.
    ess_threshold : float
        Fraction of n_particles below which resampling is triggered.
        Default is 0.5, meaning resample when ESS < N/2.

    Attributes
    ----------
    n_particles : int
        Number of particles.
    resampling_method : str
        Name of the resampling method.
    ess_threshold : float
        ESS threshold as a fraction of n_particles.
    """

    def __init__(
        self,
        n_particles: int = 1000,
        resampling_method: str = "systematic",
        ess_threshold: float = 0.5,
    ) -> None:
        if n_particles < 2:
            msg = f"n_particles must be >= 2, got {n_particles}"
            raise ValueError(msg)
        if not 0.0 < ess_threshold <= 1.0:
            msg = f"ess_threshold must be in (0, 1], got {ess_threshold}"
            raise ValueError(msg)
        if resampling_method not in (
            "systematic",
            "multinomial",
            "stratified",
            "residual",
        ):
            msg = f"Unknown resampling method: {resampling_method}"
            raise ValueError(msg)

        self.n_particles = n_particles
        self.resampling_method = resampling_method
        self.ess_threshold = ess_threshold

        # Internal state
        self._log_evidence_acc: float = 0.0
        self._ess_history: list[float] = []
        self._acceptance_rates: list[float] = []

    @abc.abstractmethod
    def run(self, data: NDArray[np.floating[Any]] | None = None) -> SMCResults:
        """Run the SMC algorithm.

        Parameters
        ----------
        data : NDArray or None
            Observed data. Some SMC methods (e.g., SMCSampler) do not
            require data, while others (e.g., SMCSquared, IBIS) do.

        Returns
        -------
        SMCResults
            Container with particles, weights, log-evidence, and diagnostics.
        """
        ...

    def _compute_ess(self, log_weights: NDArray[np.floating[Any]]) -> float:
        """Compute Effective Sample Size from log-weights.

        ESS = 1 / sum(w_i^2) where w_i are normalized weights.

        Parameters
        ----------
        log_weights : NDArray, shape (N,)
            Unnormalized log-weights.

        Returns
        -------
        float
            Effective sample size, in [1, N].
        """
        log_w_normalized = log_weights - log_sum_exp(log_weights)
        ess = float(np.exp(-log_sum_exp(2.0 * log_w_normalized)))
        return ess

    def _should_resample(self, log_weights: NDArray[np.floating[Any]]) -> bool:
        """Check if resampling is needed based on ESS threshold.

        Parameters
        ----------
        log_weights : NDArray, shape (N,)
            Unnormalized log-weights.

        Returns
        -------
        bool
            True if ESS < ess_threshold * n_particles.
        """
        ess = self._compute_ess(log_weights)
        return ess < self.ess_threshold * self.n_particles

    def _resample(
        self,
        log_weights: NDArray[np.floating[Any]],
        rng: np.random.Generator,
    ) -> NDArray[np.intp]:
        """Resample particle indices using the configured method.

        Parameters
        ----------
        log_weights : NDArray, shape (N,)
            Unnormalized log-weights.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray[np.intp], shape (N,)
            Resampled indices.
        """
        from particlefilterbox.resampling import get_resampling_fn

        weights = np.exp(log_weights - log_sum_exp(log_weights))
        resample_fn = get_resampling_fn(self.resampling_method)
        indices = resample_fn(weights, rng=rng)
        return indices

    def _resample_move(
        self,
        particles: NDArray[np.floating[Any]],
        log_weights: NDArray[np.floating[Any]],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Resample particles if ESS is below threshold.

        After resampling, weights are reset to uniform (log 1/N).

        Parameters
        ----------
        particles : NDArray, shape (N, k)
            Current particle positions.
        log_weights : NDArray, shape (N,)
            Current unnormalized log-weights.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        tuple of (particles, log_weights)
            Resampled particles with uniform weights, or unchanged if
            resampling was not triggered.
        """
        ess = self._compute_ess(log_weights)
        self._ess_history.append(ess)

        if ess < self.ess_threshold * self.n_particles:
            indices = self._resample(log_weights, rng)
            particles = particles[indices].copy()
            log_weights = np.full(self.n_particles, -np.log(self.n_particles))
            logger.debug("Resampled: ESS was %.1f", ess)

        return particles, log_weights

    def _get_log_evidence(self) -> float:
        """Return the accumulated log-evidence estimate.

        The log marginal likelihood (evidence) is estimated incrementally
        as the sum of log normalizing constant ratios at each step.

        Returns
        -------
        float
            Estimated log p(y) = log Z.
        """
        return self._log_evidence_acc

    def _accumulate_log_evidence(self, log_incremental_weights: NDArray[np.floating[Any]]) -> None:
        """Accumulate one step of log-evidence.

        log Z += log(1/N * sum(exp(log_incremental_weights)))
               = log_mean_exp(log_incremental_weights)

        Parameters
        ----------
        log_incremental_weights : NDArray, shape (N,)
            Log incremental importance weights for this step.
        """
        self._log_evidence_acc += float(log_mean_exp(log_incremental_weights))
