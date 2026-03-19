"""IBIS: Iterated Batch Importance Sampling.

Implements the IBIS algorithm of Chopin (2002) for sequential parameter
inference in models where the marginal likelihood p(y_{1:t}|theta) is
computable directly (without an internal particle filter).

IBIS processes data in batches, reweighting particles and rejuvenating
via MCMC when the ESS drops below a threshold.

Algorithm:
    INIT: theta^(i) ~ prior, w^(i) = 1/N
    FOR batch b=1..B:
      1. Compute incremental log-likelihood for each theta
      2. REWEIGHT: log_w += log p(y_new | y_old, theta)
      3. Accumulate log-evidence
      4. If ESS < threshold:
         a. RESAMPLE
         b. MCMC REJUVENATION: K steps targeting current posterior
      5. Update sufficient statistics (optional)

References:
    Chopin, N. (2002). A sequential particle filter method for static
    models. Biometrika, 89(3), 539-552.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from particlefilterbox._logging import get_logger
from particlefilterbox.smc.base import BaseSMC
from particlefilterbox.smc.mcmc_moves import RandomWalkMH, run_mcmc_chain
from particlefilterbox.smc.results import SMCResults
from particlefilterbox.utils.log_ops import log_sum_exp

logger = get_logger("smc.ibis")


class IBISModel(Protocol):
    """Protocol for models compatible with IBIS.

    The model must provide incremental log-likelihood computation,
    meaning the ability to compute log p(y_new | y_old, theta) as
    new data arrives.
    """

    def log_likelihood(
        self,
        theta: NDArray[np.floating[Any]],
        endog: NDArray[np.floating[Any]],
    ) -> float:
        """Log-likelihood of all data given parameters.

        Parameters
        ----------
        theta : NDArray, shape (k,)
            Parameters.
        endog : NDArray, shape (T, k_obs)
            All observed data up to current point.

        Returns
        -------
        float
            log p(y_{1:T} | theta)
        """
        ...


class IBISPrior(Protocol):
    """Protocol for IBIS prior distributions."""

    def logpdf(self, theta: NDArray[np.floating[Any]]) -> float:
        """Log-density of the prior."""
        ...

    def sample(self, rng: np.random.Generator) -> NDArray[np.floating[Any]]:
        """Draw one sample from the prior."""
        ...


class IBIS(BaseSMC):
    """Iterated Batch Importance Sampling.

    Processes data sequentially (one observation or batch at a time),
    maintaining a weighted particle approximation to the posterior.
    When particle weights become too uneven (low ESS), resamples
    and rejuvenates via MCMC.

    Parameters
    ----------
    model : IBISModel
        Model with log_likelihood(theta, endog) method.
    n_particles : int
        Number of parameter particles. Default 1000.
    prior : IBISPrior
        Prior distribution with logpdf() and sample() methods.
    n_mcmc_moves : int
        MCMC rejuvenation steps when triggered. Default 5.
    batch_size : int
        Number of observations per batch. Default 1 (fully sequential).
    resampling_method : str
        Resampling method. Default 'systematic'.
    ess_threshold : float
        ESS threshold for triggering rejuvenation. Default 0.5.
    seed : int
        Random seed. Default 42.

    Examples
    --------
    >>> ibis = IBIS(model=my_model, n_particles=1000, prior=my_prior)
    >>> results = ibis.run(endog=observations)
    >>> print(results.summary())
    """

    def __init__(
        self,
        model: Any,
        n_particles: int = 1000,
        prior: Any = None,
        n_mcmc_moves: int = 5,
        batch_size: int = 1,
        resampling_method: str = "systematic",
        ess_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__(
            n_particles=n_particles,
            resampling_method=resampling_method,
            ess_threshold=ess_threshold,
        )
        self.model = model
        self.prior = prior
        self.n_mcmc_moves = n_mcmc_moves
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)

        # Sufficient statistics storage
        self._sufficient_stats: dict[int, Any] = {}

    def run(
        self,
        data: NDArray[np.floating[Any]] | None = None,
        endog: NDArray[np.floating[Any]] | None = None,
    ) -> SMCResults:
        """Run IBIS on sequential observations.

        Parameters
        ----------
        data : NDArray or None
            Observed data, shape (T, k_obs). Alias for endog.
        endog : NDArray or None
            Observed data, shape (T, k_obs).

        Returns
        -------
        SMCResults
            Posterior particles, weights, and log-evidence.
        """
        if endog is None:
            endog = data
        if endog is None:
            msg = "Either data or endog must be provided"
            raise ValueError(msg)

        endog = np.atleast_2d(endog)
        if endog.ndim == 1:
            endog = endog[:, np.newaxis]

        n_obs = endog.shape[0]

        # Reset
        self._log_evidence_acc = 0.0
        self._ess_history = []
        self._acceptance_rates = []
        self._sufficient_stats = {}

        # Initialize from prior
        particles = np.array([self.prior.sample(self._rng) for _ in range(self.n_particles)])
        k_params = particles.shape[1] if particles.ndim > 1 else 1
        if particles.ndim == 1:
            particles = particles[:, np.newaxis]

        log_weights = np.full(self.n_particles, -np.log(self.n_particles))

        # Track cumulative log-likelihoods for each particle
        cum_log_lik = np.zeros(self.n_particles)

        # Process data in batches
        n_batches = (n_obs + self.batch_size - 1) // self.batch_size
        step_count = 0

        for b in range(n_batches):
            end = min((b + 1) * self.batch_size, n_obs)
            endog_so_far = endog[:end]

            step_count += 1

            # Compute incremental log-likelihood for each particle
            log_lik_increments = self._incremental_loglike(particles, cum_log_lik, endog_so_far)

            # Update cumulative log-likelihoods
            cum_log_lik = cum_log_lik + log_lik_increments

            # Reweight
            log_weights = log_weights + log_lik_increments

            # Accumulate evidence
            self._accumulate_log_evidence(log_lik_increments)

            # Check ESS
            ess = self._compute_ess(log_weights)
            self._ess_history.append(ess)

            logger.debug("Batch %d/%d: ESS=%.1f", b + 1, n_batches, ess)

            # Rejuvenate if needed
            if ess < self.ess_threshold * self.n_particles:
                # Resample
                indices = self._resample(log_weights, self._rng)
                particles = particles[indices].copy()
                cum_log_lik = cum_log_lik[indices].copy()
                log_weights = np.full(self.n_particles, -np.log(self.n_particles))

                # MCMC rejuvenation targeting current posterior
                particles, cum_log_lik, acc_rate = self._rejuvenate(
                    particles, cum_log_lik, endog_so_far, k_params
                )
                self._acceptance_rates.append(acc_rate)

                logger.debug("Rejuvenated: acc_rate=%.3f", acc_rate)

            # Update sufficient stats
            self._update_sufficient_stats(b, particles, log_weights)

        weights = np.exp(log_weights - log_sum_exp(log_weights))

        return SMCResults(
            particles=particles,
            weights=weights,
            log_evidence=self._get_log_evidence(),
            ess_history=self._ess_history.copy(),
            acceptance_rates=self._acceptance_rates.copy(),
            n_steps=step_count,
        )

    def _incremental_loglike(
        self,
        particles: NDArray[np.floating[Any]],
        cum_log_lik: NDArray[np.floating[Any]],
        endog_so_far: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Compute incremental log-likelihood for each particle.

        log p(y_new | y_old, theta) = log p(y_{1:t} | theta) - log p(y_{1:t-1} | theta)

        Parameters
        ----------
        particles : NDArray, shape (N, k)
            Parameter particles.
        cum_log_lik : NDArray, shape (N,)
            Cumulative log-likelihoods up to previous batch.
        endog_so_far : NDArray, shape (t, k_obs)
            All observations up to current batch.

        Returns
        -------
        NDArray, shape (N,)
            Incremental log-likelihoods.
        """
        increments = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            new_log_lik = self.model.log_likelihood(particles[i], endog_so_far)
            increments[i] = new_log_lik - cum_log_lik[i]
        return increments

    def _rejuvenate(
        self,
        particles: NDArray[np.floating[Any]],
        cum_log_lik: NDArray[np.floating[Any]],
        endog_so_far: NDArray[np.floating[Any]],
        k_params: int,
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        float,
    ]:
        """MCMC rejuvenation targeting current posterior.

        Parameters
        ----------
        particles : NDArray, shape (N, k)
            Parameter particles.
        cum_log_lik : NDArray, shape (N,)
            Cumulative log-likelihoods.
        endog_so_far : NDArray
            Observations processed so far.
        k_params : int
            Number of parameters.

        Returns
        -------
        tuple of (particles, cum_log_lik, acceptance_rate)
        """
        cov = np.cov(particles.T) + 1e-6 * np.eye(k_params)
        cov *= 2.38**2 / k_params
        kernel = RandomWalkMH(proposal_cov=cov)

        total_accepted = 0
        total_steps = 0

        for i in range(self.n_particles):

            def log_posterior(theta: NDArray[np.floating[Any]]) -> float:
                lp = self.prior.logpdf(theta)
                if not np.isfinite(lp):
                    return -np.inf
                ll = self.model.log_likelihood(theta, endog_so_far)
                return lp + ll

            theta_new, acc_rate = run_mcmc_chain(
                theta_init=particles[i],
                log_target=log_posterior,
                kernel=kernel,
                rng=self._rng,
                n_steps=self.n_mcmc_moves,
            )

            particles[i] = theta_new
            cum_log_lik[i] = self.model.log_likelihood(theta_new, endog_so_far)
            total_accepted += int(acc_rate * self.n_mcmc_moves)
            total_steps += self.n_mcmc_moves

        overall_rate = total_accepted / total_steps if total_steps > 0 else 0.0
        return particles, cum_log_lik, overall_rate

    def _update_sufficient_stats(
        self,
        batch_idx: int,
        particles: NDArray[np.floating[Any]],
        log_weights: NDArray[np.floating[Any]],
    ) -> None:
        """Update sufficient statistics for diagnostics.

        Parameters
        ----------
        batch_idx : int
            Current batch index.
        particles : NDArray, shape (N, k)
            Current particles.
        log_weights : NDArray, shape (N,)
            Current log-weights.
        """
        weights = np.exp(log_weights - log_sum_exp(log_weights))
        self._sufficient_stats[batch_idx] = {
            "mean": np.average(particles, weights=weights, axis=0),
            "ess": self._compute_ess(log_weights),
        }
