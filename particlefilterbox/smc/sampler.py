"""SMCSampler: General-purpose Sequential Monte Carlo sampler.

Implements the SMC sampler of Del Moral, Doucet & Jasra (2006) with
adaptive tempering schedule. Bridges from a tractable prior to a
complex target distribution through a sequence of intermediate
distributions.

Algorithm:
    1. Initialize particles from prior
    2. For each tempering step:
       a. Find next beta by bisection (maintain target ESS)
       b. Reweight particles
       c. Accumulate log-evidence
       d. Resample if ESS too low
       e. Rejuvenate with MCMC moves

References:
    Del Moral, P., Doucet, A. & Jasra, A. (2006). Sequential Monte Carlo
    samplers. JRSS-B, 68(3), 411-436.
    Neal, R.M. (2001). Annealed importance sampling. Statistics and
    Computing, 11(2), 125-139.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from particlefilterbox._logging import get_logger
from particlefilterbox.smc.base import BaseSMC
from particlefilterbox.smc.mcmc_moves import RandomWalkMH, run_mcmc_chain
from particlefilterbox.smc.results import SMCResults
from particlefilterbox.utils.log_ops import log_sum_exp

logger = get_logger("smc.sampler")


class SMCSampler(BaseSMC):
    """General-purpose SMC sampler with adaptive tempering.

    Samples from a target distribution by bridging from a tractable
    prior through a sequence of tempered distributions:

        pi_n(theta) ~ prior(theta)^(1 - beta_n) * target(theta)^beta_n

    Parameters
    ----------
    target_logpdf : callable
        Log-density of the target distribution (unnormalized OK).
        Signature: target_logpdf(theta: NDArray) -> float
    prior_logpdf : callable
        Log-density of the prior distribution.
        Signature: prior_logpdf(theta: NDArray) -> float
    prior_sample : callable
        Function to draw samples from the prior.
        Signature: prior_sample(rng: Generator) -> NDArray[shape (k,)]
    n_particles : int
        Number of particles. Default 1000.
    n_mcmc_moves : int
        Number of MCMC rejuvenation steps per tempering stage. Default 5.
    ess_target_ratio : float
        Target ESS as fraction of N for adaptive beta schedule.
        Default 0.5.
    resampling_method : str
        Resampling algorithm. Default 'systematic'.
    ess_threshold : float
        ESS threshold for triggering resampling. Default 0.5.
    proposal_cov : NDArray or None
        Proposal covariance for MCMC moves. If None, estimated from
        particle population. Default None.
    seed : int
        Random seed. Default 42.

    Examples
    --------
    >>> import numpy as np
    >>> def log_target(x):
    ...     return -0.5 * np.sum((x - 3)**2)
    >>> def log_prior(x):
    ...     return -0.5 * np.sum(x**2 / 100)
    >>> def sample_prior(rng):
    ...     return rng.standard_normal(2) * 10
    >>> sampler = SMCSampler(
    ...     target_logpdf=log_target,
    ...     prior_logpdf=log_prior,
    ...     prior_sample=sample_prior,
    ...     n_particles=500,
    ... )
    >>> results = sampler.run()
    >>> print(results.posterior_mean())
    """

    def __init__(
        self,
        target_logpdf: Callable[[NDArray[np.floating[Any]]], float],
        prior_logpdf: Callable[[NDArray[np.floating[Any]]], float],
        prior_sample: Callable[[np.random.Generator], NDArray[np.floating[Any]]],
        n_particles: int = 1000,
        n_mcmc_moves: int = 5,
        ess_target_ratio: float = 0.5,
        resampling_method: str = "systematic",
        ess_threshold: float = 0.5,
        proposal_cov: NDArray[np.floating[Any]] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            n_particles=n_particles,
            resampling_method=resampling_method,
            ess_threshold=ess_threshold,
        )
        self.target_logpdf = target_logpdf
        self.prior_logpdf = prior_logpdf
        self.prior_sample = prior_sample
        self.n_mcmc_moves = n_mcmc_moves
        self.ess_target_ratio = ess_target_ratio
        self._proposal_cov = proposal_cov
        self._rng = np.random.default_rng(seed)

        # Storage
        self._schedule: list[float] = []

    def run(self, data: NDArray[np.floating[Any]] | None = None) -> SMCResults:
        """Run the SMC sampler.

        Parameters
        ----------
        data : NDArray or None
            Not used by SMCSampler (target_logpdf already encapsulates
            the data). Kept for API compatibility with BaseSMC.

        Returns
        -------
        SMCResults
            Particles, weights, log-evidence, and diagnostics.
        """
        # Reset internal state
        self._log_evidence_acc = 0.0
        self._ess_history = []
        self._acceptance_rates = []
        self._schedule = [0.0]

        # Step 1: Initialize from prior
        particles = np.array([self.prior_sample(self._rng) for _ in range(self.n_particles)])
        k_params = particles.shape[1]
        log_weights = np.full(self.n_particles, -np.log(self.n_particles))

        # Pre-compute log-densities at initial particles
        log_prior_vals = np.array(
            [self.prior_logpdf(particles[i]) for i in range(self.n_particles)]
        )
        log_target_vals = np.array(
            [self.target_logpdf(particles[i]) for i in range(self.n_particles)]
        )
        log_lik_vals = log_target_vals - log_prior_vals

        beta = 0.0
        step_count = 0

        while beta < 1.0:
            step_count += 1

            # Step 2: Find next beta by bisection
            beta_new = self._compute_schedule(beta, log_lik_vals)
            delta_beta = beta_new - beta

            # Step 3: Reweight
            log_incremental = delta_beta * log_lik_vals
            log_weights = log_weights + log_incremental

            # Step 4: Accumulate log-evidence
            self._accumulate_log_evidence(log_incremental)

            beta = beta_new
            self._schedule.append(beta)

            logger.debug(
                "Step %d: beta=%.4f, ESS=%.1f",
                step_count,
                beta,
                self._compute_ess(log_weights),
            )

            # Step 5: Resample if needed
            particles, log_weights = self._resample_move(particles, log_weights, self._rng)

            # Step 6: MCMC rejuvenation
            if self.n_mcmc_moves > 0:
                particles, log_prior_vals, log_target_vals, acc_rate = self._mcmc_rejuvenate(
                    particles,
                    log_prior_vals,
                    log_target_vals,
                    beta,
                    k_params,
                )
                log_lik_vals = log_target_vals - log_prior_vals
                self._acceptance_rates.append(acc_rate)

            # Safety: max 500 steps
            if step_count > 500:
                logger.warning("Reached max 500 steps, stopping at beta=%.4f", beta)
                break

        # Build results
        weights = np.exp(log_weights - log_sum_exp(log_weights))

        return SMCResults(
            particles=particles,
            weights=weights,
            log_evidence=self._log_evidence_acc,
            schedule=self._schedule.copy(),
            ess_history=self._ess_history.copy(),
            acceptance_rates=self._acceptance_rates.copy(),
            n_steps=step_count,
        )

    def _compute_schedule(
        self,
        beta_current: float,
        log_lik_vals: NDArray[np.floating[Any]],
    ) -> float:
        """Find next beta by bisection to maintain target ESS.

        Solves for beta_new such that the ESS after reweighting with
        (beta_new - beta_current) * log_lik is approximately
        ess_target_ratio * N.

        Parameters
        ----------
        beta_current : float
            Current beta value.
        log_lik_vals : NDArray, shape (N,)
            Log-likelihood values at current particles.

        Returns
        -------
        float
            Next beta value in (beta_current, 1.0].
        """
        target_ess = self.ess_target_ratio * self.n_particles

        # Check if we can jump straight to beta=1
        delta_1 = 1.0 - beta_current
        log_inc = delta_1 * log_lik_vals
        log_w_test = log_inc - log_sum_exp(log_inc)
        ess_at_1 = float(np.exp(-log_sum_exp(2.0 * log_w_test)))

        if ess_at_1 >= target_ess:
            return 1.0

        # Bisection between beta_current and 1.0
        lo = beta_current
        hi = 1.0

        for _ in range(50):  # max bisection iterations
            mid = 0.5 * (lo + hi)
            delta = mid - beta_current
            log_inc = delta * log_lik_vals
            log_w_test = log_inc - log_sum_exp(log_inc)
            ess_mid = float(np.exp(-log_sum_exp(2.0 * log_w_test)))

            if ess_mid > target_ess:
                lo = mid
            else:
                hi = mid

            if hi - lo < 1e-6:
                break

        return lo

    def _incremental_weights(
        self,
        delta_beta: float,
        log_lik_vals: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Compute log incremental weights for a beta increment.

        Parameters
        ----------
        delta_beta : float
            Increment in beta.
        log_lik_vals : NDArray, shape (N,)
            Log-likelihood values.

        Returns
        -------
        NDArray, shape (N,)
            Log incremental importance weights.
        """
        return delta_beta * log_lik_vals

    def _mcmc_rejuvenate(
        self,
        particles: NDArray[np.floating[Any]],
        log_prior_vals: NDArray[np.floating[Any]],
        log_target_vals: NDArray[np.floating[Any]],
        beta: float,
        k_params: int,
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        float,
    ]:
        """Rejuvenate particles with MCMC moves targeting pi_beta.

        Parameters
        ----------
        particles : NDArray, shape (N, k)
            Current particles.
        log_prior_vals : NDArray, shape (N,)
            Log-prior at each particle.
        log_target_vals : NDArray, shape (N,)
            Log-target at each particle.
        beta : float
            Current tempering parameter.
        k_params : int
            Number of parameters.

        Returns
        -------
        tuple of (particles, log_prior_vals, log_target_vals, acceptance_rate)
        """
        # Estimate proposal covariance from current particles
        if self._proposal_cov is not None:
            cov = self._proposal_cov
        else:
            cov = np.cov(particles.T) + 1e-6 * np.eye(k_params)
            cov *= 2.38**2 / k_params  # Optimal scaling

        kernel = RandomWalkMH(proposal_cov=cov)

        total_accepted = 0
        total_steps = 0

        for i in range(self.n_particles):

            def log_pi_beta(theta: NDArray[np.floating[Any]], _beta: float = beta) -> float:
                lp = self.prior_logpdf(theta)
                lt = self.target_logpdf(theta)
                return lt if _beta >= 1.0 else (1.0 - _beta) * lp + _beta * lt

            theta_new, acc_rate = run_mcmc_chain(
                theta_init=particles[i],
                log_target=log_pi_beta,
                kernel=kernel,
                rng=self._rng,
                n_steps=self.n_mcmc_moves,
            )

            particles[i] = theta_new
            log_prior_vals[i] = self.prior_logpdf(theta_new)
            log_target_vals[i] = self.target_logpdf(theta_new)
            total_accepted += int(acc_rate * self.n_mcmc_moves)
            total_steps += self.n_mcmc_moves

        overall_rate = total_accepted / total_steps if total_steps > 0 else 0.0
        return particles, log_prior_vals, log_target_vals, overall_rate
