"""Tempering: SMC for Bayesian inference via likelihood annealing.

Specialization of the SMC sampler for Bayesian models where the bridge
is constructed by tempering the likelihood:

    pi_t(theta) = prior(theta) * likelihood(theta|y)^gamma_t

This provides estimates of the marginal likelihood (Bayes factor)
and posterior samples simultaneously.

References:
    Del Moral, P., Doucet, A. & Jasra, A. (2006). Sequential Monte Carlo
    samplers. JRSS-B, 68(3), 411-436.
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

logger = get_logger("smc.tempering")


class BayesianModel(Protocol):
    """Protocol for Bayesian models compatible with Tempering."""

    def log_likelihood(
        self,
        theta: NDArray[np.floating[Any]],
        endog: NDArray[np.floating[Any]],
    ) -> float:
        """Log-likelihood of data given parameters."""
        ...


class Prior(Protocol):
    """Protocol for prior distributions."""

    def logpdf(self, theta: NDArray[np.floating[Any]]) -> float:
        """Log-density of the prior."""
        ...

    def sample(self, rng: np.random.Generator) -> NDArray[np.floating[Any]]:
        """Sample from the prior."""
        ...


class Tempering(BaseSMC):
    """SMC Tempering for Bayesian inference.

    Constructs a sequence of tempered distributions:

        pi_t(theta) = prior(theta) * likelihood(theta|y)^gamma_t

    where 0 = gamma_0 < gamma_1 < ... < gamma_P = 1, with adaptive
    gamma schedule to maintain ESS near a target.

    Parameters
    ----------
    model : BayesianModel
        Model with log_likelihood(theta, endog) method.
    prior : Prior
        Prior with logpdf(theta) and sample(rng) methods.
    n_particles : int
        Number of particles. Default 1000.
    n_mcmc_moves : int
        MCMC rejuvenation steps per stage. Default 5.
    ess_target_ratio : float
        Target ESS ratio for adaptive gamma. Default 0.5.
    resampling_method : str
        Resampling method. Default 'systematic'.
    ess_threshold : float
        ESS threshold for resampling. Default 0.5.
    seed : int
        Random seed. Default 42.

    Examples
    --------
    >>> tempering = Tempering(model=my_model, prior=my_prior, n_particles=1000)
    >>> results = tempering.run(endog=observations)
    >>> print(f"Log Bayes factor: {results.log_evidence:.2f}")
    """

    def __init__(
        self,
        model: Any,
        prior: Any,
        n_particles: int = 1000,
        n_mcmc_moves: int = 5,
        ess_target_ratio: float = 0.5,
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
        self.ess_target_ratio = ess_target_ratio
        self._rng = np.random.default_rng(seed)
        self._schedule: list[float] = []

    def run(
        self,
        data: NDArray[np.floating[Any]] | None = None,
        endog: NDArray[np.floating[Any]] | None = None,
    ) -> SMCResults:
        """Run the tempering SMC algorithm.

        Parameters
        ----------
        data : NDArray or None
            Observed data (alias for endog).
        endog : NDArray or None
            Observed data. Either data or endog must be provided.

        Returns
        -------
        SMCResults
            Posterior samples, weights, log-evidence.
        """
        if endog is None:
            endog = data
        if endog is None:
            msg = "Either data or endog must be provided"
            raise ValueError(msg)

        # Reset
        self._log_evidence_acc = 0.0
        self._ess_history = []
        self._acceptance_rates = []
        self._schedule = [0.0]

        # Initialize from prior
        particles = np.array([self.prior.sample(self._rng) for _ in range(self.n_particles)])
        k_params = particles.shape[1] if particles.ndim > 1 else 1
        if particles.ndim == 1:
            particles = particles[:, np.newaxis]

        log_weights = np.full(self.n_particles, -np.log(self.n_particles))

        # Compute log-likelihoods
        log_lik_vals = np.array(
            [self.model.log_likelihood(particles[i], endog) for i in range(self.n_particles)]
        )

        gamma = 0.0
        step_count = 0

        while gamma < 1.0:
            step_count += 1

            # Adaptive gamma
            gamma_new = self._adaptive_gamma(gamma, log_lik_vals)
            delta_gamma = gamma_new - gamma

            # Reweight
            log_incremental = delta_gamma * log_lik_vals
            log_weights = log_weights + log_incremental

            # Accumulate evidence
            self._accumulate_log_evidence(log_incremental)

            gamma = gamma_new
            self._schedule.append(gamma)

            logger.debug(
                "Step %d: gamma=%.4f, ESS=%.1f",
                step_count,
                gamma,
                self._compute_ess(log_weights),
            )

            # Resample if needed
            particles, log_weights = self._resample_move(particles, log_weights, self._rng)

            # MCMC rejuvenation
            if self.n_mcmc_moves > 0:
                particles, log_lik_vals, acc_rate = self._rejuvenate(
                    particles, log_lik_vals, gamma, endog, k_params
                )
                self._acceptance_rates.append(acc_rate)

            if step_count > 500:
                logger.warning("Reached max 500 steps, stopping at gamma=%.4f", gamma)
                break

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

    def _adaptive_gamma(
        self,
        gamma_current: float,
        log_lik_vals: NDArray[np.floating[Any]],
    ) -> float:
        """Find next gamma by bisection to maintain target ESS.

        Parameters
        ----------
        gamma_current : float
            Current annealing parameter.
        log_lik_vals : NDArray, shape (N,)
            Log-likelihood values.

        Returns
        -------
        float
            Next gamma in (gamma_current, 1.0].
        """
        target_ess = self.ess_target_ratio * self.n_particles

        # Try jumping to gamma=1
        delta = 1.0 - gamma_current
        log_inc = delta * log_lik_vals
        log_w_test = log_inc - log_sum_exp(log_inc)
        ess_at_1 = float(np.exp(-log_sum_exp(2.0 * log_w_test)))

        if ess_at_1 >= target_ess:
            return 1.0

        # Bisection
        lo = gamma_current
        hi = 1.0

        for _ in range(50):
            mid = 0.5 * (lo + hi)
            delta = mid - gamma_current
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

    def _rejuvenate(
        self,
        particles: NDArray[np.floating[Any]],
        log_lik_vals: NDArray[np.floating[Any]],
        gamma: float,
        endog: NDArray[np.floating[Any]],
        k_params: int,
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        float,
    ]:
        """Rejuvenate particles via MCMC targeting pi_gamma.

        Parameters
        ----------
        particles : NDArray, shape (N, k)
            Current particles.
        log_lik_vals : NDArray, shape (N,)
            Log-likelihood at each particle.
        gamma : float
            Current annealing parameter.
        endog : NDArray
            Observed data.
        k_params : int
            Number of parameters.

        Returns
        -------
        tuple of (particles, log_lik_vals, acceptance_rate)
        """
        cov = np.cov(particles.T) + 1e-6 * np.eye(k_params)
        cov *= 2.38**2 / k_params
        kernel = RandomWalkMH(proposal_cov=cov)

        total_accepted = 0
        total_steps = 0

        for i in range(self.n_particles):

            def log_pi_gamma(
                theta: NDArray[np.floating[Any]],
                _gamma: float = gamma,
                _endog: NDArray[np.floating[Any]] = endog,
            ) -> float:
                lp = self.prior.logpdf(theta)
                ll = self.model.log_likelihood(theta, _endog)
                return lp + _gamma * ll

            theta_new, acc_rate = run_mcmc_chain(
                theta_init=particles[i],
                log_target=log_pi_gamma,
                kernel=kernel,
                rng=self._rng,
                n_steps=self.n_mcmc_moves,
            )

            particles[i] = theta_new
            log_lik_vals[i] = self.model.log_likelihood(theta_new, endog)
            total_accepted += int(acc_rate * self.n_mcmc_moves)
            total_steps += self.n_mcmc_moves

        overall_rate = total_accepted / total_steps if total_steps > 0 else 0.0
        return particles, log_lik_vals, overall_rate

    def log_bayes_factor(self) -> float:
        """Return the log Bayes factor (log marginal likelihood).

        This is the log-evidence accumulated during the run. Can be used
        to compare models via Bayes factors:
            log BF(M1 vs M2) = log Z_1 - log Z_2

        Returns
        -------
        float
            Log marginal likelihood estimate.
        """
        return self._log_evidence_acc
