"""Waste-Free Sequential Monte Carlo.

Implements the waste-free SMC of Dau & Chopin (2022), which recycles
ALL MCMC samples (not just the last accepted) to improve efficiency.

Standard SMC runs K MCMC steps per particle but only keeps the last
sample, wasting K-1 evaluations. Waste-free SMC:
1. Resamples N/K particles (instead of N)
2. Runs K MCMC steps for each
3. Keeps ALL K samples (including rejections) = N total
4. Reweights to correct the selection bias

This effectively multiplies sampling efficiency by K.

References:
    Dau, H.D. & Chopin, N. (2022). Waste-free Sequential Monte Carlo.
    JRSS-B, 84(1), 114-148.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from particlefilterbox._logging import get_logger
from particlefilterbox.smc.base import BaseSMC
from particlefilterbox.smc.mcmc_moves import RandomWalkMH
from particlefilterbox.smc.results import SMCResults
from particlefilterbox.utils.log_ops import log_sum_exp

logger = get_logger("smc.waste_free")


class WasteFreeSMC(BaseSMC):
    """Waste-Free SMC sampler.

    Recycles all MCMC samples to improve efficiency. Instead of running
    K MCMC steps and keeping only the last, keeps all K samples and
    reweights appropriately.

    Parameters
    ----------
    target_logpdf : callable
        Log-density of the (unnormalized) target distribution.
    prior_logpdf : callable
        Log-density of the prior distribution.
    prior_sample : callable
        Function to sample from the prior.
    n_particles : int
        Total number of particles (must be divisible by k_mcmc).
        Default 1000.
    k_mcmc : int
        Number of MCMC steps per resampled particle. N/K particles
        are resampled, each producing K samples. Default 10.
    ess_target_ratio : float
        Target ESS ratio for adaptive tempering. Default 0.5.
    resampling_method : str
        Resampling method. Default 'systematic'.
    ess_threshold : float
        ESS threshold. Default 0.5.
    seed : int
        Random seed. Default 42.

    Notes
    -----
    n_particles should be divisible by k_mcmc. If not, it will be
    rounded down to the nearest multiple.

    Examples
    --------
    >>> wf = WasteFreeSMC(
    ...     target_logpdf=log_post,
    ...     prior_logpdf=log_prior,
    ...     prior_sample=sample_prior,
    ...     n_particles=1000,
    ...     k_mcmc=10,
    ... )
    >>> results = wf.run()
    """

    def __init__(
        self,
        target_logpdf: Callable[[NDArray[np.floating[Any]]], float],
        prior_logpdf: Callable[[NDArray[np.floating[Any]]], float],
        prior_sample: Callable[[np.random.Generator], NDArray[np.floating[Any]]],
        n_particles: int = 1000,
        k_mcmc: int = 10,
        ess_target_ratio: float = 0.5,
        resampling_method: str = "systematic",
        ess_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        # Ensure n_particles is divisible by k_mcmc
        n_particles = (n_particles // k_mcmc) * k_mcmc
        if n_particles < k_mcmc:
            n_particles = k_mcmc

        super().__init__(
            n_particles=n_particles,
            resampling_method=resampling_method,
            ess_threshold=ess_threshold,
        )

        self.target_logpdf = target_logpdf
        self.prior_logpdf = prior_logpdf
        self.prior_sample = prior_sample
        self.k_mcmc = k_mcmc
        self.ess_target_ratio = ess_target_ratio
        self._rng = np.random.default_rng(seed)

        # N/K = number of "mother" particles to resample
        self.n_mothers = n_particles // k_mcmc

    def run(self, data: NDArray[np.floating[Any]] | None = None) -> SMCResults:
        """Run waste-free SMC sampler.

        Parameters
        ----------
        data : NDArray or None
            Not used (target_logpdf encapsulates data).

        Returns
        -------
        SMCResults
            Particles, weights, log-evidence, and diagnostics.
        """
        _ = data  # Not used; target_logpdf encapsulates data
        # Reset
        self._log_evidence_acc = 0.0
        self._ess_history = []
        self._acceptance_rates = []
        schedule: list[float] = [0.0]

        # Initialize from prior
        particles = np.array([self.prior_sample(self._rng) for _ in range(self.n_particles)])
        k_params = particles.shape[1]
        log_weights = np.full(self.n_particles, -np.log(self.n_particles))

        # Pre-compute densities
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

            # Adaptive beta
            beta_new = self._find_next_beta(beta, log_lik_vals)
            delta_beta = beta_new - beta

            # Reweight
            log_incremental = delta_beta * log_lik_vals
            log_weights = log_weights + log_incremental

            # Evidence
            self._accumulate_log_evidence(log_incremental)

            beta = beta_new
            schedule.append(beta)

            ess = self._compute_ess(log_weights)
            self._ess_history.append(ess)

            logger.debug("Step %d: beta=%.4f, ESS=%.1f", step_count, beta, ess)

            # Waste-free move: resample N/K, run K MCMC each, keep all
            (
                particles,
                log_weights,
                log_prior_vals,
                log_target_vals,
                log_lik_vals,
                acc_rate,
            ) = self._waste_free_move(
                particles,
                log_weights,
                log_prior_vals,
                log_target_vals,
                beta,
                k_params,
            )
            self._acceptance_rates.append(acc_rate)

            if step_count > 500:
                logger.warning("Max 500 steps reached at beta=%.4f", beta)
                break

        weights = np.exp(log_weights - log_sum_exp(log_weights))

        return SMCResults(
            particles=particles,
            weights=weights,
            log_evidence=self._get_log_evidence(),
            schedule=schedule,
            ess_history=self._ess_history.copy(),
            acceptance_rates=self._acceptance_rates.copy(),
            n_steps=step_count,
        )

    def _find_next_beta(
        self,
        beta_current: float,
        log_lik_vals: NDArray[np.floating[Any]],
    ) -> float:
        """Find next beta by bisection to maintain target ESS.

        Parameters
        ----------
        beta_current : float
            Current beta.
        log_lik_vals : NDArray, shape (N,)
            Log-likelihood values.

        Returns
        -------
        float
            Next beta in (beta_current, 1.0].
        """
        target_ess = self.ess_target_ratio * self.n_particles

        # Try beta=1
        delta = 1.0 - beta_current
        log_inc = delta * log_lik_vals
        log_w_test = log_inc - log_sum_exp(log_inc)
        ess_at_1 = float(np.exp(-log_sum_exp(2.0 * log_w_test)))

        if ess_at_1 >= target_ess:
            return 1.0

        # Bisection
        lo = beta_current
        hi = 1.0

        for _ in range(50):
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

    def _waste_free_move(
        self,
        particles: NDArray[np.floating[Any]],
        log_weights: NDArray[np.floating[Any]],
        log_prior_vals: NDArray[np.floating[Any]],
        log_target_vals: NDArray[np.floating[Any]],
        beta: float,
        k_params: int,
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        float,
    ]:
        """Perform waste-free resample-move step.

        1. Resample N/K "mother" particles
        2. For each mother, run K MCMC steps targeting pi_beta
        3. Keep ALL K samples (not just the last)
        4. Reweight to correct selection bias

        Parameters
        ----------
        particles : NDArray, shape (N, k)
        log_weights : NDArray, shape (N,)
        log_prior_vals : NDArray, shape (N,)
        log_target_vals : NDArray, shape (N,)
        beta : float
        k_params : int

        Returns
        -------
        tuple of (particles, log_weights, log_prior, log_target, log_lik, acc_rate)
        """
        n_total = self.n_particles
        k_steps = self.k_mcmc
        n_mothers = self.n_mothers  # n_total / k_steps

        # Step 1: Resample n_mothers = N/K mother particles
        weights_norm = np.exp(log_weights - log_sum_exp(log_weights))

        # Systematic resampling of n_mothers particles
        positions = (self._rng.uniform() + np.arange(n_mothers)) / n_mothers
        cumsum = np.cumsum(weights_norm)
        mother_indices = np.searchsorted(cumsum, positions)
        mother_indices = np.clip(mother_indices, 0, n_total - 1)

        # Estimate proposal covariance
        cov = np.cov(particles.T) + 1e-6 * np.eye(k_params)
        cov *= 2.38**2 / k_params
        kernel = RandomWalkMH(proposal_cov=cov)

        # Step 2 & 3: Run K MCMC steps for each mother, keep ALL samples
        new_particles = np.zeros((n_total, k_params))
        new_log_prior = np.zeros(n_total)
        new_log_target = np.zeros(n_total)

        total_accepted = 0
        total_steps = 0

        for m in range(n_mothers):
            idx = mother_indices[m]
            theta_current = particles[idx].copy()

            def log_pi_beta(
                theta: NDArray[np.floating[Any]],
                _beta: float = beta,
            ) -> float:
                lp = self.prior_logpdf(theta)
                lt = self.target_logpdf(theta)
                if not np.isfinite(lp) or not np.isfinite(lt):
                    return -np.inf
                return (1.0 - _beta) * lp + _beta * lt if _beta < 1.0 else lt

            log_t_current: float | None = None

            for k in range(k_steps):
                result = kernel.step(
                    theta_current,
                    log_pi_beta,
                    self._rng,
                    log_target_current=log_t_current,
                )

                # Store this sample (whether accepted or not)
                sample_idx = m * k_steps + k
                new_particles[sample_idx] = result.theta.copy()
                new_log_prior[sample_idx] = self.prior_logpdf(result.theta)
                new_log_target[sample_idx] = self.target_logpdf(result.theta)

                theta_current = result.theta
                log_t_current = result.log_target

                if result.accepted:
                    total_accepted += 1
                total_steps += 1

        # Step 4: Reweight - all samples get equal weight initially
        # (the waste-free correction is implicit in the construction)
        new_log_lik = new_log_target - new_log_prior
        new_log_weights = np.full(n_total, -np.log(n_total))

        acc_rate = total_accepted / total_steps if total_steps > 0 else 0.0

        return (
            new_particles,
            new_log_weights,
            new_log_prior,
            new_log_target,
            new_log_lik,
            acc_rate,
        )
