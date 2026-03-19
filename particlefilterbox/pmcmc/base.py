"""Base class for Particle MCMC methods.

All PMCMC methods in particlefilterbox inherit from BasePMCMC, which provides
common functionality: prior evaluation, particle filter likelihood estimation,
iteration storage, and chain management.

References:
    Andrieu, C., Doucet, A. & Holenstein, R. (2010). Particle Markov chain
    Monte Carlo methods. JRSS-B, 72(3), 269-342.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from particlefilterbox.pmcmc.results import PMCMCResults

__all__ = ["BasePMCMC"]


class BasePMCMC(abc.ABC):
    """Abstract base class for all Particle MCMC methods.

    Provides shared infrastructure for PMCMC samplers including prior
    evaluation, particle filter likelihood estimation, and chain storage.

    Parameters
    ----------
    model : Any
        State-space model with methods for simulation and likelihood.
        Must support parameter updates and provide a particle filter interface.
    prior : Any
        Prior distribution object. Must implement ``logpdf(theta)`` method
        returning the log-prior density, and optionally ``sample(rng)`` for
        initialization.
    n_particles : int
        Number of particles used in the particle filter for likelihood
        estimation. More particles reduce variance of the likelihood estimate
        but increase computational cost.
    n_iterations : int
        Total number of MCMC iterations to run (including burn-in).
    burnin : int
        Number of initial iterations to discard as burn-in. Default is
        ``n_iterations // 2``.
    thin : int
        Thinning factor. Only every ``thin``-th iteration is stored.
        Default is 1 (no thinning).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model: Any,
        prior: Any,
        n_particles: int = 200,
        n_iterations: int = 5000,
        burnin: int | None = None,
        thin: int = 1,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.prior = prior
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.burnin = burnin if burnin is not None else n_iterations // 2
        self.thin = thin
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Storage -- initialized in run()
        self._chains: list[NDArray[np.float64]] = []
        self._log_likelihoods: list[float] = []
        self._acceptance_history: list[bool] = []
        self._param_names: list[str] | None = None

    @abc.abstractmethod
    def run(self, endog: NDArray[np.float64], **kwargs: Any) -> PMCMCResults:
        """Run the PMCMC sampler.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observed time series data of shape ``(T,)`` or ``(T, d_y)``.
        **kwargs : Any
            Additional keyword arguments for specific implementations.

        Returns
        -------
        PMCMCResults
            Container with posterior samples and diagnostics.
        """
        ...

    def _log_prior(self, theta: NDArray[np.float64]) -> float:
        """Evaluate the log-prior density at theta.

        Parameters
        ----------
        theta : NDArray[np.float64]
            Parameter vector of shape ``(k,)``.

        Returns
        -------
        float
            Log-prior density. Returns ``-inf`` if theta is outside support.
        """
        try:
            lp = self.prior.logpdf(theta)
            if np.isnan(lp):
                return -np.inf
            return float(lp)
        except (ValueError, RuntimeError):
            return -np.inf

    def _pf_loglike(
        self,
        theta: NDArray[np.float64],
        endog: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> float:
        """Estimate log-likelihood using a particle filter.

        Runs a bootstrap particle filter with ``n_particles`` particles
        and the given parameters to obtain an unbiased estimate of the
        log marginal likelihood p(y_{1:T} | theta).

        Parameters
        ----------
        theta : NDArray[np.float64]
            Parameter vector of shape ``(k,)``.
        endog : NDArray[np.float64]
            Observed data of shape ``(T,)`` or ``(T, d_y)``.
        rng : np.random.Generator | None
            Random number generator. Uses internal RNG if None.

        Returns
        -------
        float
            Estimated log marginal likelihood.
        """
        if rng is None:
            rng = self._rng

        # Update model parameters
        self.model.set_params(theta)

        # Run particle filter and get log-likelihood estimate
        pf_result = self.model.filter(
            endog=endog,
            n_particles=self.n_particles,
            rng=rng,
        )

        ll = pf_result.log_likelihood
        if np.isnan(ll):
            return -np.inf
        return float(ll)

    def _store_iteration(
        self,
        theta: NDArray[np.float64],
        log_likelihood: float,
        accepted: bool,
    ) -> None:
        """Store results from one MCMC iteration.

        Parameters
        ----------
        theta : NDArray[np.float64]
            Current parameter vector of shape ``(k,)``.
        log_likelihood : float
            Log-likelihood at current theta.
        accepted : bool
            Whether the proposal was accepted at this iteration.
        """
        self._chains.append(theta.copy())
        self._log_likelihoods.append(log_likelihood)
        self._acceptance_history.append(accepted)

    def _build_results(self) -> PMCMCResults:
        """Build PMCMCResults from stored chain data.

        Returns
        -------
        PMCMCResults
            Container with posterior samples and diagnostics.
        """
        from particlefilterbox.pmcmc.results import PMCMCResults

        chains = np.array(self._chains)  # (n_iter, k_params)
        log_likelihoods = np.array(self._log_likelihoods)
        acceptance_history = np.array(self._acceptance_history)

        return PMCMCResults(
            chains=chains,
            param_names=self._param_names,
            log_likelihood_chain=log_likelihoods,
            acceptance_history=acceptance_history,
            burnin=self.burnin,
            thin=self.thin,
        )
