"""Particle Marginal Metropolis-Hastings (PMMH).

The most fundamental PMCMC algorithm. Uses a particle filter to obtain
unbiased estimates of the marginal likelihood, which are plugged into
a Metropolis-Hastings acceptance ratio.

The key insight (Andrieu, Doucet & Holenstein, 2010) is that replacing
the exact likelihood with an unbiased particle filter estimate still
yields a Markov chain with the correct stationary distribution.

References:
    Andrieu, C., Doucet, A. & Holenstein, R. (2010). Particle Markov chain
    Monte Carlo methods. JRSS-B, 72(3), 269-342.
    Andrieu, C. & Roberts, G. O. (2009). The pseudo-marginal approach for
    efficient Monte Carlo computations. Annals of Statistics, 37(2), 697-725.
    Doucet, A., Pitt, M. K., Deligiannidis, G. & Kohn, R. (2015). Efficient
    implementation of Markov chain Monte Carlo when using an unbiased
    likelihood estimator. Biometrika, 102(2), 295-313.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.pmcmc.base import BasePMCMC
from particlefilterbox.pmcmc.proposals import (
    AdaptiveGaussian,
    BaseProposal,
    GaussianRandomWalk,
)
from particlefilterbox.pmcmc.results import PMCMCResults

__all__ = ["PMMH"]


class PMMH(BasePMCMC):
    """Particle Marginal Metropolis-Hastings sampler.

    Combines Metropolis-Hastings MCMC with particle filter likelihood
    estimation for Bayesian inference in state-space models.

    Parameters
    ----------
    model : Any
        State-space model. Must implement:
        - ``set_params(theta)``: Set model parameters.
        - ``get_params()``: Get current parameters as NDArray.
        - ``filter(endog, n_particles, rng)``: Run particle filter,
          returning object with ``log_likelihood`` attribute.
        - ``param_names`` (optional): List of parameter names.
    prior : Any
        Prior distribution. Must implement:
        - ``logpdf(theta)``: Log-prior density.
        - ``sample(rng)`` (optional): Draw from prior.
    n_particles : int
        Number of particles for likelihood estimation. Default 200.
    n_iterations : int
        Number of MCMC iterations. Default 5000.
    proposal_cov : NDArray | float | str | None
        Proposal covariance. Options:
        - NDArray: Fixed covariance matrix.
        - float: Scalar for isotropic covariance.
        - ``'adaptive'``: Use AdaptiveGaussian with Roberts-Rosenthal.
        - None: Use default ``(2.38^2/d)*I``.
        Default ``'adaptive'``.
    proposal : BaseProposal | None
        Custom proposal distribution. If provided, overrides ``proposal_cov``.
    target_acceptance : float
        Target acceptance rate for adaptive proposal. Default 0.234.
    burnin : int | None
        Burn-in iterations. Default ``n_iterations // 2``.
    thin : int
        Thinning factor. Default 1.
    seed : int | None
        Random seed. Default None.

    Examples
    --------
    >>> pmmh = PMMH(
    ...     model=sv_model,
    ...     prior=prior,
    ...     n_particles=200,
    ...     n_iterations=5000,
    ...     proposal_cov='adaptive',
    ... )
    >>> results = pmmh.run(endog=observations)
    >>> print(results.summary())
    """

    def __init__(
        self,
        model: Any,
        prior: Any,
        n_particles: int = 200,
        n_iterations: int = 5000,
        proposal_cov: NDArray[np.float64] | float | str | None = "adaptive",
        proposal: BaseProposal | None = None,
        target_acceptance: float = 0.234,
        burnin: int | None = None,
        thin: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            prior=prior,
            n_particles=n_particles,
            n_iterations=n_iterations,
            burnin=burnin,
            thin=thin,
            seed=seed,
        )

        self.target_acceptance = target_acceptance
        self._proposal_cov_spec = proposal_cov
        self._custom_proposal = proposal

        # Proposal will be initialized in run() once we know dimension
        self._proposal: BaseProposal | None = None

        # Try to get param_names from model
        if hasattr(model, "param_names"):
            self._param_names = list(model.param_names)

    def run(self, endog: NDArray[np.float64], **kwargs: Any) -> PMCMCResults:
        """Run the PMMH sampler.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observed time series of shape ``(T,)`` or ``(T, d_y)``.
        **kwargs : Any
            Additional arguments:
            - ``theta_init``: Initial parameter vector. If not provided,
              samples from prior or uses model's current parameters.
            - ``verbose``: Print progress every N iterations.

        Returns
        -------
        PMCMCResults
            Posterior samples and diagnostics.
        """
        endog = np.asarray(endog, dtype=np.float64)
        verbose = kwargs.get("verbose", 0)

        # Initialize theta
        theta_init = kwargs.get("theta_init")
        if theta_init is not None:
            theta_current = np.asarray(theta_init, dtype=np.float64)
        elif hasattr(self.prior, "sample"):
            theta_current = self.prior.sample(self._rng)
        else:
            theta_current = self.model.get_params()

        theta_current = np.atleast_1d(theta_current).astype(np.float64)
        dim = len(theta_current)

        # Initialize proposal
        self._proposal = self._init_proposal(dim)

        # Initialize: run PF at theta_0
        loglik_current = self._pf_loglike(theta_current, endog, self._rng)
        logprior_current = self._log_prior(theta_current)

        # If initial point has -inf likelihood, try a few more from prior
        max_init_tries = 50
        n_try = 0
        while not np.isfinite(loglik_current + logprior_current) and n_try < max_init_tries:
            if hasattr(self.prior, "sample"):
                theta_current = self.prior.sample(self._rng)
            else:
                theta_current = theta_current + 0.01 * self._rng.standard_normal(dim)
            loglik_current = self._pf_loglike(theta_current, endog, self._rng)
            logprior_current = self._log_prior(theta_current)
            n_try += 1

        # Reset storage
        self._chains = []
        self._log_likelihoods = []
        self._acceptance_history = []

        # Main PMMH loop
        for i in range(self.n_iterations):
            # 1. Propose
            theta_proposed = self._propose(theta_current)

            # 2. Evaluate prior at proposal
            logprior_proposed = self._log_prior(theta_proposed)

            if np.isfinite(logprior_proposed):
                # 3. Run PF at proposal
                loglik_proposed = self._pf_loglike(theta_proposed, endog, self._rng)

                # 4. Compute acceptance ratio
                log_alpha = self._log_acceptance_ratio(
                    logprior_proposed=logprior_proposed,
                    loglik_proposed=loglik_proposed,
                    logprior_current=logprior_current,
                    loglik_current=loglik_current,
                    theta_proposed=theta_proposed,
                    theta_current=theta_current,
                )
            else:
                # Prior is -inf, reject immediately
                log_alpha = -np.inf
                loglik_proposed = -np.inf

            # 5. Accept/Reject
            log_u = np.log(self._rng.random())
            accepted = bool(log_u < log_alpha)

            if accepted:
                theta_current = theta_proposed
                loglik_current = loglik_proposed
                logprior_current = logprior_proposed

            # 6. Store iteration
            self._store_iteration(theta_current, loglik_current, accepted)

            # 7. Adapt proposal
            self._adapt_proposal(theta_current, accepted)

            # Progress
            if verbose > 0 and (i + 1) % verbose == 0:
                rate = np.mean(
                    self._acceptance_history[-verbose:]  # type: ignore[arg-type]
                )
                print(
                    f"PMMH iteration {i + 1}/{self.n_iterations}, "
                    f"acceptance rate (last {verbose}): {rate:.3f}, "
                    f"loglik: {loglik_current:.2f}"
                )

        return self._build_results()

    def _pf_loglike(
        self,
        theta: NDArray[np.float64],
        endog: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> float:
        """Estimate log-likelihood, returning -inf for invalid parameters.

        Wraps the base implementation to catch errors from invalid parameter
        values (e.g., negative scale parameters).

        Parameters
        ----------
        theta : NDArray[np.float64]
            Parameter vector of shape ``(k,)``.
        endog : NDArray[np.float64]
            Observed data of shape ``(T,)`` or ``(T, d_y)``.
        rng : np.random.Generator | None
            Random number generator.

        Returns
        -------
        float
            Estimated log marginal likelihood.
        """
        try:
            return super()._pf_loglike(theta, endog, rng)
        except (ValueError, FloatingPointError):
            return -np.inf

    def _propose(self, theta_current: NDArray[np.float64]) -> NDArray[np.float64]:
        """Generate a proposal from the current state.

        Parameters
        ----------
        theta_current : NDArray[np.float64]
            Current parameter vector.

        Returns
        -------
        NDArray[np.float64]
            Proposed parameter vector.
        """
        assert self._proposal is not None
        return self._proposal.propose(theta_current, rng=self._rng)

    def _log_acceptance_ratio(
        self,
        logprior_proposed: float,
        loglik_proposed: float,
        logprior_current: float,
        loglik_current: float,
        theta_proposed: NDArray[np.float64],
        theta_current: NDArray[np.float64],
    ) -> float:
        """Compute log Metropolis-Hastings acceptance ratio.

        log alpha = [log p(theta*) + log p(y|theta*)]
                   - [log p(theta) + log p(y|theta)]
                   + log q(theta|theta*) - log q(theta*|theta)

        Parameters
        ----------
        logprior_proposed : float
            Log-prior at proposed theta.
        loglik_proposed : float
            Log-likelihood at proposed theta.
        logprior_current : float
            Log-prior at current theta.
        loglik_current : float
            Log-likelihood at current theta.
        theta_proposed : NDArray[np.float64]
            Proposed parameter vector.
        theta_current : NDArray[np.float64]
            Current parameter vector.

        Returns
        -------
        float
            Log acceptance ratio.
        """
        if not np.isfinite(logprior_proposed + loglik_proposed):
            return -np.inf

        assert self._proposal is not None
        log_q_ratio = self._proposal.log_ratio(theta_proposed, theta_current)

        log_alpha = (
            (logprior_proposed + loglik_proposed)
            - (logprior_current + loglik_current)
            + log_q_ratio
        )

        return float(log_alpha)

    def _adapt_proposal(
        self,
        theta: NDArray[np.float64],
        accepted: bool,
    ) -> None:
        """Adapt the proposal distribution.

        Parameters
        ----------
        theta : NDArray[np.float64]
            Current parameter vector.
        accepted : bool
            Whether the last proposal was accepted.
        """
        if self._proposal is not None:
            self._proposal.adapt(theta, accepted)

    def _init_proposal(self, dim: int) -> BaseProposal:
        """Initialize the proposal distribution.

        Parameters
        ----------
        dim : int
            Dimension of the parameter space.

        Returns
        -------
        BaseProposal
            Initialized proposal distribution.
        """
        if self._custom_proposal is not None:
            return self._custom_proposal

        if isinstance(self._proposal_cov_spec, str):
            if self._proposal_cov_spec == "adaptive":
                initial_cov = self._initial_proposal_cov(dim)
                return AdaptiveGaussian(
                    dim=dim,
                    initial_cov=initial_cov,
                    target_acceptance=self.target_acceptance,
                    seed=int(self._rng.integers(0, 2**31)),
                )
            else:
                msg = f"Unknown proposal_cov string: {self._proposal_cov_spec}"
                raise ValueError(msg)
        elif self._proposal_cov_spec is None:
            return GaussianRandomWalk(
                dim=dim,
                seed=int(self._rng.integers(0, 2**31)),
            )
        else:
            return GaussianRandomWalk(
                dim=dim,
                cov=self._proposal_cov_spec,
                seed=int(self._rng.integers(0, 2**31)),
            )

    def _initial_proposal_cov(self, dim: int) -> NDArray[np.float64]:
        """Compute initial proposal covariance.

        Uses (2.38^2/d)*I_d as default, or estimates from prior if available.

        Parameters
        ----------
        dim : int
            Dimension of the parameter space.

        Returns
        -------
        NDArray[np.float64]
            Initial covariance matrix of shape ``(d, d)``.
        """
        scale = (2.38**2) / max(dim, 1)

        # Try to get scale from prior covariance
        if hasattr(self.prior, "cov"):
            prior_cov = np.asarray(self.prior.cov)
            if prior_cov.ndim == 2:
                return scale * prior_cov
            elif prior_cov.ndim == 1:
                return scale * np.diag(prior_cov)

        return scale * np.eye(dim)
