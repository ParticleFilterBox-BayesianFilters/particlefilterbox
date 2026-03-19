"""Particle Gibbs sampler.

Combines Gibbs sampling with Conditional SMC for joint inference of
states and parameters in state-space models. At each iteration:
1. Sample states via Conditional SMC (conditioned on reference trajectory)
2. Sample parameters from p(theta | x, y) (via conjugacy or MH)
3. Update reference trajectory

References:
    Andrieu, C., Doucet, A. & Holenstein, R. (2010). Particle Markov chain
    Monte Carlo methods. JRSS-B, 72(3), 269-342. Section 2.4.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.pmcmc.base import BasePMCMC
from particlefilterbox.pmcmc.conditional_smc import ConditionalSMC
from particlefilterbox.pmcmc.results import PMCMCResults

__all__ = ["ParticleGibbs"]


class ParticleGibbs(BasePMCMC):
    """Particle Gibbs sampler for state-space models.

    Alternates between sampling states via Conditional SMC and sampling
    parameters from their full conditional distribution.

    Parameters
    ----------
    model : Any
        State-space model. Must implement interfaces for both the particle
        filter and the Conditional SMC:
        - ``set_params(theta)``: Set model parameters.
        - ``get_params()``: Get current parameters.
        - ``initial_sample(n, rng)``: Sample initial states.
        - ``transition_sample(x, rng)``: Propagate state.
        - ``observation_logpdf(y, x)``: Log observation density.
        - ``filter(endog, n_particles, rng)``: Run standard PF.
        Optionally:
        - ``sample_params_given_states(x, y, rng)``: Sample from
          p(theta | x, y) analytically.
    prior : Any
        Prior distribution with ``logpdf(theta)`` and ``sample(rng)``.
    n_particles : int
        Number of particles for CSMC. Default 100.
    n_iterations : int
        Number of Gibbs iterations. Default 2000.
    param_sampler : Callable | None
        Custom function ``f(model, states, endog, theta_current, rng) -> theta``
        for sampling parameters given states. If None and model has
        ``sample_params_given_states``, uses that. Otherwise uses MH step.
    burnin : int | None
        Burn-in iterations.
    thin : int
        Thinning factor.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        model: Any,
        prior: Any,
        n_particles: int = 100,
        n_iterations: int = 2000,
        param_sampler: Callable[..., NDArray[np.float64]] | None = None,
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

        self._param_sampler = param_sampler
        self._csmc = ConditionalSMC(model=model, n_particles=n_particles)

        # Storage for state trajectories
        self._state_trajectories: list[NDArray[np.float64]] = []

        if hasattr(model, "param_names"):
            self._param_names = list(model.param_names)

    def run(self, endog: NDArray[np.float64], **kwargs: Any) -> PMCMCResults:
        """Run the Particle Gibbs sampler.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observations of shape ``(T,)`` or ``(T, d_y)``.
        **kwargs : Any
            Additional arguments:
            - ``theta_init``: Initial parameter vector.
            - ``x_ref_init``: Initial reference trajectory. If not provided,
              runs a standard PF to obtain one.
            - ``verbose``: Print progress every N iterations.

        Returns
        -------
        PMCMCResults
            Posterior samples and diagnostics.
        """
        endog = np.asarray(endog, dtype=np.float64)
        verbose: int = kwargs.get("verbose", 0)

        # Initialize theta
        theta_init = kwargs.get("theta_init")
        if theta_init is not None:
            theta_current = np.asarray(theta_init, dtype=np.float64)
        elif hasattr(self.prior, "sample"):
            theta_current = self.prior.sample(self._rng)
        else:
            theta_current = self.model.get_params()

        theta_current = np.atleast_1d(theta_current).astype(np.float64)

        # Initialize reference trajectory
        x_ref: NDArray[np.float64] | None = kwargs.get("x_ref_init")
        if x_ref is None:
            x_ref = self._init_reference_trajectory(endog, theta_current)

        x_ref = np.asarray(x_ref, dtype=np.float64)

        # Reset storage
        self._chains = []
        self._log_likelihoods = []
        self._acceptance_history = []
        self._state_trajectories = []

        # Main Particle Gibbs loop
        for i in range(self.n_iterations):
            # 1. SAMPLE STATES via CSMC
            x_ref, log_lik = self._sample_states(endog, theta_current, x_ref)

            # 2. SAMPLE PARAMETERS
            theta_current = self._sample_params(endog, theta_current, x_ref)

            # 3. Store
            self._store_iteration(theta_current, log_lik, accepted=True)
            self._state_trajectories.append(x_ref.copy())

            if verbose > 0 and (i + 1) % verbose == 0:
                print(f"PG iteration {i + 1}/{self.n_iterations}, loglik: {log_lik:.2f}")

        return self._build_results()

    def _sample_states(
        self,
        endog: NDArray[np.float64],
        theta: NDArray[np.float64],
        x_ref: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        """Sample new state trajectory via Conditional SMC.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observations.
        theta : NDArray[np.float64]
            Current parameters.
        x_ref : NDArray[np.float64]
            Current reference trajectory.

        Returns
        -------
        tuple[NDArray[np.float64], float]
            New trajectory and log-likelihood.
        """
        try:
            result = self._csmc.run(
                endog=endog,
                theta=theta,
                x_ref=x_ref,
                rng=self._rng,
            )
            return result.trajectory, result.log_likelihood
        except (ValueError, FloatingPointError):
            # Invalid parameters cause PF to fail; keep old trajectory
            return x_ref, -np.inf

    def _sample_params(
        self,
        endog: NDArray[np.float64],
        theta_current: NDArray[np.float64],
        states: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sample parameters given states.

        Uses custom sampler, model's conjugate sampler, or falls back to
        a Metropolis-Hastings step.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observations.
        theta_current : NDArray[np.float64]
            Current parameters.
        states : NDArray[np.float64]
            Current state trajectory.

        Returns
        -------
        NDArray[np.float64]
            New parameter vector.
        """
        if self._param_sampler is not None:
            return self._param_sampler(self.model, states, endog, theta_current, self._rng)

        if hasattr(self.model, "sample_params_given_states"):
            return self.model.sample_params_given_states(states, endog, self._rng)

        # Fallback: simple random walk MH step
        return self._mh_param_step(endog, theta_current, states)

    def _mh_param_step(
        self,
        endog: NDArray[np.float64],
        theta_current: NDArray[np.float64],
        states: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Simple MH step for parameter sampling.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observations.
        theta_current : NDArray[np.float64]
            Current parameters.
        states : NDArray[np.float64]
            Current state trajectory.

        Returns
        -------
        NDArray[np.float64]
            Updated parameter vector.
        """
        dim = len(theta_current)
        scale = (2.38**2) / max(dim, 1) * 0.1

        # Propose
        theta_proposed = theta_current + np.sqrt(scale) * self._rng.standard_normal(dim)

        # Evaluate
        logprior_current = self._log_prior(theta_current)
        logprior_proposed = self._log_prior(theta_proposed)

        if not np.isfinite(logprior_proposed):
            return theta_current

        # Compute conditional likelihood p(y, x | theta)
        loglik_current = self._conditional_loglik(endog, states, theta_current)
        loglik_proposed = self._conditional_loglik(endog, states, theta_proposed)

        log_alpha = (logprior_proposed + loglik_proposed) - (logprior_current + loglik_current)

        if np.log(self._rng.random()) < log_alpha:
            return theta_proposed
        return theta_current

    def _conditional_loglik(
        self,
        endog: NDArray[np.float64],
        states: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> float:
        """Compute log p(y, x | theta) = log p(y|x,theta) + log p(x|theta).

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observations.
        states : NDArray[np.float64]
            State trajectory.
        theta : NDArray[np.float64]
            Parameters.

        Returns
        -------
        float
            Conditional log-likelihood.
        """
        self.model.set_params(theta)
        t_len = len(endog)
        loglik = 0.0

        try:
            for t in range(t_len):
                x_t = states[t]
                if hasattr(self.model, "observation_logpdf"):
                    val = self.model.observation_logpdf(endog[t], x_t)
                    if not np.isfinite(val):
                        return -np.inf
                    loglik += val

                if t > 0 and hasattr(self.model, "transition_logpdf"):
                    x_prev = states[t - 1]
                    val = self.model.transition_logpdf(x_t, x_prev)
                    if not np.isfinite(val):
                        return -np.inf
                    loglik += val
        except (ValueError, FloatingPointError):
            return -np.inf

        return float(loglik)

    def _init_reference_trajectory(
        self,
        endog: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Initialize reference trajectory by running a standard PF.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observations.
        theta : NDArray[np.float64]
            Initial parameters.

        Returns
        -------
        NDArray[np.float64]
            Initial reference trajectory.
        """
        self.model.set_params(theta)
        t_len = len(endog)

        # Try to use model's filter
        if hasattr(self.model, "filter"):
            result = self.model.filter(
                endog=endog,
                n_particles=self.n_particles,
                rng=self._rng,
            )
            if hasattr(result, "filtered_means") and result.filtered_means is not None:
                return result.filtered_means

        # Fallback: simple forward simulation
        x_ref = np.zeros(t_len)
        if hasattr(self.model, "initial_sample"):
            x_ref[0] = float(self.model.initial_sample(1, self._rng))
        else:
            x_ref[0] = self._rng.standard_normal()

        for t in range(1, t_len):
            if hasattr(self.model, "transition_sample"):
                x_ref[t] = float(self.model.transition_sample(x_ref[t - 1], self._rng))
            else:
                x_ref[t] = x_ref[t - 1] + self._rng.standard_normal()

        return x_ref
