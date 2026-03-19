"""Particle Gibbs with Ancestor Sampling (PGAS).

Improves upon the standard Particle Gibbs by adding ancestor sampling
to the Conditional SMC step. This breaks the path degeneracy problem
that limits mixing in standard PG.

At each time step in the CSMC, instead of keeping the reference particle's
own ancestor, we resample the ancestor using weights that account for the
transition density from each particle to the reference state. This allows
the reference trajectory to "reconnect" with high-probability ancestors,
dramatically improving mixing.

References:
    Lindsten, F., Jordan, M. I. & Schon, T. B. (2014). Particle Gibbs
    with ancestor sampling. JMLR, 15, 2145-2184.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.pmcmc.base import BasePMCMC
from particlefilterbox.pmcmc.conditional_smc import CSMCResult
from particlefilterbox.pmcmc.results import PMCMCResults

__all__ = ["PGAS"]


class PGAS(BasePMCMC):
    """Particle Gibbs with Ancestor Sampling.

    Like standard Particle Gibbs but with ancestor sampling in the CSMC step.
    The ancestor of the reference particle is resampled at each time step
    using weights that incorporate the transition density, leading to much
    better mixing.

    Parameters
    ----------
    model : Any
        State-space model. Must implement:
        - ``set_params(theta)``: Set model parameters.
        - ``get_params()``: Get current parameters.
        - ``initial_sample(n, rng)``: Sample initial states.
        - ``transition_sample(x, rng)``: Propagate state.
        - ``transition_logpdf(x_new, x_old)``: Log transition density.
        - ``observation_logpdf(y, x)``: Log observation density.
        - ``filter(endog, n_particles, rng)``: Run standard PF.
    prior : Any
        Prior distribution with ``logpdf(theta)`` and ``sample(rng)``.
    n_particles : int
        Number of particles for CSMC. Default 50.
    n_iterations : int
        Number of Gibbs iterations. Default 2000.
    param_sampler : Callable | None
        Custom parameter sampler. See ParticleGibbs for details.
    burnin : int | None
        Burn-in iterations.
    thin : int
        Thinning factor.
    seed : int | None
        Random seed.

    Examples
    --------
    >>> pgas = PGAS(
    ...     model=sv_model,
    ...     prior=prior,
    ...     n_particles=50,
    ...     n_iterations=2000,
    ... )
    >>> results = pgas.run(endog=observations)
    """

    def __init__(
        self,
        model: Any,
        prior: Any,
        n_particles: int = 50,
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
        self._state_trajectories: list[NDArray[np.float64]] = []

        if hasattr(model, "param_names"):
            self._param_names = list(model.param_names)

    def run(self, endog: NDArray[np.float64], **kwargs: Any) -> PMCMCResults:
        """Run the PGAS sampler.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observations of shape ``(T,)`` or ``(T, d_y)``.
        **kwargs : Any
            Additional arguments:
            - ``theta_init``: Initial parameter vector.
            - ``x_ref_init``: Initial reference trajectory.
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

        # Initialize reference trajectory
        x_ref = kwargs.get("x_ref_init")
        if x_ref is None:
            x_ref = self._init_reference_trajectory(endog, theta_current)

        x_ref = np.asarray(x_ref, dtype=np.float64)

        # Reset storage
        self._chains = []
        self._log_likelihoods = []
        self._acceptance_history = []
        self._state_trajectories = []

        # Main PGAS loop
        for i in range(self.n_iterations):
            # 1. SAMPLE STATES via CSMC with ancestor sampling
            csmc_result = self._csmc_with_ancestor_sampling(endog, theta_current, x_ref)
            x_ref = csmc_result.trajectory
            log_lik = csmc_result.log_likelihood

            # 2. SAMPLE PARAMETERS
            theta_current = self._sample_params(endog, theta_current, x_ref)

            # 3. Store
            self._store_iteration(theta_current, log_lik, accepted=True)
            self._state_trajectories.append(x_ref.copy())

            if verbose > 0 and (i + 1) % verbose == 0:
                print(f"PGAS iteration {i + 1}/{self.n_iterations}, loglik: {log_lik:.2f}")

        return self._build_results()

    def _csmc_with_ancestor_sampling(
        self,
        endog: NDArray[np.float64],
        theta: NDArray[np.float64],
        x_ref: NDArray[np.float64],
    ) -> CSMCResult:
        """Run Conditional SMC with ancestor sampling.

        This is the core of PGAS. Like standard CSMC, particle N is fixed
        to the reference trajectory. The key difference is that at each
        time step, the ancestor of particle N is resampled using weights
        that incorporate the transition density to the reference state.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observations of shape ``(T,)``.
        theta : NDArray[np.float64]
            Current parameters.
        x_ref : NDArray[np.float64]
            Reference trajectory of shape ``(T,)``.

        Returns
        -------
        CSMCResult
            CSMC result with trajectory and diagnostics.
        """
        self.model.set_params(theta)

        t_len = len(endog)
        n_p = self.n_particles
        rng = self._rng

        # Determine state dimension
        if x_ref.ndim == 1:
            d_x = 1
            x_ref_2d = x_ref.reshape(-1, 1)
        else:
            d_x = x_ref.shape[1]
            x_ref_2d = x_ref

        # Storage
        all_particles = np.zeros((t_len, n_p, d_x))
        all_ancestors = np.zeros((t_len, n_p), dtype=np.int64)
        log_likelihood = 0.0

        # --- Time t=0: Initialize ---
        particles = np.zeros((n_p, d_x))

        if hasattr(self.model, "initial_sample"):
            x_init = self.model.initial_sample(n_p - 1, rng)
            if np.isscalar(x_init) or (hasattr(x_init, "ndim") and x_init.ndim == 0):
                particles[0, 0] = x_init  # type: ignore[assignment]
            elif x_init.ndim == 1:
                particles[: n_p - 1, 0] = x_init
            else:
                particles[: n_p - 1] = x_init
        else:
            particles[: n_p - 1] = rng.standard_normal((n_p - 1, d_x))

        # Fix reference particle
        particles[n_p - 1] = x_ref_2d[0]

        # Compute weights at t=0
        log_weights = np.zeros(n_p)
        for j in range(n_p):
            x_j = particles[j, 0] if d_x == 1 else particles[j]
            if hasattr(self.model, "observation_logpdf"):
                log_weights[j] = self.model.observation_logpdf(endog[0], x_j)

        # Normalize
        max_lw = np.max(log_weights)
        weights = np.exp(log_weights - max_lw)
        sum_w = np.sum(weights)
        if sum_w < 1e-300:
            weights = np.ones(n_p) / n_p
        else:
            log_likelihood += max_lw + np.log(sum_w) - np.log(n_p)
            weights = weights / sum_w

        all_particles[0] = particles.copy()
        all_ancestors[0] = np.arange(n_p)

        # --- Time t=1..T-1 ---
        for t in range(1, t_len):
            # Ancestor sampling for reference particle
            ancestor_ref = self._ancestor_sampling_weights(
                particles, weights, x_ref_2d[t], d_x, rng
            )

            # Standard resampling for particles 0..N-2
            ancestors = np.zeros(n_p, dtype=np.int64)
            ancestors[: n_p - 1] = rng.choice(n_p, size=n_p - 1, p=weights)
            ancestors[n_p - 1] = ancestor_ref  # PGAS: resampled ancestor!

            # Propagate particles 0..N-2
            new_particles = np.zeros((n_p, d_x))
            for j in range(n_p - 1):
                parent = particles[ancestors[j]]
                x_p = parent[0] if d_x == 1 else parent
                if hasattr(self.model, "transition_sample"):
                    x_new = self.model.transition_sample(x_p, rng)
                    if np.isscalar(x_new):
                        new_particles[j, 0] = x_new  # type: ignore[assignment]
                    else:
                        new_particles[j] = x_new
                else:
                    new_particles[j] = parent + rng.standard_normal(d_x)

            # Fix reference particle
            new_particles[n_p - 1] = x_ref_2d[t]

            particles = new_particles

            # Compute weights
            log_weights = np.zeros(n_p)
            for j in range(n_p):
                x_j = particles[j, 0] if d_x == 1 else particles[j]
                if hasattr(self.model, "observation_logpdf"):
                    log_weights[j] = self.model.observation_logpdf(endog[t], x_j)

            # Normalize
            max_lw = np.max(log_weights)
            weights_raw = np.exp(log_weights - max_lw)
            sum_w = np.sum(weights_raw)
            if sum_w < 1e-300:
                weights = np.ones(n_p) / n_p
            else:
                log_likelihood += max_lw + np.log(sum_w) - np.log(n_p)
                weights = weights_raw / sum_w

            all_particles[t] = particles.copy()
            all_ancestors[t] = ancestors

        # Sample trajectory
        trajectory = self._trace_trajectory(all_particles, all_ancestors, weights, rng)

        if d_x == 1:
            trajectory = trajectory.squeeze(-1)
            final_particles = particles.squeeze(-1)
        else:
            final_particles = particles

        return CSMCResult(
            trajectory=trajectory,
            particles=final_particles,
            weights=weights,
            log_likelihood=log_likelihood,
            ancestors=all_ancestors,
        )

    def _ancestor_sampling_weights(
        self,
        particles: NDArray[np.float64],
        weights: NDArray[np.float64],
        x_ref_t: NDArray[np.float64],
        d_x: int,
        rng: np.random.Generator,
    ) -> int:
        """Compute ancestor sampling weights and sample ancestor for reference.

        For each particle j, compute:
            log w_tilde^(j) = log w_{t-1}^(j) + log p(x_t^ref | x_{t-1}^(j))

        Then sample ancestor from Categorical(normalize(w_tilde)).

        Parameters
        ----------
        particles : NDArray[np.float64]
            Current particles, shape ``(N, d_x)``.
        weights : NDArray[np.float64]
            Current normalized weights, shape ``(N,)``.
        x_ref_t : NDArray[np.float64]
            Reference state at time t, shape ``(d_x,)``.
        d_x : int
            State dimension.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        int
            Sampled ancestor index for the reference particle.
        """
        n_p = len(weights)
        log_w_tilde = np.full(n_p, -np.inf)

        for j in range(n_p):
            if weights[j] < 1e-300:
                continue

            log_wj = np.log(max(weights[j], 1e-300))

            x_j = particles[j, 0] if d_x == 1 else particles[j]
            x_ref = x_ref_t[0] if d_x == 1 else x_ref_t

            if hasattr(self.model, "transition_logpdf"):
                log_trans = self.model.transition_logpdf(x_ref, x_j)
            else:
                # Fallback: assume Gaussian transition with unit variance
                diff_arr = np.asarray(x_ref, dtype=np.float64) - np.asarray(x_j, dtype=np.float64)
                log_trans = float(-0.5 * np.sum(diff_arr**2))

            log_w_tilde[j] = log_wj + log_trans

        # Normalize
        max_lw = np.max(log_w_tilde)
        if not np.isfinite(max_lw):
            # Fallback: uniform
            return int(rng.integers(0, n_p))

        w_tilde = np.exp(log_w_tilde - max_lw)
        sum_w = np.sum(w_tilde)
        if sum_w < 1e-300:
            return int(rng.integers(0, n_p))

        w_tilde = w_tilde / sum_w
        return int(rng.choice(n_p, p=w_tilde))

    def _trace_trajectory(
        self,
        all_particles: NDArray[np.float64],
        all_ancestors: NDArray[np.int64],
        final_weights: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Trace back a trajectory from the final particles.

        Parameters
        ----------
        all_particles : NDArray[np.float64]
            All particles, shape ``(T, N, d_x)``.
        all_ancestors : NDArray[np.int64]
            Ancestor indices, shape ``(T, N)``.
        final_weights : NDArray[np.float64]
            Final normalized weights.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray[np.float64]
            Sampled trajectory of shape ``(T, d_x)``.
        """
        t_len, _, d_x = all_particles.shape

        idx: int = int(rng.choice(all_particles.shape[1], p=final_weights))  # pyright: ignore[reportUnknownArgumentType]

        trajectory = np.zeros((t_len, d_x))
        trajectory[t_len - 1] = all_particles[t_len - 1, idx]

        for t in range(t_len - 2, -1, -1):
            idx = all_ancestors[t + 1, idx]
            trajectory[t] = all_particles[t, idx]

        return trajectory

    def _sample_params(
        self,
        endog: NDArray[np.float64],
        theta_current: NDArray[np.float64],
        states: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sample parameters given states.

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

        # Fallback: MH step
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

        theta_proposed = theta_current + np.sqrt(scale) * self._rng.standard_normal(dim)

        logprior_current = self._log_prior(theta_current)
        logprior_proposed = self._log_prior(theta_proposed)

        if not np.isfinite(logprior_proposed):
            return theta_current

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
        """Compute conditional log-likelihood p(y, x | theta)."""
        self.model.set_params(theta)
        t_len = len(endog)
        loglik = 0.0

        for t in range(t_len):
            x_t = states[t]
            if hasattr(self.model, "observation_logpdf"):
                loglik += self.model.observation_logpdf(endog[t], x_t)
            if t > 0 and hasattr(self.model, "transition_logpdf"):
                loglik += self.model.transition_logpdf(states[t], states[t - 1])

        return float(loglik)

    def _init_reference_trajectory(
        self,
        endog: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Initialize reference trajectory by running a standard PF."""
        self.model.set_params(theta)
        t_len = len(endog)

        if hasattr(self.model, "filter"):
            result = self.model.filter(
                endog=endog,
                n_particles=self.n_particles,
                rng=self._rng,
            )
            if hasattr(result, "filtered_means") and result.filtered_means is not None:
                return result.filtered_means

        # Fallback
        x_ref = np.zeros(t_len)
        if hasattr(self.model, "initial_sample"):
            init = self.model.initial_sample(1, self._rng)
            x_ref[0] = init if np.isscalar(init) else init[0]  # type: ignore[index]
        else:
            x_ref[0] = self._rng.standard_normal()

        for t in range(1, t_len):
            if hasattr(self.model, "transition_sample"):
                x_ref[t] = float(self.model.transition_sample(x_ref[t - 1], self._rng))
            else:
                x_ref[t] = x_ref[t - 1] + self._rng.standard_normal()

        return x_ref
