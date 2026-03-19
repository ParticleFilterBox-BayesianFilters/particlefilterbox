"""SMC^2: Sequential Monte Carlo Squared.

Implements the SMC^2 algorithm of Chopin, Jacob & Papaspiliopoulos (2013)
for joint parameter and state inference in state-space models.

The algorithm maintains a population of parameter particles, each with
its own internal particle filter for state estimation. This provides:
- Online parameter learning
- Sequential model evidence estimation
- Full posterior over parameters and states

Algorithm:
    INIT: theta^(i) ~ prior, run empty PF for each
    FOR t=1..T:
      1. EXTEND internal PF one observation: log_lik_inc = PF step
      2. REWEIGHT: log_w += log_lik_inc
      3. LOG-EVIDENCE: log_mean_exp(log_w)
      4. RESAMPLE + REJUVENATE: propose theta*, run full PF, accept/reject MH

Complexity: O(T * N_theta * N_x)

References:
    Chopin, N., Jacob, P.E. & Papaspiliopoulos, O. (2013). SMC^2: an
    efficient algorithm for sequential analysis of state space models.
    JRSS-B, 75(3), 397-426.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from particlefilterbox._logging import get_logger
from particlefilterbox.smc.base import BaseSMC
from particlefilterbox.smc.results import SMCResults
from particlefilterbox.utils.log_ops import log_mean_exp, log_sum_exp

logger = get_logger("smc.smc_squared")


class StateSpaceModel(Protocol):
    """Protocol for state-space models compatible with SMC^2.

    A state-space model must define:
    - initial_state_sample: x_0 ~ p(x_0 | theta)
    - transition_sample: x_t ~ p(x_t | x_{t-1}, theta)
    - observation_logpdf: log p(y_t | x_t, theta)
    """

    def initial_state_sample(
        self,
        theta: NDArray[np.floating[Any]],
        n_particles: int,
        rng: np.random.Generator,
    ) -> NDArray[np.floating[Any]]:
        """Sample initial states x_0 ~ p(x_0 | theta).

        Parameters
        ----------
        theta : NDArray, shape (k_params,)
            Model parameters.
        n_particles : int
            Number of state particles.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray, shape (n_particles, k_states)
            Initial state particles.
        """
        ...

    def transition_sample(
        self,
        x_prev: NDArray[np.floating[Any]],
        theta: NDArray[np.floating[Any]],
        rng: np.random.Generator,
    ) -> NDArray[np.floating[Any]]:
        """Sample state transition x_t ~ p(x_t | x_{t-1}, theta).

        Parameters
        ----------
        x_prev : NDArray, shape (n_particles, k_states)
            Previous state particles.
        theta : NDArray, shape (k_params,)
            Model parameters.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray, shape (n_particles, k_states)
            Transitioned state particles.
        """
        ...

    def observation_logpdf(
        self,
        y_t: NDArray[np.floating[Any]],
        x_t: NDArray[np.floating[Any]],
        theta: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Log-density of observation given state.

        Parameters
        ----------
        y_t : NDArray, shape (k_obs,)
            Observation at time t.
        x_t : NDArray, shape (n_particles, k_states)
            State particles at time t.
        theta : NDArray, shape (k_params,)
            Model parameters.

        Returns
        -------
        NDArray, shape (n_particles,)
            Log p(y_t | x_t^(j), theta) for each state particle.
        """
        ...


class PriorDistribution(Protocol):
    """Protocol for prior distributions in SMC^2."""

    def logpdf(self, theta: NDArray[np.floating[Any]]) -> float:
        """Log-density of the prior."""
        ...

    def sample(self, rng: np.random.Generator) -> NDArray[np.floating[Any]]:
        """Draw one sample from the prior."""
        ...


class InternalPF:
    """Internal particle filter for a single theta particle.

    Maintains state particles and weights, and can be extended
    one observation at a time.

    Parameters
    ----------
    n_x : int
        Number of state particles.
    model : StateSpaceModel
        State-space model.
    theta : NDArray
        Parameter vector for this PF.
    rng : np.random.Generator
        Random number generator.
    """

    def __init__(
        self,
        n_x: int,
        model: Any,
        theta: NDArray[np.floating[Any]],
        rng: np.random.Generator,
    ) -> None:
        self.n_x = n_x
        self.model = model
        self.theta = theta.copy()
        self.rng = rng

        # State particles and weights
        self.x_particles: NDArray[np.floating[Any]] | None = None
        self.log_weights: NDArray[np.floating[Any]] = np.full(n_x, -np.log(n_x))
        self.log_evidence: float = 0.0
        self.initialized: bool = False
        self.t: int = 0

    def initialize(self) -> None:
        """Initialize state particles from the model's initial distribution."""
        self.x_particles = self.model.initial_state_sample(self.theta, self.n_x, self.rng)
        self.log_weights = np.full(self.n_x, -np.log(self.n_x))
        self.log_evidence = 0.0
        self.initialized = True
        self.t = 0

    def extend(self, y_t: NDArray[np.floating[Any]]) -> float:
        """Extend the PF by one observation.

        Performs one bootstrap PF step:
        1. Resample (if needed)
        2. Propagate: x_t ~ p(x_t | x_{t-1}, theta)
        3. Reweight: w_t = p(y_t | x_t, theta)

        Parameters
        ----------
        y_t : NDArray, shape (k_obs,)
            New observation.

        Returns
        -------
        float
            Log incremental likelihood: log p(y_t | y_{1:t-1}, theta).
        """
        if not self.initialized:
            self.initialize()

        assert self.x_particles is not None

        # Resample if ESS is low
        ess = self._compute_ess()
        if ess < 0.5 * self.n_x and self.t > 0:
            self._resample()

        # Propagate
        if self.t > 0:
            self.x_particles = self.model.transition_sample(self.x_particles, self.theta, self.rng)

        # Compute log-weights from observation
        log_obs = self.model.observation_logpdf(y_t, self.x_particles, self.theta)

        # Incremental likelihood: log p(y_t | y_{1:t-1}, theta)
        log_lik_inc = float(log_mean_exp(self.log_weights + log_obs))

        # Update weights
        self.log_weights = self.log_weights + log_obs
        # Normalize
        self.log_weights = self.log_weights - log_sum_exp(self.log_weights)

        self.log_evidence += log_lik_inc
        self.t += 1

        return log_lik_inc

    def _compute_ess(self) -> float:
        """Compute ESS from current log-weights."""
        log_w_norm = self.log_weights - log_sum_exp(self.log_weights)
        return float(np.exp(-log_sum_exp(2.0 * log_w_norm)))

    def _resample(self) -> None:
        """Systematic resampling of state particles."""
        assert self.x_particles is not None
        weights = np.exp(self.log_weights - log_sum_exp(self.log_weights))
        n = self.n_x
        # Systematic resampling
        positions = (self.rng.uniform() + np.arange(n)) / n
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n - 1)

        self.x_particles = self.x_particles[indices].copy()
        self.log_weights = np.full(n, -np.log(n))

    def run_full(self, endog: NDArray[np.floating[Any]]) -> float:
        """Run a full PF over all observations.

        Parameters
        ----------
        endog : NDArray, shape (T, k_obs)
            All observations.

        Returns
        -------
        float
            Total log marginal likelihood: log p(y_{1:T} | theta).
        """
        self.initialize()
        total_log_lik = 0.0

        for t in range(endog.shape[0]):
            y_t = endog[t]
            log_lik_inc = self.extend(y_t)
            total_log_lik += log_lik_inc

        return total_log_lik

    def copy(self) -> InternalPF:
        """Create a deep copy of this internal PF.

        Returns
        -------
        InternalPF
            A new InternalPF with copied state.
        """
        new_pf = InternalPF(
            n_x=self.n_x,
            model=self.model,
            theta=self.theta.copy(),
            rng=np.random.default_rng(
                int(self.rng.integers(0, 2**31))  # type: ignore[arg-type]
            ),
        )
        if self.x_particles is not None:
            new_pf.x_particles = self.x_particles.copy()
        new_pf.log_weights = self.log_weights.copy()
        new_pf.log_evidence = self.log_evidence
        new_pf.initialized = self.initialized
        new_pf.t = self.t
        return new_pf


class SMCSquared(BaseSMC):
    """SMC^2: Joint parameter and state inference.

    Maintains a population of parameter particles, each with an internal
    particle filter for state estimation. Provides online learning of
    parameters as data arrives sequentially.

    Parameters
    ----------
    model_class : type or callable
        State-space model class/factory. Called as model_class() to create
        instances, or used directly if it satisfies StateSpaceModel protocol.
    n_theta : int
        Number of parameter particles. Default 200.
    n_x : int
        Number of state particles per parameter particle. Default 50.
    prior : PriorDistribution
        Prior distribution over parameters.
    n_mcmc_moves : int
        Number of MCMC rejuvenation moves. Default 3.
    resampling_method : str
        Resampling method. Default 'systematic'.
    ess_threshold : float
        ESS threshold. Default 0.5.
    seed : int
        Random seed. Default 42.

    Examples
    --------
    >>> smc2 = SMCSquared(
    ...     model_class=SVModel,
    ...     n_theta=500,
    ...     n_x=100,
    ...     prior=prior,
    ... )
    >>> results = smc2.run(endog=observations)
    >>> print(results.posterior_mean())
    """

    def __init__(
        self,
        model_class: Any,
        n_theta: int = 200,
        n_x: int = 50,
        prior: Any = None,
        n_mcmc_moves: int = 3,
        resampling_method: str = "systematic",
        ess_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__(
            n_particles=n_theta,
            resampling_method=resampling_method,
            ess_threshold=ess_threshold,
        )
        self.model_class = model_class
        self.n_theta = n_theta
        self.n_x = n_x
        self.prior = prior
        self.n_mcmc_moves = n_mcmc_moves
        self._rng = np.random.default_rng(seed)

    def _make_seed(self) -> int:
        """Generate a reproducible seed from the internal RNG."""
        seed: int = self._rng.integers(0, 2**31)  # type: ignore[assignment]
        return seed

    def run(
        self,
        data: NDArray[np.floating[Any]] | None = None,
        endog: NDArray[np.floating[Any]] | None = None,
    ) -> SMCResults:
        """Run SMC^2 over sequential observations.

        Parameters
        ----------
        data : NDArray or None
            Observed data, shape (T, k_obs). Alias for endog.
        endog : NDArray or None
            Observed data, shape (T, k_obs).

        Returns
        -------
        SMCResults
            Posterior over parameters with log-evidence.
        """
        if endog is None:
            endog = data
        if endog is None:
            msg = "Either data or endog must be provided"
            raise ValueError(msg)

        endog = np.atleast_2d(endog)
        if endog.ndim == 1:
            endog = endog[:, np.newaxis]

        t_total = endog.shape[0]

        # Reset
        self._log_evidence_acc = 0.0
        self._ess_history = []
        self._acceptance_rates = []

        # Step 1: Initialize theta particles from prior
        thetas = np.array([self.prior.sample(self._rng) for _ in range(self.n_theta)])
        k_params = thetas.shape[1] if thetas.ndim > 1 else 1
        if thetas.ndim == 1:
            thetas = thetas[:, np.newaxis]

        log_weights = np.full(self.n_theta, -np.log(self.n_theta))

        # Create model instance
        model = self.model_class() if isinstance(self.model_class, type) else self.model_class

        # Initialize internal PFs
        internal_pfs: list[InternalPF] = []
        for i in range(self.n_theta):
            pf = InternalPF(
                n_x=self.n_x,
                model=model,
                theta=thetas[i],
                rng=np.random.default_rng(self._make_seed()),
            )
            pf.initialize()
            internal_pfs.append(pf)

        # Step 2: Process observations sequentially
        for t in range(t_total):
            y_t = endog[t]

            # Extend each internal PF
            log_lik_increments = np.zeros(self.n_theta)
            for i in range(self.n_theta):
                log_lik_increments[i] = self._extend_pf(internal_pfs[i], y_t)

            # Reweight
            log_weights = log_weights + log_lik_increments

            # Accumulate evidence
            self._accumulate_log_evidence(log_lik_increments)

            # Check ESS and resample+rejuvenate if needed
            ess = self._compute_ess(log_weights)
            self._ess_history.append(ess)

            if ess < self.ess_threshold * self.n_theta:
                logger.debug("t=%d: ESS=%.1f, resampling+rejuvenating", t, ess)

                # Resample
                indices = self._resample(log_weights, self._rng)
                thetas = thetas[indices].copy()
                new_pfs: list[InternalPF] = [internal_pfs[int(idx)].copy() for idx in indices]
                log_weights = np.full(self.n_theta, -np.log(self.n_theta))

                # Rejuvenate
                thetas, new_pfs, acc_rate = self._rejuvenate_theta(
                    thetas, new_pfs, endog[: t + 1], model, k_params
                )
                internal_pfs = new_pfs
                self._acceptance_rates.append(acc_rate)

        # Build results
        weights = np.exp(log_weights - log_sum_exp(log_weights))

        return SMCResults(
            particles=thetas,
            weights=weights,
            log_evidence=self._log_evidence_acc,
            ess_history=self._ess_history.copy(),
            acceptance_rates=self._acceptance_rates.copy(),
            n_steps=t_total,
        )

    def _extend_pf(
        self,
        pf: InternalPF,
        y_t: NDArray[np.floating[Any]],
    ) -> float:
        """Extend an internal PF by one observation.

        Parameters
        ----------
        pf : InternalPF
            Internal particle filter.
        y_t : NDArray
            Observation at time t.

        Returns
        -------
        float
            Log incremental likelihood.
        """
        return pf.extend(y_t)

    def _rejuvenate_theta(
        self,
        thetas: NDArray[np.floating[Any]],
        pfs: list[InternalPF],
        endog_so_far: NDArray[np.floating[Any]],
        model: Any,
        k_params: int,
    ) -> tuple[
        NDArray[np.floating[Any]],
        list[InternalPF],
        float,
    ]:
        """Rejuvenate parameter particles via MCMC.

        For each theta particle:
        1. Propose theta* from RW-MH
        2. Run full PF for theta* to get log p(y_{1:t} | theta*)
        3. Accept/reject with MH ratio

        Parameters
        ----------
        thetas : NDArray, shape (N_theta, k)
            Current parameter particles.
        pfs : list[InternalPF]
            Internal PFs (one per theta).
        endog_so_far : NDArray, shape (t, k_obs)
            Observations processed so far.
        model : StateSpaceModel
            State-space model.
        k_params : int
            Number of parameters.

        Returns
        -------
        tuple of (thetas, pfs, acceptance_rate)
        """
        cov = np.cov(thetas.T) + 1e-6 * np.eye(k_params)
        cov *= 2.38**2 / k_params

        n_accepted = 0

        for i in range(self.n_theta):
            for _ in range(self.n_mcmc_moves):
                # Current log-posterior
                log_prior_current = self.prior.logpdf(thetas[i])
                log_lik_current = pfs[i].log_evidence

                # Propose
                z = self._rng.standard_normal(k_params)
                chol = np.linalg.cholesky(cov)
                theta_prop = thetas[i] + chol @ z

                log_prior_prop = self.prior.logpdf(theta_prop)

                if np.isfinite(log_prior_prop):
                    # Run full PF for proposed theta
                    log_lik_prop = self._run_full_pf(model, theta_prop, endog_so_far)

                    # MH acceptance ratio
                    log_alpha = (log_prior_prop + log_lik_prop) - (
                        log_prior_current + log_lik_current
                    )

                    if np.log(self._rng.uniform()) < log_alpha:
                        thetas[i] = theta_prop
                        # Create new PF with accepted theta
                        new_pf = InternalPF(
                            n_x=self.n_x,
                            model=model,
                            theta=theta_prop,
                            rng=np.random.default_rng(self._make_seed()),
                        )
                        new_pf.run_full(endog_so_far)
                        pfs[i] = new_pf
                        n_accepted += 1

        total = self.n_theta * self.n_mcmc_moves
        acc_rate = n_accepted / total if total > 0 else 0.0
        return thetas, pfs, acc_rate

    def _run_full_pf(
        self,
        model: Any,
        theta: NDArray[np.floating[Any]],
        endog: NDArray[np.floating[Any]],
    ) -> float:
        """Run a full particle filter for given theta.

        Parameters
        ----------
        model : StateSpaceModel
            State-space model.
        theta : NDArray, shape (k_params,)
            Parameter vector.
        endog : NDArray, shape (T, k_obs)
            Observations.

        Returns
        -------
        float
            Log marginal likelihood log p(y_{1:T} | theta).
        """
        pf = InternalPF(
            n_x=self.n_x,
            model=model,
            theta=theta,
            rng=np.random.default_rng(self._make_seed()),
        )
        return pf.run_full(endog)
