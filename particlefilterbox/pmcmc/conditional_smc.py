"""Conditional Sequential Monte Carlo (CSMC).

SMC algorithm conditioned on a reference trajectory. The N-th particle
is fixed to follow the reference path at each time step, while the
remaining N-1 particles are propagated as in a standard bootstrap PF.

This is a key building block for Particle Gibbs methods.

References:
    Andrieu, C., Doucet, A. & Holenstein, R. (2010). Particle Markov chain
    Monte Carlo methods. JRSS-B, 72(3), 269-342.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["ConditionalSMC", "CSMCResult"]


@dataclass
class CSMCResult:
    """Result of a Conditional SMC run.

    Attributes
    ----------
    trajectory : NDArray[np.float64]
        Sampled state trajectory of shape ``(T,)`` or ``(T, d_x)``.
    particles : NDArray[np.float64]
        Final particles of shape ``(N, d_x)`` or ``(N,)``.
    weights : NDArray[np.float64]
        Final normalized weights of shape ``(N,)``.
    log_likelihood : float
        Estimated log marginal likelihood.
    ancestors : NDArray[np.int64]
        Ancestor indices at each time step, shape ``(T, N)``.
    """

    trajectory: NDArray[np.float64]
    particles: NDArray[np.float64]
    weights: NDArray[np.float64]
    log_likelihood: float
    ancestors: NDArray[np.int64]


class ConditionalSMC:
    """Conditional SMC (CSMC) algorithm.

    Runs a bootstrap particle filter conditioned on a reference trajectory.
    Particle N is fixed to follow x_ref at each time step.

    Parameters
    ----------
    model : Any
        State-space model. Must implement:
        - ``initial_sample(n_particles, rng)``: Sample initial states.
        - ``transition_sample(x_prev, rng)``: Propagate states forward.
        - ``observation_logpdf(y_t, x_t)``: Log observation density.
    n_particles : int
        Number of particles (including the reference).
    """

    def __init__(self, model: Any, n_particles: int = 100) -> None:
        self.model = model
        self.n_particles = n_particles

    def run(
        self,
        endog: NDArray[np.float64],
        theta: NDArray[np.float64],
        x_ref: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> CSMCResult:
        """Run Conditional SMC.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observations of shape ``(T,)`` or ``(T, d_y)``.
        theta : NDArray[np.float64]
            Parameter vector. Model is set to these parameters.
        x_ref : NDArray[np.float64]
            Reference trajectory of shape ``(T,)`` or ``(T, d_x)``.
        rng : np.random.Generator | None
            Random number generator.

        Returns
        -------
        CSMCResult
            CSMC result with sampled trajectory and diagnostics.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Set model parameters
        self.model.set_params(theta)

        endog = np.asarray(endog, dtype=np.float64)
        x_ref = np.asarray(x_ref, dtype=np.float64)
        t_len = len(endog)
        n = self.n_particles

        # Determine state dimension
        if x_ref.ndim == 1:
            d_x = 1
            x_ref_2d = x_ref.reshape(-1, 1)
        else:
            d_x = x_ref.shape[1]
            x_ref_2d = x_ref

        # Storage
        all_particles = np.zeros((t_len, n, d_x))
        all_ancestors = np.zeros((t_len, n), dtype=np.int64)
        log_likelihood = 0.0

        # --- Time t=0: Initialize ---
        # Particles 0..N-2: sample from initial distribution
        if hasattr(self.model, "initial_sample"):
            x_init = self.model.initial_sample(n - 1, rng)
            if x_init.ndim == 1:
                x_init = x_init.reshape(-1, 1)
        else:
            x_init = rng.standard_normal((n - 1, d_x))

        # Particle N-1 (last): fixed to reference
        particles = np.zeros((n, d_x))
        particles[: n - 1] = x_init
        particles[n - 1] = x_ref_2d[0]

        # Compute weights at t=0
        log_weights = self._compute_log_weights(endog[0], particles, d_x, n)

        # Normalize
        weights, ll_inc = self._normalize_weights(log_weights, n)
        log_likelihood += ll_inc

        all_particles[0] = particles.copy()
        all_ancestors[0] = np.arange(n)

        # --- Time t=1..T-1 ---
        for t in range(1, t_len):
            # Resample (for particles 0..N-2 only)
            ancestors = np.zeros(n, dtype=np.int64)
            ancestors[: n - 1] = rng.choice(n, size=n - 1, p=weights)
            ancestors[n - 1] = n - 1  # Reference keeps its own ancestor

            # Propagate
            new_particles = np.zeros((n, d_x))

            for j in range(n - 1):
                parent = particles[ancestors[j]]
                if hasattr(self.model, "transition_sample"):
                    x_p = parent[0] if d_x == 1 else parent
                    x_new = self.model.transition_sample(x_p, rng)
                    if np.isscalar(x_new):
                        new_particles[j, 0] = x_new
                    else:
                        new_particles[j] = x_new
                else:
                    new_particles[j] = parent + rng.standard_normal(d_x)

            # Fix reference particle
            new_particles[n - 1] = x_ref_2d[t]

            particles = new_particles

            # Compute weights
            log_weights = self._compute_log_weights(endog[t], particles, d_x, n)

            # Normalize
            weights, ll_inc = self._normalize_weights(log_weights, n)
            log_likelihood += ll_inc

            all_particles[t] = particles.copy()
            all_ancestors[t] = ancestors

        # Sample a trajectory by tracing back ancestors
        trajectory = self._trace_trajectory(all_particles, all_ancestors, weights, rng)

        # Squeeze if d_x == 1
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

    def _compute_log_weights(
        self,
        y_t: Any,
        particles: NDArray[np.float64],
        d_x: int,
        n: int,
    ) -> NDArray[np.float64]:
        """Compute log observation weights for all particles.

        Parameters
        ----------
        y_t : Any
            Observation at time t.
        particles : NDArray[np.float64]
            Current particles, shape ``(N, d_x)``.
        d_x : int
            State dimension.
        n : int
            Number of particles.

        Returns
        -------
        NDArray[np.float64]
            Log weights of shape ``(N,)``.
        """
        log_weights = np.zeros(n)
        for j in range(n):
            if hasattr(self.model, "observation_logpdf"):
                x_j = particles[j, 0] if d_x == 1 else particles[j]
                log_weights[j] = self.model.observation_logpdf(y_t, x_j)
            else:
                log_weights[j] = 0.0
        return log_weights

    @staticmethod
    def _normalize_weights(
        log_weights: NDArray[np.float64], n: int
    ) -> tuple[NDArray[np.float64], float]:
        """Normalize log weights and return increment to log-likelihood.

        Parameters
        ----------
        log_weights : NDArray[np.float64]
            Unnormalized log weights.
        n : int
            Number of particles.

        Returns
        -------
        tuple[NDArray[np.float64], float]
            Normalized weights and log-likelihood increment.
        """
        max_lw = np.max(log_weights)
        weights_raw = np.exp(log_weights - max_lw)
        sum_w = np.sum(weights_raw)
        if sum_w < 1e-300:
            return np.ones(n) / n, 0.0
        ll_inc = max_lw + np.log(sum_w) - np.log(n)
        return weights_raw / sum_w, ll_inc

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
            Final normalized weights, shape ``(N,)``.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray[np.float64]
            Sampled trajectory of shape ``(T, d_x)``.
        """
        t_len, _, d_x = all_particles.shape

        # Sample final particle index
        idx = int(rng.choice(len(final_weights), p=final_weights))

        trajectory = np.zeros((t_len, d_x))
        trajectory[t_len - 1] = all_particles[t_len - 1, idx]

        # Trace backwards
        for t in range(t_len - 2, -1, -1):
            idx = int(all_ancestors[t + 1, idx])
            trajectory[t] = all_particles[t, idx]

        return trajectory

    def _propagate_conditional(
        self,
        particles: NDArray[np.float64],
        x_ref_t: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Propagate particles with conditional constraint.

        The last particle is fixed to x_ref_t.

        Parameters
        ----------
        particles : NDArray[np.float64]
            Current particles, shape ``(N, d_x)``.
        x_ref_t : NDArray[np.float64]
            Reference state at time t, shape ``(d_x,)``.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray[np.float64]
            Propagated particles with last fixed to reference.
        """
        n = particles.shape[0]
        d_x = particles.shape[1] if particles.ndim > 1 else 1
        new_particles = np.zeros_like(particles)

        for j in range(n - 1):
            if hasattr(self.model, "transition_sample"):
                x_p = particles[j, 0] if d_x == 1 else particles[j]
                x_new = self.model.transition_sample(x_p, rng)
                if np.isscalar(x_new):
                    new_particles[j, 0] = x_new
                else:
                    new_particles[j] = x_new
            else:
                new_particles[j] = particles[j] + rng.standard_normal(d_x)

        # Fix last particle
        new_particles[n - 1] = x_ref_t

        return new_particles
