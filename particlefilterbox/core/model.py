"""ParticleFilterModel - Abstract base class for particle filter models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class ParticleFilterModel(ABC):
    """Abstract base class for particle filter models.

    Every particle filter model must inherit from this class and implement
    the three core methods: transition, log_observation_likelihood, and
    initial_distribution.

    Attributes
    ----------
    k_states : int
        Dimension of the state space.
    k_obs : int
        Dimension of the observation space.
    param_names : list[str]
        Names of model parameters.
    params : dict
        Current parameter values.
    """

    k_states: int
    k_obs: int

    @property
    def param_names(self) -> list[str]:
        """Names of model parameters."""
        return list(self.params.keys())

    @property
    def params(self) -> dict[str, float]:
        """Current parameter values. Subclasses should override."""
        return {}

    # --- Required methods (subclass MUST implement) ---

    @abstractmethod
    def transition(
        self,
        particles: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Propagate particles forward: x_t ~ p(x_t | x_{t-1}).

        Parameters
        ----------
        particles : ndarray, shape (N, k_states)
            Current particle positions.
        t : int
            Time step.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        ndarray, shape (N, k_states)
            New particle positions.
        """

    @abstractmethod
    def log_observation_likelihood(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t) for each particle.

        Parameters
        ----------
        particles : ndarray, shape (N, k_states)
            Particle positions.
        y_t : ndarray
            Observation at time t.
        t : int
            Time step.

        Returns
        -------
        ndarray, shape (N,)
            Log-likelihood for each particle.
        """

    @abstractmethod
    def initial_distribution(
        self,
        n_particles: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample from the prior: x_0 ~ p(x_0).

        Parameters
        ----------
        n_particles : int
            Number of particles to sample.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        ndarray, shape (N, k_states)
            Initial particle positions.
        """

    # --- Optional methods (subclass MAY override) ---

    def proposal(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Proposal distribution q(x_t | x_{t-1}, y_t).

        Default: use transition (bootstrap filter).

        Parameters
        ----------
        particles : ndarray, shape (N, k_states)
            Current particle positions.
        y_t : ndarray
            Observation at time t.
        t : int
            Time step.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        tuple[ndarray, ndarray]
            (new_particles, log_proposal_density) both shape (N, k_states) and (N,).
        """
        new_particles = self.transition(particles, t, rng)
        # For bootstrap filter, proposal = transition, so log_proposal cancels
        log_proposal = np.zeros(particles.shape[0])
        return new_particles, log_proposal

    def log_transition_density(
        self,
        x_new: NDArray[np.float64],
        x_old: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        """Compute log p(x_t | x_{t-1}).

        Required for SIR with proposal != transition.
        Default: raises NotImplementedError.

        Parameters
        ----------
        x_new : ndarray, shape (N, k_states)
            New particle positions.
        x_old : ndarray, shape (N, k_states)
            Previous particle positions.
        t : int
            Time step.

        Returns
        -------
        ndarray, shape (N,)
            Log-transition densities.
        """
        msg = "log_transition_density not implemented. Required for non-bootstrap proposals."
        raise NotImplementedError(msg)

    def has_linear_substate(self) -> bool:
        """Return True if part of the state is linear (for Rao-Blackwellization)."""
        return False
