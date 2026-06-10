"""
DSGE (Dynamic Stochastic General Equilibrium) models for particle filtering.

Supports:
- First-order (linear) approximation: should use Kalman Filter
- Second-order (nonlinear): needs particle filter
- Zero Lower Bound (ZLB) constraint
- Impulse response functions via PF simulation

References:
- Fernandez-Villaverde & Rubio-Ramirez (2007)
- An & Schorfheide (2007)
- Herbst & Schorfheide (2015)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class DSGE:
    """DSGE model for particle filtering.

    Parameters
    ----------
    A : NDArray
        State transition matrix, shape (k_states, k_states).
    B : NDArray
        Shock impact matrix, shape (k_states, k_shocks).
    C : NDArray
        Observation matrix, shape (k_obs, k_states).
    Z : NDArray | None
        Shock-to-observation matrix, shape (k_obs, k_shocks).
    H : NDArray | None
        Measurement error std, shape (k_obs, k_obs) or (k_obs,).
    order : int
        Approximation order (1 = linear, 2 = quadratic). Default 1.
    zlb : bool
        Whether to apply Zero Lower Bound. Default False.
    zlb_index : int
        Index of the interest rate in observation vector. Default 0.
    sigma_scale : float
        Perturbation parameter for second-order terms. Default 1.0.
    quadratic_terms : NDArray | None
        Tensor for second-order correction, shape (k_states, k_states, k_states).
    params : dict[str, float] | None
        Additional model parameters.
    """

    def __init__(
        self,
        A: NDArray[np.float64] | None = None,
        B: NDArray[np.float64] | None = None,
        C: NDArray[np.float64] | None = None,
        Z: NDArray[np.float64] | None = None,
        H: NDArray[np.float64] | None = None,
        order: int = 1,
        zlb: bool = False,
        zlb_index: int = 0,
        sigma_scale: float = 1.0,
        quadratic_terms: NDArray[np.float64] | None = None,
        params: dict[str, float] | None = None,
    ) -> None:
        if A is None:
            # Default: simple 3-equation New Keynesian
            A = np.array([
                [0.8, 0.1, 0.0],
                [-0.2, 0.9, 0.3],
                [0.0, -0.1, 0.7],
            ])
        if B is None:
            B = np.eye(A.shape[0]) * 0.1
        if C is None:
            C = np.eye(A.shape[0])

        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.Z = np.asarray(Z, dtype=np.float64) if Z is not None else None
        self.H = np.asarray(H, dtype=np.float64) if H is not None else None
        self.order = order
        self.zlb = zlb
        self.zlb_index = zlb_index
        self.sigma_scale = sigma_scale
        self.quadratic_terms = quadratic_terms
        self.params = params if params is not None else {}

        self.k_states = self.A.shape[0]
        self.k_shocks = self.B.shape[1]
        self.k_obs = self.C.shape[0]

        self.param_names = list(self.params.keys()) if self.params else [
            "sigma_scale"
        ]

    @classmethod
    def from_matrices(
        cls,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        C: NDArray[np.float64],
        Z: NDArray[np.float64] | None = None,
        H: NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> DSGE:
        """Create DSGE model from system matrices.

        Parameters
        ----------
        A : NDArray
            State transition matrix.
        B : NDArray
            Shock impact matrix.
        C : NDArray
            Observation matrix.
        Z : NDArray | None
            Shock-to-observation matrix.
        H : NDArray | None
            Measurement error.
        **kwargs
            Additional arguments passed to constructor.

        Returns
        -------
        DSGE
            Configured DSGE model.
        """
        return cls(A=A, B=B, C=C, Z=Z, H=H, **kwargs)

    def _quadratic_term(
        self, x: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute second-order correction term.

        Parameters
        ----------
        x : NDArray
            State vector, shape (n_particles, k_states).

        Returns
        -------
        NDArray
            Quadratic correction, shape (n_particles, k_states).
        """
        n = x.shape[0]
        if self.quadratic_terms is not None:
            # quadratic_terms shape: (k_states, k_states, k_states)
            # For each state i: sum_{j,k} Q[i,j,k] * x_j * x_k
            correction = np.zeros((n, self.k_states))
            for i in range(self.k_states):
                for j in range(self.k_states):
                    for k in range(self.k_states):
                        correction[:, i] += (
                            self.quadratic_terms[i, j, k]
                            * x[:, j]
                            * x[:, k]
                        )
            return correction
        else:
            # Default: simple diagonal quadratic term
            return 0.1 * x**2

    def _apply_zlb(
        self, y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply Zero Lower Bound constraint.

        Parameters
        ----------
        y : NDArray
            Observations, shape (n, k_obs) or (k_obs,).

        Returns
        -------
        NDArray
            Constrained observations.
        """
        y_constrained = y.copy()
        if y.ndim == 1:
            y_constrained[self.zlb_index] = max(
                0.0, y_constrained[self.zlb_index]
            )
        else:
            y_constrained[:, self.zlb_index] = np.maximum(
                0.0, y_constrained[:, self.zlb_index]
            )
        return y_constrained

    def default_prior(self) -> dict[str, dict[str, Any]]:
        """Return default prior distributions."""
        priors: dict[str, dict[str, Any]] = {
            "sigma_scale": {
                "distribution": "inverse_gamma",
                "a": 3.0,
                "b": 1.0,
            },
        }
        return priors

    def initial_state(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample initial state from stationary distribution."""
        return rng.standard_normal((n_particles, self.k_states)) * 0.01

    def transition(
        self,
        state: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Propagate state forward one step.

        Parameters
        ----------
        state : NDArray
            Current state, shape (n_particles, k_states).
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray
            Next state, shape (n_particles, k_states).
        """
        n = state.shape[0]
        eps = rng.standard_normal((n, self.k_shocks))

        # First order: x_t = A x_{t-1} + B eps_t
        x_next = state @ self.A.T + eps @ self.B.T

        if self.order >= 2:
            # Second order correction
            quad = self._quadratic_term(state)
            x_next += 0.5 * self.sigma_scale**2 * quad

        return x_next

    def log_observation_density(
        self,
        y: float | NDArray[np.float64],
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log p(y_t | x_t).

        Parameters
        ----------
        y : float or NDArray
            Observation at time t.
        state : NDArray
            Particles, shape (n_particles, k_states).

        Returns
        -------
        NDArray
            Log-density for each particle, shape (n_particles,).
        """
        y_arr = np.atleast_1d(y).astype(np.float64)

        # Expected observation: C @ x_t
        y_hat = state @ self.C.T  # (n, k_obs)

        if self.zlb:
            y_hat = self._apply_zlb(y_hat)

        # Measurement error
        if self.H is not None:
            obs_std = self.H if self.H.ndim == 1 else np.sqrt(np.diag(self.H @ self.H.T))
        else:
            obs_std = np.full(self.k_obs, 0.1)

        # Log-density
        diff = y_arr[np.newaxis, :] - y_hat  # (n, k_obs)
        log_dens = np.sum(
            -0.5 * np.log(2 * np.pi)
            - np.log(obs_std[np.newaxis, :])
            - 0.5 * (diff / obs_std[np.newaxis, :]) ** 2,
            axis=1,
        )
        return log_dens

    def impulse_response(
        self,
        shock: int | NDArray[np.float64],
        periods: int = 40,
        n_particles: int = 1000,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute impulse response function via PF simulation.

        Parameters
        ----------
        shock : int or NDArray
            If int, index of shock to apply (unit shock).
            If NDArray, the shock vector of shape (k_shocks,).
        periods : int
            Number of periods. Default 40.
        n_particles : int
            Number of particles for simulation. Default 1000.
        seed : int | None
            Random seed.

        Returns
        -------
        NDArray
            IRF, shape (periods, k_states).
        """
        rng = np.random.default_rng(seed)

        if isinstance(shock, int):
            shock_vec = np.zeros(self.k_shocks)
            shock_vec[shock] = 1.0
        else:
            shock_vec = np.asarray(shock)

        # Baseline (no shock)
        baseline = np.zeros((periods, n_particles, self.k_states))
        state_base = self.initial_state(n_particles, rng)
        rng_base = np.random.default_rng(seed)
        for t in range(periods):
            state_base = self.transition(state_base, rng_base)
            baseline[t] = state_base

        # Shocked path
        shocked = np.zeros((periods, n_particles, self.k_states))
        rng_shock = np.random.default_rng(seed)
        state_shock = self.initial_state(n_particles, rng_shock)
        # Apply shock at t=0
        state_shock += shock_vec[np.newaxis, :] @ self.B.T
        for t in range(periods):
            state_shock = self.transition(state_shock, rng_shock)
            shocked[t] = state_shock

        # IRF = mean difference
        irf = np.mean(shocked, axis=1) - np.mean(baseline, axis=1)
        return irf

    def simulate(
        self, T: int, seed: int | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate from the model.

        Parameters
        ----------
        T : int
            Number of time steps.
        seed : int | None
            Random seed.

        Returns
        -------
        dict
            Keys: 'observations' shape (T, k_obs),
                  'states' shape (T, k_states).
        """
        rng = np.random.default_rng(seed)

        states = np.zeros((T, self.k_states))
        observations = np.zeros((T, self.k_obs))

        x = rng.standard_normal(self.k_states) * 0.01
        for t in range(T):
            eps = rng.standard_normal(self.k_shocks)
            x = self.A @ x + self.B @ eps
            if self.order >= 2:
                x_2d = x.reshape(1, -1)
                quad = self._quadratic_term(x_2d).flatten()
                x += 0.5 * self.sigma_scale**2 * quad

            y = self.C @ x
            if self.Z is not None:
                y += self.Z @ eps
            if self.H is not None:
                if self.H.ndim == 1:
                    y += self.H * rng.standard_normal(self.k_obs)
                else:
                    y += self.H @ rng.standard_normal(self.k_obs)

            if self.zlb:
                y[self.zlb_index] = max(0.0, y[self.zlb_index])

            states[t] = x
            observations[t] = y

        return {"observations": observations, "states": states}
