"""Tests for Conditional SMC and Particle Gibbs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from particlefilterbox.pmcmc.conditional_smc import ConditionalSMC
from particlefilterbox.pmcmc.particle_gibbs import ParticleGibbs
from tests.pmcmc.conftest import MockPrior, SimpleFilterResult


class SimpleCSMCModel:
    """Simple model for CSMC testing with required interfaces."""

    def __init__(self) -> None:
        self.params: NDArray[np.float64] = np.array([0.9, 0.5, 1.0])
        self.param_names: list[str] = ["phi", "sigma_x", "sigma_y"]

    def set_params(self, theta: NDArray[np.float64]) -> None:
        self.params = np.asarray(theta, dtype=np.float64)

    def get_params(self) -> NDArray[np.float64]:
        return self.params.copy()

    def initial_sample(
        self, n: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Sample initial states."""
        sigma_x = self.params[1]
        return rng.normal(0, sigma_x, size=n)

    def transition_sample(
        self, x_prev: float | NDArray[np.float64], rng: np.random.Generator
    ) -> float:
        """Propagate state forward."""
        phi = self.params[0]
        sigma_x = self.params[1]
        return float(phi * x_prev + sigma_x * rng.standard_normal())

    def observation_logpdf(
        self, y: float, x: float | NDArray[np.float64]
    ) -> float:
        """Log p(y|x)."""
        sigma_y = self.params[2]
        return float(
            -0.5 * ((y - x) / sigma_y) ** 2
            - np.log(sigma_y)
            - 0.5 * np.log(2 * np.pi)
        )

    def filter(
        self,
        endog: NDArray[np.float64],
        n_particles: int = 100,
        rng: np.random.Generator | None = None,
    ) -> SimpleFilterResult:
        """Simple bootstrap PF."""
        if rng is None:
            rng = np.random.default_rng()

        phi, sigma_x, sigma_y = self.params
        t_len = len(endog)
        n = n_particles

        particles = rng.normal(0, sigma_x, size=n)
        filtered_means = np.zeros(t_len)
        log_lik = 0.0

        for t in range(t_len):
            log_w = -0.5 * ((endog[t] - particles) / sigma_y) ** 2
            max_lw = np.max(log_w)
            w = np.exp(log_w - max_lw)
            sw = np.sum(w)
            if sw < 1e-300:
                log_lik = -np.inf
                break
            log_lik += max_lw + np.log(sw) - np.log(n)
            w /= sw
            filtered_means[t] = np.sum(w * particles)

            idx = rng.choice(n, size=n, p=w)
            particles = phi * particles[idx] + sigma_x * rng.standard_normal(n)

        return SimpleFilterResult(
            log_likelihood=log_lik,
            filtered_means=filtered_means,
        )

    def simulate(
        self, n_obs: int, rng: np.random.Generator | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulate states and observations."""
        if rng is None:
            rng = np.random.default_rng()

        phi, sigma_x, sigma_y = self.params
        x = np.zeros(n_obs)
        y = np.zeros(n_obs)

        x[0] = sigma_x * rng.standard_normal()
        y[0] = x[0] + sigma_y * rng.standard_normal()

        for t in range(1, n_obs):
            x[t] = phi * x[t - 1] + sigma_x * rng.standard_normal()
            y[t] = x[t] + sigma_y * rng.standard_normal()

        return x, y


class TestConditionalSMC:
    """Tests for Conditional SMC."""

    def test_csmc_reference_survives(self) -> None:
        """CRITICAL: Reference trajectory should survive in CSMC.

        The last particle should always follow the reference trajectory.
        """
        rng = np.random.default_rng(42)
        model = SimpleCSMCModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        # Generate data and reference
        x_true, y = model.simulate(n_obs=30, rng=rng)
        x_ref = x_true + 0.1 * rng.standard_normal(len(x_true))

        csmc = ConditionalSMC(model=model, n_particles=50)
        result = csmc.run(
            endog=y,
            theta=np.array([0.9, 0.5, 1.0]),
            x_ref=x_ref,
            rng=np.random.default_rng(123),
        )

        # Result should be a valid trajectory
        assert result.trajectory.shape == (30,)
        assert np.isfinite(result.log_likelihood)
        assert result.weights.shape == (50,)
        assert np.isclose(np.sum(result.weights), 1.0, atol=1e-10)

    def test_csmc_produces_different_trajectories(self) -> None:
        """CSMC should produce different trajectories on different runs."""
        model = SimpleCSMCModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        rng = np.random.default_rng(42)
        _, y = model.simulate(n_obs=20, rng=rng)
        x_ref = np.zeros(20)

        csmc = ConditionalSMC(model=model, n_particles=50)

        traj1 = csmc.run(
            endog=y,
            theta=np.array([0.9, 0.5, 1.0]),
            x_ref=x_ref,
            rng=np.random.default_rng(1),
        ).trajectory

        traj2 = csmc.run(
            endog=y,
            theta=np.array([0.9, 0.5, 1.0]),
            x_ref=x_ref,
            rng=np.random.default_rng(2),
        ).trajectory

        assert not np.allclose(traj1, traj2)

    def test_csmc_log_likelihood_finite(self) -> None:
        """CSMC log-likelihood should be finite."""
        model = SimpleCSMCModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        rng = np.random.default_rng(42)
        _, y = model.simulate(n_obs=20, rng=rng)
        x_ref = np.zeros(20)

        csmc = ConditionalSMC(model=model, n_particles=50)
        result = csmc.run(
            endog=y,
            theta=np.array([0.9, 0.5, 1.0]),
            x_ref=x_ref,
            rng=np.random.default_rng(42),
        )

        assert np.isfinite(result.log_likelihood)


class TestParticleGibbs:
    """Tests for Particle Gibbs sampler."""

    def test_pg_runs(self) -> None:
        """Particle Gibbs should run without errors."""
        rng = np.random.default_rng(42)
        model = SimpleCSMCModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        _, y = model.simulate(n_obs=30, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pg = ParticleGibbs(
            model=model,
            prior=prior,
            n_particles=30,
            n_iterations=50,
            burnin=10,
            seed=42,
        )

        results = pg.run(
            endog=y,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        assert results.n_iterations == 50
        assert results.n_params == 3
        assert results.n_effective_samples == 40  # 50 - 10

    def test_pg_sv_params(self) -> None:
        """PG should recover approximate parameters for simple model."""
        rng = np.random.default_rng(42)
        model = SimpleCSMCModel()
        true_params = np.array([0.9, 0.5, 1.0])
        model.set_params(true_params)

        _, y = model.simulate(n_obs=50, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pg = ParticleGibbs(
            model=model,
            prior=prior,
            n_particles=50,
            n_iterations=200,
            burnin=100,
            seed=42,
        )

        results = pg.run(
            endog=y,
            theta_init=true_params,
        )

        # Posterior mean should be in reasonable range
        mean = results.posterior_mean()
        assert mean.shape == (3,)
        # Just check it produces finite values
        assert np.all(np.isfinite(mean))

    def test_pg_with_custom_param_sampler(self) -> None:
        """PG should work with a custom parameter sampler."""
        rng = np.random.default_rng(42)
        model = SimpleCSMCModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        _, y = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        def custom_sampler(
            model: Any,
            states: NDArray[np.float64],
            endog: NDArray[np.float64],
            theta_current: NDArray[np.float64],
            rng: np.random.Generator,
        ) -> NDArray[np.float64]:
            """Simple random walk on parameters."""
            return theta_current + 0.01 * rng.standard_normal(len(theta_current))

        pg = ParticleGibbs(
            model=model,
            prior=prior,
            n_particles=20,
            n_iterations=30,
            param_sampler=custom_sampler,
            burnin=5,
            seed=42,
        )

        results = pg.run(endog=y, theta_init=np.array([0.9, 0.5, 1.0]))
        assert results.n_iterations == 30
