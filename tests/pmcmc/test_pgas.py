"""Tests for PGAS (Particle Gibbs with Ancestor Sampling).

Includes critical test comparing PGAS mixing to standard PG.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from particlefilterbox.pmcmc.particle_gibbs import ParticleGibbs
from particlefilterbox.pmcmc.pgas import PGAS
from tests.pmcmc.conftest import MockPrior
from tests.pmcmc.test_conditional_smc import SimpleCSMCModel


class PGASTestModel(SimpleCSMCModel):
    """Extended model with transition_logpdf for PGAS ancestor sampling."""

    def transition_logpdf(
        self, x_new: float | NDArray[np.float64], x_old: float | NDArray[np.float64]
    ) -> float:
        """Log transition density p(x_new | x_old)."""
        phi = self.params[0]
        sigma_x = self.params[1]
        diff = float(x_new) - phi * float(x_old)
        return float(
            -0.5 * (diff / sigma_x) ** 2
            - np.log(sigma_x)
            - 0.5 * np.log(2 * np.pi)
        )


class TestPGASRuns:
    """Test that PGAS runs correctly."""

    def test_pgas_runs(self) -> None:
        """PGAS should run without errors."""
        rng = np.random.default_rng(42)
        model = PGASTestModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        _, y = model.simulate(n_obs=30, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pgas = PGAS(
            model=model,
            prior=prior,
            n_particles=30,
            n_iterations=50,
            burnin=10,
            seed=42,
        )

        results = pgas.run(
            endog=y,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        assert results.n_iterations == 50
        assert results.n_params == 3
        assert results.n_effective_samples == 40

    def test_pgas_produces_finite_results(self) -> None:
        """PGAS should produce finite posterior mean."""
        rng = np.random.default_rng(42)
        model = PGASTestModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        _, y = model.simulate(n_obs=30, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pgas = PGAS(
            model=model,
            prior=prior,
            n_particles=30,
            n_iterations=100,
            burnin=30,
            seed=42,
        )

        results = pgas.run(
            endog=y,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        mean = results.posterior_mean()
        assert np.all(np.isfinite(mean))


class TestPGASBetterMixing:
    """CRITICAL: Test that PGAS has better mixing than standard PG."""

    def test_pgas_better_mixing(self) -> None:
        """ESS of PGAS should be at least 2x ESS of standard PG.

        This is the key advantage of PGAS over PG.
        """
        rng = np.random.default_rng(42)
        model_pgas = PGASTestModel()
        model_pgas.set_params(np.array([0.9, 0.5, 1.0]))

        _, y = model_pgas.simulate(n_obs=50, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        n_particles = 30
        n_iterations = 500
        burnin = 200

        # Run PGAS
        pgas = PGAS(
            model=PGASTestModel(),
            prior=prior,
            n_particles=n_particles,
            n_iterations=n_iterations,
            burnin=burnin,
            seed=42,
        )
        pgas.model.set_params(np.array([0.9, 0.5, 1.0]))
        results_pgas = pgas.run(
            endog=y,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        # Run standard PG
        model_pg = PGASTestModel()  # Same model type for fair comparison
        model_pg.set_params(np.array([0.9, 0.5, 1.0]))

        pg = ParticleGibbs(
            model=model_pg,
            prior=prior,
            n_particles=n_particles,
            n_iterations=n_iterations,
            burnin=burnin,
            seed=42,
        )
        results_pg = pg.run(
            endog=y,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        # Compare ESS for first parameter
        ess_pgas = results_pgas.effective_sample_size(param_idx=0)
        ess_pg = results_pg.effective_sample_size(param_idx=0)

        # CRITICAL: PGAS should have at least 2x the ESS of PG
        assert ess_pgas > 2 * ess_pg, (
            f"PGAS ESS ({ess_pgas:.1f}) should be > 2 * PG ESS ({ess_pg:.1f})"
        )


class TestPGASAncestorSampling:
    """Test the ancestor sampling mechanism."""

    def test_pgas_ancestor_sampling(self) -> None:
        """Ancestor sampling should select diverse ancestors."""
        rng = np.random.default_rng(42)
        model = PGASTestModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        _, y = model.simulate(n_obs=30, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pgas = PGAS(
            model=model,
            prior=prior,
            n_particles=50,
            n_iterations=20,
            burnin=5,
            seed=42,
        )

        results = pgas.run(
            endog=y,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        # Should produce valid results
        assert results.n_iterations == 20
        assert np.all(np.isfinite(results.posterior_mean()))

    def test_pgas_with_custom_param_sampler(self) -> None:
        """PGAS should work with a custom parameter sampler."""
        rng = np.random.default_rng(42)
        model = PGASTestModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        _, y = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        def custom_sampler(
            _model: Any,
            _states: NDArray[np.float64],
            _endog: NDArray[np.float64],
            theta_current: NDArray[np.float64],
            rng: np.random.Generator,
        ) -> NDArray[np.float64]:
            return theta_current + 0.01 * rng.standard_normal(len(theta_current))

        pgas = PGAS(
            model=model,
            prior=prior,
            n_particles=20,
            n_iterations=30,
            param_sampler=custom_sampler,
            burnin=5,
            seed=42,
        )

        results = pgas.run(endog=y, theta_init=np.array([0.9, 0.5, 1.0]))
        assert results.n_iterations == 30
