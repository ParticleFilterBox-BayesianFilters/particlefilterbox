"""Tests for SMC2Online streaming parameter inference."""

from __future__ import annotations

import numpy as np

from particlefilterbox.pmcmc.smc2_online import SMC2Online
from tests.pmcmc.conftest import MockPrior
from tests.pmcmc.test_conditional_smc import SimpleCSMCModel


class TestSMC2OnlineRuns:
    """Test that SMC2Online runs correctly."""

    def test_smc2_online_runs(self) -> None:
        """SMC2Online should run without errors."""
        model = SimpleCSMCModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        smc2 = SMC2Online(
            model=model,
            n_theta=50,
            n_x=20,
            prior=prior,
            seed=42,
        )

        # Generate some data
        rng = np.random.default_rng(42)
        _, y = model.simulate(n_obs=20, rng=rng)

        # Process all observations
        for t in range(len(y)):
            smc2.update(y[t])

        state = smc2.current_posterior()
        assert state.n_observations == 20
        assert state.theta_particles.shape == (50, 3)
        assert np.isclose(np.sum(state.theta_weights), 1.0, atol=1e-10)

    def test_smc2_online_streaming(self) -> None:
        """SMC2Online should work in streaming mode (one obs at a time)."""
        model = SimpleCSMCModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        smc2 = SMC2Online(
            model=model,
            n_theta=30,
            n_x=10,
            prior=prior,
            seed=42,
        )

        rng = np.random.default_rng(123)
        _, y = model.simulate(n_obs=30, rng=rng)

        # Process one at a time and track posterior mean
        means = []
        for t in range(len(y)):
            smc2.update(y[t])
            means.append(smc2.posterior_mean().copy())

        means_arr = np.array(means)

        # Should have 30 posterior means
        assert means_arr.shape == (30, 3)

        # All means should be finite
        assert np.all(np.isfinite(means_arr))

        # Posterior mean should evolve over time (not constant)
        assert not np.allclose(means_arr[0], means_arr[-1])

    def test_smc2_online_reset(self) -> None:
        """Reset should re-initialize the state."""
        model = SimpleCSMCModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        smc2 = SMC2Online(
            model=model,
            n_theta=30,
            n_x=10,
            prior=prior,
            seed=42,
        )

        rng = np.random.default_rng(42)
        _, y = model.simulate(n_obs=10, rng=rng)

        for t in range(len(y)):
            smc2.update(y[t])

        assert smc2.current_posterior().n_observations == 10

        smc2.reset()

        assert smc2.current_posterior().n_observations == 0
        assert np.isclose(smc2.current_posterior().log_evidence, 0.0)

    def test_smc2_online_posterior_std(self) -> None:
        """Posterior std should be positive."""
        model = SimpleCSMCModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        smc2 = SMC2Online(
            model=model,
            n_theta=30,
            n_x=10,
            prior=prior,
            seed=42,
        )

        rng = np.random.default_rng(42)
        _, y = model.simulate(n_obs=10, rng=rng)

        for t in range(len(y)):
            smc2.update(y[t])

        std = smc2.posterior_std()
        assert np.all(std >= 0)

    def test_smc2_online_ess_history(self) -> None:
        """ESS history should be tracked."""
        model = SimpleCSMCModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        smc2 = SMC2Online(
            model=model,
            n_theta=30,
            n_x=10,
            prior=prior,
            seed=42,
        )

        rng = np.random.default_rng(42)
        _, y = model.simulate(n_obs=15, rng=rng)

        for t in range(len(y)):
            smc2.update(y[t])

        state = smc2.current_posterior()
        assert len(state.ess_history) == 15
        assert all(ess > 0 for ess in state.ess_history)
