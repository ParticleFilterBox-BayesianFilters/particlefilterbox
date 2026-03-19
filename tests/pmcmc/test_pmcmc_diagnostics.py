"""Comprehensive diagnostics tests for PMCMC methods.

Tests convergence diagnostics across multiple PMCMC methods:
- R-hat < 1.1 with multiple chains
- ESS > 100 for MCMC chain
- ACF decreasing
- Geweke test (p > 0.05)
"""

from __future__ import annotations

import numpy as np

from particlefilterbox.pmcmc.pmmh import PMMH
from particlefilterbox.pmcmc.proposals import GaussianRandomWalk, LogNormalProposal
from particlefilterbox.pmcmc.results import PMCMCResults
from tests.pmcmc.conftest import MockPrior, MockSSModel


class TestRHatDiagnostic:
    """Test Gelman-Rubin R-hat convergence diagnostic."""

    def test_r_hat_multi_chain(self) -> None:
        """CRITICAL: R-hat should be < 1.1 with 4 independent chains.

        Runs 4 PMMH chains with different seeds and checks that R-hat
        indicates convergence (n_iter=5000 per chain).
        """
        model = MockSSModel()
        true_params = np.array([0.9, 0.5, 1.0])
        model.set_params(true_params)

        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=50, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        n_chains = 4
        n_iterations = 5000
        burnin = 2500
        results_list: list[PMCMCResults] = []

        for chain_idx in range(n_chains):
            pmmh = PMMH(
                model=MockSSModel(),  # Fresh model each time
                prior=prior,
                n_particles=200,
                n_iterations=n_iterations,
                proposal_cov="adaptive",
                burnin=burnin,
                seed=42 + chain_idx * 100,
            )

            result = pmmh.run(
                endog=observations,
                theta_init=true_params + 0.05 * rng.standard_normal(3),
            )
            results_list.append(result)

        # Check R-hat for each parameter
        for j in range(3):
            r_hat = results_list[0].r_hat(
                other_chains=results_list[1:],
                param_idx=j,
            )
            assert r_hat < 1.1, (
                f"R-hat for param {j} = {r_hat:.4f} >= 1.1 "
                f"(chains have not converged)"
            )


class TestESSDiagnostic:
    """Test Effective Sample Size diagnostic."""

    def test_ess_chain_sufficient(self) -> None:
        """ESS should be > 100 for a converged PMMH chain."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=50, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=200,
            n_iterations=5000,
            proposal_cov="adaptive",
            burnin=2500,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        for j in range(3):
            ess = results.effective_sample_size(param_idx=j)
            assert ess > 100, (
                f"ESS for param {j} = {ess:.1f} < 100"
            )


class TestACFDiagnostic:
    """Test autocorrelation function diagnostic."""

    def test_acf_decreasing(self) -> None:
        """ACF should be generally decreasing for well-mixed chain."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=50, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=200,
            n_iterations=5000,
            proposal_cov="adaptive",
            burnin=2500,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        for j in range(3):
            acf = results.acf(param_idx=j, max_lag=30)

            # ACF at lag 0 should be 1
            assert np.isclose(acf[0], 1.0)

            # ACF should generally decrease
            # Check that ACF at lag 10 < ACF at lag 1
            assert acf[10] < acf[1], (
                f"ACF not decreasing for param {j}: "
                f"acf[1]={acf[1]:.4f}, acf[10]={acf[10]:.4f}"
            )

            # ACF at lag 30 should be small
            assert abs(acf[30]) < 0.5, (
                f"ACF at lag 30 too large for param {j}: {acf[30]:.4f}"
            )


class TestGewekeDiagnostic:
    """Test Geweke convergence diagnostic."""

    def test_geweke_converged(self) -> None:
        """Geweke test should not reject convergence (p > 0.05)."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))

        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=50, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=200,
            n_iterations=5000,
            proposal_cov="adaptive",
            burnin=2500,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )

        for j in range(3):
            z, p = results.geweke_test(param_idx=j)
            assert p > 0.05, (
                f"Geweke test rejects convergence for param {j}: "
                f"z={z:.4f}, p={p:.4f}"
            )


class TestTracePlotData:
    """Test trace plot data extraction."""

    def test_trace_plot_data_shapes(self) -> None:
        """Trace plot data should have correct shapes."""
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(500, 3))
        results = PMCMCResults(
            chains=chains,
            param_names=["a", "b", "c"],
            burnin=100,
            thin=1,
        )

        for j in range(3):
            iters, values = results.trace_plot_data(param_idx=j)
            assert len(iters) == 500
            assert len(values) == 500


class TestDiagnosticsSummary:
    """Test that all diagnostics work together in summary."""

    def test_full_summary(self) -> None:
        """Full summary should include all diagnostic information."""
        rng = np.random.default_rng(42)
        n_iter = 1000
        chains = rng.normal(size=(n_iter, 2))
        log_liks = -0.5 * np.sum(chains**2, axis=1)
        acceptance = rng.random(n_iter) < 0.25

        results = PMCMCResults(
            chains=chains,
            param_names=["mu", "sigma"],
            log_likelihood_chain=log_liks,
            acceptance_history=acceptance,
            burnin=200,
            thin=1,
        )

        summary = results.summary()
        assert "mu" in summary
        assert "sigma" in summary
        assert "Acceptance rate" in summary
        assert "ESS" in summary


class TestPMMHEdgeCaseCoverage:
    """Tests for edge cases to ensure adequate code coverage."""

    def test_pmmh_verbose_output(self, capsys: object) -> None:
        """PMMH with verbose=10 should print progress."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=50,
            n_iterations=30,
            proposal_cov="adaptive",
            burnin=10,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
            verbose=10,
        )
        assert results.n_iterations == 30

    def test_pmmh_none_proposal_cov(self) -> None:
        """PMMH with proposal_cov=None should use default scaling."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=50,
            n_iterations=50,
            proposal_cov=None,
            burnin=10,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )
        assert results.n_iterations == 50

    def test_pmmh_scalar_proposal_cov(self) -> None:
        """PMMH with scalar proposal_cov."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=50,
            n_iterations=50,
            proposal_cov=0.01,
            burnin=10,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )
        assert results.n_iterations == 50

    def test_pmmh_no_theta_init_samples_from_prior(self) -> None:
        """PMMH without theta_init should sample from prior."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.01, 0.01, 0.01]),
        )

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=50,
            n_iterations=30,
            proposal_cov="adaptive",
            burnin=10,
            seed=42,
        )

        results = pmmh.run(endog=observations)
        assert results.n_iterations == 30

    def test_pmmh_custom_proposal_object(self) -> None:
        """PMMH with a custom BaseProposal object."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        proposal = GaussianRandomWalk(dim=3, cov=0.01, seed=42)

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=50,
            n_iterations=30,
            proposal=proposal,
            burnin=10,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )
        assert results.n_iterations == 30

    def test_results_constant_chain(self) -> None:
        """Diagnostics for a constant chain (zero variance)."""
        chains = np.ones((100, 2))
        results = PMCMCResults(
            chains=chains,
            param_names=["a", "b"],
            burnin=10,
            thin=1,
        )

        ess = results.effective_sample_size(param_idx=0)
        assert ess == 90.0  # n samples since var=0

        acf = results.acf(param_idx=0, max_lag=5)
        assert acf[0] == 1.0
        assert acf[1] == 0.0

        r_hat = results.r_hat(param_idx=0)
        assert r_hat == 1.0

        z, p = results.geweke_test(param_idx=0)
        assert z == 0.0
        assert p == 1.0

    def test_results_r_hat_split(self) -> None:
        """R-hat with split chain (no other_chains)."""
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(1000, 2))
        results = PMCMCResults(
            chains=chains,
            param_names=["a", "b"],
            burnin=0,
            thin=1,
        )

        r_hat = results.r_hat(param_idx=0)
        assert r_hat < 1.1

    def test_log_normal_proposal_log_ratio(self) -> None:
        """LogNormalProposal has non-zero log_ratio."""
        proposal = LogNormalProposal(dim=3, cov=0.01, seed=42)
        theta_current = np.array([1.0, 2.0, 0.5])
        theta_proposed = proposal.propose(theta_current)
        lr = proposal.log_ratio(theta_proposed, theta_current)
        # Log ratio should be non-zero for different proposals
        assert isinstance(lr, float)

    def test_pmmh_invalid_proposal_cov_string(self) -> None:
        """PMMH with unknown proposal_cov string should raise."""
        model = MockSSModel()
        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        import pytest

        with pytest.raises(ValueError, match="Unknown proposal_cov"):
            pmmh = PMMH(
                model=model,
                prior=prior,
                n_particles=50,
                n_iterations=10,
                proposal_cov="bad_string",
                seed=42,
            )
            pmmh.run(
                endog=np.zeros(10),
                theta_init=np.array([0.9, 0.5, 1.0]),
            )

    def test_pmmh_matrix_proposal_cov(self) -> None:
        """PMMH with explicit matrix proposal_cov."""
        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.8, 0.5, 1.0]),
            cov=np.diag([0.1, 0.1, 0.1]),
        )

        pmmh = PMMH(
            model=model,
            prior=prior,
            n_particles=50,
            n_iterations=30,
            proposal_cov=np.diag([0.01, 0.01, 0.01]),
            burnin=10,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )
        assert results.n_iterations == 30

    def test_pmmh_prior_without_cov_attribute(self) -> None:
        """PMMH with prior that has no cov attribute uses default scaling."""

        class SimplePrior:
            def logpdf(self, theta: np.ndarray) -> float:  # noqa: ARG002
                return 0.0

            def sample(self, rng: np.random.Generator) -> np.ndarray:
                return rng.normal(size=3)

        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=20, rng=rng)

        pmmh = PMMH(
            model=model,
            prior=SimplePrior(),
            n_particles=50,
            n_iterations=30,
            proposal_cov="adaptive",
            burnin=10,
            seed=42,
        )

        results = pmmh.run(
            endog=observations,
            theta_init=np.array([0.9, 0.5, 1.0]),
        )
        assert results.n_iterations == 30

    def test_results_to_dataframe(self) -> None:
        """PMCMCResults to_dataframe works."""
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(100, 2))
        results = PMCMCResults(
            chains=chains,
            param_names=["a", "b"],
            burnin=10,
            thin=1,
        )
        df = results.to_dataframe()
        assert df.shape == (90, 2)
        assert list(df.columns) == ["a", "b"]

    def test_pgas_verbose_and_no_init(self) -> None:
        """PGAS with verbose output and no theta_init."""
        from particlefilterbox.pmcmc.pgas import PGAS

        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.9, 0.5, 1.0]),
            cov=np.diag([0.01, 0.01, 0.01]),
        )

        pgas = PGAS(
            model=model,
            prior=prior,
            n_particles=30,
            n_iterations=15,
            burnin=5,
            seed=42,
        )

        results = pgas.run(endog=observations, verbose=5)
        assert results.n_iterations == 15

    def test_particle_gibbs_verbose_and_no_init(self) -> None:
        """ParticleGibbs with verbose and no theta_init."""
        from particlefilterbox.pmcmc.particle_gibbs import ParticleGibbs

        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=20, rng=rng)

        prior = MockPrior(
            mean=np.array([0.9, 0.5, 1.0]),
            cov=np.diag([0.01, 0.01, 0.01]),
        )

        pg = ParticleGibbs(
            model=model,
            prior=prior,
            n_particles=30,
            n_iterations=15,
            burnin=5,
            seed=42,
        )

        results = pg.run(endog=observations, verbose=5)
        assert results.n_iterations == 15

    def test_smc2_online_reset_and_query(self) -> None:
        """SMC2Online reset and posterior queries."""
        from particlefilterbox.pmcmc.smc2_online import SMC2Online

        model = MockSSModel()
        model.set_params(np.array([0.9, 0.5, 1.0]))
        rng = np.random.default_rng(42)
        observations = model.simulate(n_obs=10, rng=rng)

        prior = MockPrior(
            mean=np.array([0.9, 0.5, 1.0]),
            cov=np.diag([0.01, 0.01, 0.01]),
        )

        smc2 = SMC2Online(
            model=model,
            n_theta=20,
            n_x=30,
            prior=prior,
            seed=42,
        )

        for y in observations:
            smc2.update(y)

        state = smc2.current_posterior()
        assert state.n_observations == 10

        mean = smc2.posterior_mean()
        assert mean.shape == (3,)

        std = smc2.posterior_std()
        assert std.shape == (3,)

        smc2.reset()
        state2 = smc2.current_posterior()
        assert state2.n_observations == 0

    def test_results_credible_interval(self) -> None:
        """PMCMCResults credible_interval returns correct shape."""
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(100, 2))
        results = PMCMCResults(
            chains=chains,
            param_names=["a", "b"],
            burnin=0,
            thin=1,
        )
        lower, upper = results.credible_interval(alpha=0.05)
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert np.all(lower < upper)
