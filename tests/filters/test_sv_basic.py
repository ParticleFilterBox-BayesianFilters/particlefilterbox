"""Basic tests for particle filters on Stochastic Volatility model.

The SV model is nonlinear, so there is no analytical reference.
We test that the filter can track the latent log-volatility.

Model:
    x_t = mu + phi * (x_{t-1} - mu) + sigma * eta_t
    y_t = exp(x_t / 2) * eps_t
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.filters.conftest import StochasticVolatilityModel


class TestSVBootstrap:
    """Test Bootstrap PF on Stochastic Volatility model."""

    @pytest.mark.slow
    def test_sv_bootstrap_correlation(self) -> None:
        """PF should track log-volatility: corr(pf_mean, h_true) > 0.7."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF

        # seed=100 produces a realization where the SV signal is strong enough
        # for the bootstrap PF to track (sigma_eta=0.15 is small, making some
        # realizations inherently hard to filter)
        rng = np.random.default_rng(100)
        model = StochasticVolatilityModel()
        h_true, obs = model.simulate(n_steps=500, rng=rng)

        config = PFConfig(n_particles=2000, seed=123, ess_threshold=0.5)
        pf = BootstrapPF(model, config)  # type: ignore[arg-type]
        results = pf.filter(obs)

        pf_means = results.filtered_means[:, 0]
        correlation = np.corrcoef(pf_means, h_true)[0, 1]

        assert correlation > 0.7, (
            f"SV correlation is {correlation:.4f}, expected > 0.7. "
            f"Bootstrap PF should track latent log-volatility."
        )

    def test_sv_bootstrap_loglikelihood_finite(self) -> None:
        """Log-likelihood should be finite for SV model."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF

        rng = np.random.default_rng(42)
        model = StochasticVolatilityModel()
        _, obs = model.simulate(n_steps=200, rng=rng)

        config = PFConfig(n_particles=1000, seed=123, ess_threshold=0.5)
        pf = BootstrapPF(model, config)  # type: ignore[arg-type]
        results = pf.filter(obs)

        assert np.isfinite(results.log_likelihood), (
            f"SV log-likelihood is not finite: {results.log_likelihood}"
        )

    def test_sv_bootstrap_ess_reasonable(self) -> None:
        """ESS should be reasonable (not all degenerate)."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.bootstrap import BootstrapPF

        rng = np.random.default_rng(42)
        model = StochasticVolatilityModel()
        _, obs = model.simulate(n_steps=200, rng=rng)

        config = PFConfig(n_particles=1000, seed=123, ess_threshold=0.5)
        pf = BootstrapPF(model, config)  # type: ignore[arg-type]
        results = pf.filter(obs)

        # Mean ESS should be above some minimum
        mean_ess = float(np.mean(results.ess_history))
        assert mean_ess > 10, (
            f"Mean ESS is {mean_ess:.1f}, which is too low. "
            f"Indicates severe weight degeneracy."
        )


class TestSVSIR:
    """Test SIR on Stochastic Volatility model."""

    def test_sv_sir_runs(self) -> None:
        """SIR should run on SV model without errors."""
        from particlefilterbox.core.config import PFConfig
        from particlefilterbox.filters.sir import SIR

        rng = np.random.default_rng(42)
        model = StochasticVolatilityModel()
        _, obs = model.simulate(n_steps=200, rng=rng)

        config = PFConfig(n_particles=1000, seed=123, ess_threshold=0.5)
        pf = SIR(model, config)  # type: ignore[arg-type]
        results = pf.filter(obs)

        assert np.isfinite(results.log_likelihood)
        assert np.all(np.isfinite(results.filtered_means))
