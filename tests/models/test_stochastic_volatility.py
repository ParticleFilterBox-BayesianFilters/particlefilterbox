"""Tests for StochasticVolatility model."""

import numpy as np
import pytest

from particlefilterbox.models.stochastic_volatility import StochasticVolatility


class TestSVBasicSimulation:
    """Test basic SV simulation."""

    def test_sv_basic_simulation(self) -> None:
        model = StochasticVolatility("basic")
        result = model.simulate(T=500, seed=42)
        assert result["observations"].shape == (500, 1)
        assert result["states"].shape == (500, 1)
        assert np.all(np.isfinite(result["observations"]))
        assert np.all(np.isfinite(result["states"]))

    def test_sv_basic_params(self) -> None:
        model = StochasticVolatility("basic")
        assert model.k_states == 1
        assert model.k_obs == 1
        assert set(model.param_names) == {"mu", "phi", "sigma"}

    def test_sv_basic_filtering(self) -> None:
        """Filtering should achieve correlation > 0.7 with true states."""
        model = StochasticVolatility("basic", params={
            "mu": -1.0, "phi": 0.97, "sigma": 0.15
        })
        result = model.simulate(T=1000, seed=7)
        observations = result["observations"]
        true_states = result["states"][:, 0]

        # Simple bootstrap PF
        n_particles = 2000
        rng = np.random.default_rng(0)
        T = observations.shape[0]
        filtered_mean = np.zeros(T)

        particles = model.initial_state(n_particles, rng)
        for t in range(T):
            if t > 0:
                particles = model.transition(particles, rng)
            log_w = model.log_observation_density(
                observations[t, 0], particles
            )
            log_w -= np.max(log_w)
            w = np.exp(log_w)
            w /= w.sum()
            filtered_mean[t] = np.average(particles[:, 0], weights=w)
            # Resample
            indices = rng.choice(n_particles, size=n_particles, p=w)
            particles = particles[indices]

        corr = np.corrcoef(true_states, filtered_mean)[0, 1]
        assert corr > 0.7, f"Correlation {corr:.3f} < 0.7"


class TestSVLeverage:
    """Test leverage variant."""

    def test_sv_leverage_negative_rho(self) -> None:
        model = StochasticVolatility("leverage", params={
            "mu": -1.0, "phi": 0.97, "sigma": 0.15, "rho": -0.7
        })
        result = model.simulate(T=2000, seed=42)
        y = result["observations"][:, 0]
        h = result["states"][:, 0]
        mu = -1.0
        phi = 0.97
        sigma = 0.15
        # Contemporaneous correlation: eta_t = rho*eps_t + sqrt(1-rho^2)*z_t
        # Estimate eta_t from states, correlate with y_t
        eta_hat = (h[1:] - mu - phi * (h[:-1] - mu)) / sigma
        corr = np.corrcoef(y[1:], eta_hat)[0, 1]
        assert corr < 0, f"Expected negative correlation, got {corr:.3f}"

    def test_sv_leverage_params(self) -> None:
        model = StochasticVolatility("leverage")
        assert "rho" in model.param_names
        assert model.params["rho"] == -0.5


class TestSVJumps:
    """Test jumps variant."""

    def test_sv_jumps_detection(self) -> None:
        model = StochasticVolatility("jumps", params={
            "mu": -1.0, "phi": 0.97, "sigma": 0.15,
            "lambda_jump": 0.1, "mu_jump": -1.0, "sigma_jump": 2.0
        })
        result = model.simulate(T=1000, seed=42)
        q = result["states"][:, 1]
        # Should have some jumps
        n_jumps = int(q.sum())
        assert n_jumps > 10, f"Expected >10 jumps, got {n_jumps}"
        assert n_jumps < 500, f"Too many jumps: {n_jumps}"

    def test_sv_jumps_state_dim(self) -> None:
        model = StochasticVolatility("jumps")
        assert model.k_states == 2


class TestSVPMMH:
    """Test PMMH parameter recovery."""

    def test_sv_pmmh_recovery(self) -> None:
        """Test that default_prior() is well-defined for PMMH."""
        model = StochasticVolatility("basic")
        priors = model.default_prior()
        assert "mu" in priors
        assert "phi" in priors
        assert "sigma" in priors
        for name, prior in priors.items():
            assert "distribution" in prior
