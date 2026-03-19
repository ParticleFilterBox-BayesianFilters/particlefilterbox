"""Tests for CountStateSpace and NonlinearRegime models."""

import numpy as np
import pytest

from particlefilterbox.models.count_state_space import CountStateSpace
from particlefilterbox.models.nonlinear_regime import NonlinearRegime


class TestPoissonFiltering:
    """Test Poisson count model."""

    def test_poisson_filtering(self) -> None:
        model = CountStateSpace("poisson")
        result = model.simulate(T=300, seed=42)
        obs = result["observations"]
        true_states = result["states"][:, 0]
        assert obs.shape == (300, 1)
        assert np.all(obs >= 0)
        assert np.all(obs == obs.astype(int))

        # Simple BPF
        n_particles = 300
        rng = np.random.default_rng(123)
        T = obs.shape[0]
        filtered = np.zeros(T)
        particles = model.initial_state(n_particles, rng)
        for t in range(T):
            if t > 0:
                particles = model.transition(particles, rng)
            log_w = model.log_observation_density(obs[t, 0], particles)
            log_w -= np.max(log_w)
            w = np.exp(log_w)
            w /= w.sum() + 1e-300
            filtered[t] = np.average(particles[:, 0], weights=w)
            idx = rng.choice(n_particles, size=n_particles, p=w)
            particles = particles[idx]

        corr = np.corrcoef(true_states, filtered)[0, 1]
        assert corr > 0.5, f"Correlation {corr:.3f} too low"


class TestSIREpidemic:
    """Test SIR epidemic model."""

    def test_sir_epidemic(self) -> None:
        model = CountStateSpace("sir", population=10000)
        result = model.simulate(T=200, seed=42)
        obs = result["observations"]
        states = result["states"]
        assert states.shape == (200, 3)
        assert obs.shape == (200, 1)
        assert np.all(obs >= 0)
        # S + I + R should be approximately N
        total = states[:, 0] + states[:, 1] + states[:, 2]
        assert np.all(total > 0)

    def test_sir_r0_estimation(self) -> None:
        model = CountStateSpace("sir", params={
            "beta": 0.4, "gamma": 0.1, "rho": 0.5,
            "sigma_S": 5.0, "sigma_I": 3.0,
        })
        r0 = model.r0()
        assert abs(r0 - 4.0) < 0.01, f"R0 should be 4.0, got {r0}"


class TestRegimeSwitching:
    """Test NonlinearRegime model."""

    def test_regime_switching_detection(self) -> None:
        model = NonlinearRegime(
            n_regimes=2,
            regime_params=[
                {"mu": 2.0, "phi": 0.9, "sigma": 0.1, "obs_sigma": 0.3},
                {"mu": -2.0, "phi": 0.9, "sigma": 0.1, "obs_sigma": 0.3},
            ],
            transition_matrix=np.array([
                [0.95, 0.05],
                [0.05, 0.95],
            ]),
        )
        result = model.simulate(T=500, seed=42)
        regimes = result["states"][:, -1]
        obs = result["observations"][:, 0]

        # Should have both regimes
        unique_regimes = np.unique(regimes)
        assert len(unique_regimes) == 2

        # Observations in regime 0 should be positive on average
        regime0_obs = obs[regimes == 0]
        regime1_obs = obs[regimes == 1]
        assert np.mean(regime0_obs) > 0
        assert np.mean(regime1_obs) < 0

    def test_regime_default_prior(self) -> None:
        model = NonlinearRegime()
        priors = model.default_prior()
        assert len(priors) > 0

    def test_regime_simulate(self) -> None:
        model = NonlinearRegime()
        result = model.simulate(T=100, seed=42)
        assert result["observations"].shape == (100, 1)
        assert result["states"].shape == (100, 2)
