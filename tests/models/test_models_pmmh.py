"""Integration tests for all models - PMMH workflow and standard interface."""

import numpy as np
import pytest

from particlefilterbox.models.bounded_states import BoundedStates
from particlefilterbox.models.continuous_time import ContinuousTime
from particlefilterbox.models.count_state_space import CountStateSpace
from particlefilterbox.models.dsge import DSGE
from particlefilterbox.models.jump_diffusion import JumpDiffusion
from particlefilterbox.models.mixture import Mixture
from particlefilterbox.models.nonlinear_regime import NonlinearRegime
from particlefilterbox.models.stochastic_volatility import StochasticVolatility

ALL_MODELS = [
    StochasticVolatility("basic"),
    StochasticVolatility("leverage"),
    StochasticVolatility("jumps"),
    StochasticVolatility("factor"),
    JumpDiffusion("merton"),
    JumpDiffusion("kou"),
    JumpDiffusion("bates"),
    DSGE(),
    CountStateSpace("poisson"),
    CountStateSpace("binomial"),
    CountStateSpace("sir"),
    NonlinearRegime(),
    BoundedStates(),
    Mixture(),
    ContinuousTime("cir"),
    ContinuousTime("vasicek"),
    ContinuousTime("heston"),
]


class TestAllModelsHaveDefaultPrior:
    """Test that all models have default_prior()."""

    @pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: type(m).__name__)
    def test_all_models_have_default_prior(self, model: object) -> None:
        priors = model.default_prior()  # type: ignore[attr-defined]
        assert isinstance(priors, dict)
        assert len(priors) > 0
        for name, prior in priors.items():
            assert isinstance(name, str)
            assert isinstance(prior, dict)
            assert "distribution" in prior


class TestAllModelsHaveSimulation:
    """Test that all models have simulate()."""

    @pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: type(m).__name__)
    def test_all_models_have_simulation(self, model: object) -> None:
        result = model.simulate(T=50, seed=42)  # type: ignore[attr-defined]
        assert isinstance(result, dict)
        assert "observations" in result
        assert "states" in result
        assert result["observations"].shape[0] == 50
        assert result["states"].shape[0] == 50
        assert np.all(np.isfinite(result["observations"]))
        assert np.all(np.isfinite(result["states"]))


class TestSVFullWorkflow:
    """Test full PMMH workflow for SV model."""

    def test_sv_full_workflow(self) -> None:
        """Test simulate -> filter -> evaluate cycle."""
        model = StochasticVolatility("basic", params={
            "mu": -1.0, "phi": 0.97, "sigma": 0.15
        })

        # 1. Simulate
        result = model.simulate(T=200, seed=42)
        observations = result["observations"]
        true_states = result["states"][:, 0]  # noqa: F841

        # 2. Filter with BPF
        n_particles = 300
        rng = np.random.default_rng(123)
        n_steps = observations.shape[0]
        log_likelihood = 0.0
        particles = model.initial_state(n_particles, rng)

        for t in range(n_steps):
            if t > 0:
                particles = model.transition(particles, rng)
            log_w = model.log_observation_density(
                observations[t, 0], particles
            )
            max_lw = np.max(log_w)
            w = np.exp(log_w - max_lw)
            log_likelihood += max_lw + np.log(w.mean())
            w /= w.sum()
            indices = rng.choice(n_particles, size=n_particles, p=w)
            particles = particles[indices]

        # 3. Log-likelihood should be finite
        assert np.isfinite(log_likelihood)

        # 4. Prior should be well-defined
        priors = model.default_prior()
        assert "mu" in priors
        assert "phi" in priors
        assert "sigma" in priors
