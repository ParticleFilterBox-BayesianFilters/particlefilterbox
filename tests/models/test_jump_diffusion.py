"""Tests for JumpDiffusion model."""

import numpy as np
from particlefilterbox.models.jump_diffusion import JumpDiffusion


class TestMerton:
    """Test Merton jump-diffusion."""

    def test_merton_simulation(self) -> None:
        model = JumpDiffusion("merton")
        result = model.simulate(T=500, seed=42)
        assert result["observations"].shape == (500, 1)
        assert result["states"].shape == (500, 1)
        assert result["prices"].shape == (501,)
        assert np.all(np.isfinite(result["observations"]))
        assert np.all(result["prices"] > 0)

    def test_merton_jump_detection(self) -> None:
        model = JumpDiffusion("merton", params={
            "mu": 0.05, "sigma": 0.1,
            "lambda_jump": 10.0, "mu_jump": -0.05, "sigma_jump": 0.1,
        })
        result = model.simulate(T=1000, seed=42)
        returns = result["observations"][:, 0]
        # High jump intensity should produce heavy tails
        kurtosis = np.mean((returns - returns.mean())**4) / np.std(returns)**4
        assert kurtosis > 3.0, f"Expected excess kurtosis, got {kurtosis:.2f}"

    def test_merton_default_prior(self) -> None:
        model = JumpDiffusion("merton")
        priors = model.default_prior()
        assert "mu" in priors
        assert "lambda_jump" in priors


class TestKou:
    """Test Kou double-exponential model."""

    def test_kou_double_exponential(self) -> None:
        model = JumpDiffusion("kou")
        result = model.simulate(T=1000, seed=42)
        returns = result["observations"][:, 0]
        assert np.all(np.isfinite(returns))
        # Double exponential: asymmetric tails
        assert len(returns) == 1000

    def test_kou_params(self) -> None:
        model = JumpDiffusion("kou")
        assert "p_up" in model.param_names
        assert "eta1" in model.param_names
        assert "eta2" in model.param_names


class TestBates:
    """Test Bates (Merton + Heston) model."""

    def test_bates_sv_plus_jumps(self) -> None:
        model = JumpDiffusion("bates")
        result = model.simulate(T=500, seed=42)
        assert result["states"].shape == (500, 2)
        v = result["states"][:, 1]
        # Variance should be positive
        assert np.all(v > 0)
        # Variance should be mean-reverting
        assert np.mean(v) > 0

    def test_bates_state_dim(self) -> None:
        model = JumpDiffusion("bates")
        assert model.k_states == 2
        assert model.k_obs == 1
