"""Systematic comparison of all particle filters.

Tests that all filters converge on a linear Gaussian model and that
advanced filters generally match or exceed Bootstrap performance.
"""

from __future__ import annotations

import numpy as np
import pytest


class LinearGaussianModelFull:
    """Full-interface linear Gaussian model for all filter types.

    x_t = phi * x_{t-1} + sigma_x * eps
    y_t = x_t + sigma_y * eps
    """

    def __init__(
        self,
        phi: float = 0.95,
        sigma_x: float = 1.0,
        sigma_y: float = 0.2,
    ) -> None:
        self.phi = phi
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.k_states = 1
        self.k_obs = 1

    def initial_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> np.ndarray:
        return rng.normal(0, 1, size=(n_particles, 1))

    def transition(
        self, particles: np.ndarray, t: int, rng: np.random.Generator
    ) -> np.ndarray:
        return self.phi * particles + rng.normal(
            0, self.sigma_x, size=particles.shape
        )

    def transition_function(self, x: np.ndarray, t: int) -> np.ndarray:
        return self.phi * np.atleast_1d(x)

    def transition_mean(self, particles: np.ndarray, t: int) -> np.ndarray:
        return self.phi * particles

    def observation_function(self, x: np.ndarray, t: int) -> np.ndarray:
        return np.atleast_1d(x)[: self.k_obs]

    def observation_mean(self, particles: np.ndarray, t: int) -> np.ndarray:
        return particles[:, : self.k_obs]

    def Q(self, t: int) -> np.ndarray:
        return np.array([[self.sigma_x**2]])

    def R(self, t: int) -> np.ndarray:
        return np.array([[self.sigma_y**2]])

    def process_noise_cov(self, t: int) -> np.ndarray:
        return self.Q(t)

    def observation_noise_cov(self, t: int) -> np.ndarray:
        return self.R(t)

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        diff = particles[:, 0] - y_t[0]
        return (
            -0.5 * diff**2 / self.sigma_y**2
            - 0.5 * np.log(2 * np.pi * self.sigma_y**2)
        )

    def optimal_proposal_params(
        self, particles: np.ndarray, observation: np.ndarray, t: int
    ) -> tuple[np.ndarray, np.ndarray]:
        n = particles.shape[0]
        prec_x = 1.0 / self.sigma_x**2
        prec_y = 1.0 / self.sigma_y**2
        P = 1.0 / (prec_x + prec_y)
        pred_mean = self.phi * particles
        y = observation.flatten()[0]
        means = P * (pred_mean * prec_x + y * prec_y)
        covs = np.full((n, 1, 1), P)
        return means, covs

    def predictive_log_likelihood(
        self, observation: np.ndarray, particles: np.ndarray, t: int
    ) -> np.ndarray:
        pred_mean = self.phi * particles[:, 0]
        pred_var = self.sigma_x**2 + self.sigma_y**2
        diff = observation.flatten()[0] - pred_mean
        return (
            -0.5 * diff**2 / pred_var - 0.5 * np.log(2 * np.pi * pred_var)
        )


def generate_data(
    T: int = 100,
    phi: float = 0.95,
    sigma_x: float = 1.0,
    sigma_y: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.empty(T)
    obs = np.empty(T)
    x = rng.normal(0, 1)
    for t in range(T):
        x = phi * x + sigma_x * rng.normal()
        states[t] = x
        obs[t] = x + sigma_y * rng.normal()
    return states, obs.reshape(-1, 1)


def _make_config(
    n_particles: int = 2000,
    ess_threshold: float = 0.5,
    resampling: str = "systematic",
    seed: int = 42,
) -> "PFConfig":
    from particlefilterbox.core.config import PFConfig

    return PFConfig(
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        resampling=resampling,
        seed=seed,
    )


class TestAllFiltersConvergeLinear:
    """All filters should converge on linear Gaussian model (corr > 0.95)."""

    @pytest.fixture
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        return generate_data(T=100, seed=42)

    def test_bootstrap_converges(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        from particlefilterbox.filters import BootstrapPF

        true_states, obs = data
        model = LinearGaussianModelFull()
        config = _make_config()

        pf = BootstrapPF(model=model, config=config)
        result = pf.filter(obs)
        corr = np.corrcoef(true_states, result.filtered_means[:, 0])[0, 1]
        assert corr > 0.95, f"Bootstrap corr={corr:.4f}"

    def test_auxiliary_converges(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        from particlefilterbox.filters import AuxiliaryPF

        true_states, obs = data
        model = LinearGaussianModelFull()
        config = _make_config()

        pf = AuxiliaryPF(model=model, config=config)
        result = pf.filter(obs)
        corr = np.corrcoef(true_states, result.filtered_means[:, 0])[0, 1]
        assert corr > 0.95, f"Auxiliary corr={corr:.4f}"

    def test_locally_optimal_converges(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        from particlefilterbox.filters import LocallyOptimalPF

        true_states, obs = data
        model = LinearGaussianModelFull()
        config = _make_config()

        pf = LocallyOptimalPF(model=model, config=config)
        result = pf.filter(obs)
        corr = np.corrcoef(true_states, result.filtered_means[:, 0])[0, 1]
        assert corr > 0.95, f"LocallyOptimal corr={corr:.4f}"

    def test_regularized_converges(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        from particlefilterbox.filters import RegularizedPF

        true_states, obs = data
        model = LinearGaussianModelFull()
        config = _make_config()

        pf = RegularizedPF(model=model, config=config, kernel="gaussian")
        result = pf.filter(obs)
        corr = np.corrcoef(true_states, result.filtered_means[:, 0])[0, 1]
        assert corr > 0.95, f"Regularized corr={corr:.4f}"

    def test_unscented_converges(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        from particlefilterbox.filters import UnscentedPF

        true_states, obs = data
        model = LinearGaussianModelFull()
        config = _make_config()

        pf = UnscentedPF(model=model, config=config)
        result = pf.filter(obs)
        corr = np.corrcoef(true_states, result.filtered_means[:, 0])[0, 1]
        assert corr > 0.95, f"Unscented corr={corr:.4f}"

    def test_ensemble_converges(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        from particlefilterbox.filters import EnsemblePF

        true_states, obs = data
        model = LinearGaussianModelFull()
        config = _make_config()

        pf = EnsemblePF(model=model, config=config)
        result = pf.filter(obs)
        corr = np.corrcoef(true_states, result.filtered_means[:, 0])[0, 1]
        assert corr > 0.95, f"Ensemble corr={corr:.4f}"

    def test_guided_converges(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        from particlefilterbox.filters import GuidedPF

        true_states, obs = data
        model = LinearGaussianModelFull()
        config = _make_config()

        pf = GuidedPF(model=model, config=config, guide_mode="linearization")
        result = pf.filter(obs)
        corr = np.corrcoef(true_states, result.filtered_means[:, 0])[0, 1]
        assert corr > 0.95, f"Guided corr={corr:.4f}"


class TestLogLikelihoodOrdering:
    """Advanced filters should match or exceed Bootstrap in mean log-likelihood."""

    def test_loglike_ordering(self) -> None:
        from particlefilterbox.filters import AuxiliaryPF, BootstrapPF, LocallyOptimalPF

        # Use moderate SNR so log-likelihoods are meaningfully different
        model = LinearGaussianModelFull(sigma_x=0.5, sigma_y=1.0)
        config = _make_config()

        _, obs = generate_data(T=100, sigma_x=0.5, sigma_y=1.0, seed=42)

        bpf = BootstrapPF(model=model, config=config)
        apf = AuxiliaryPF(model=model, config=config)
        lopf = LocallyOptimalPF(model=model, config=config)

        result_bpf = bpf.filter(obs)
        result_apf = apf.filter(obs)
        result_lopf = lopf.filter(obs)

        mean_ll_bpf = np.mean(result_bpf.log_likelihoods)
        mean_ll_apf = np.mean(result_apf.log_likelihoods)
        mean_ll_lopf = np.mean(result_lopf.log_likelihoods)

        # Advanced filters should generally have >= Bootstrap log-likelihood
        # Allow small tolerance for stochastic variation
        assert mean_ll_apf >= mean_ll_bpf - 0.5, (
            f"APF loglik ({mean_ll_apf:.4f}) should be >= Bootstrap ({mean_ll_bpf:.4f}) - 0.5"
        )
        assert mean_ll_lopf >= mean_ll_bpf - 0.5, (
            f"LOPF loglik ({mean_ll_lopf:.4f}) should be >= Bootstrap ({mean_ll_bpf:.4f}) - 0.5"
        )


class TestESSOrdering:
    """Test ESS ordering: advanced filters should have better ESS."""

    def test_ess_ordering(self) -> None:
        from particlefilterbox.filters import AuxiliaryPF, BootstrapPF, LocallyOptimalPF

        # Use low SNR so Bootstrap has weight degeneracy (ESS < N)
        model = LinearGaussianModelFull(sigma_x=0.5, sigma_y=1.0)
        config = _make_config(ess_threshold=0.1)

        _, obs = generate_data(T=100, sigma_x=0.5, sigma_y=1.0, seed=42)

        bpf = BootstrapPF(model=model, config=config)
        apf = AuxiliaryPF(model=model, config=config)
        lopf = LocallyOptimalPF(model=model, config=config)

        result_bpf = bpf.filter(obs)
        result_apf = apf.filter(obs)
        result_lopf = lopf.filter(obs)

        mean_ess_bpf = np.mean(result_bpf.ess_history)
        mean_ess_apf = np.mean(result_apf.ess_history)
        mean_ess_lopf = np.mean(result_lopf.ess_history)

        # APF ESS >= Bootstrap ESS (pre-selection helps)
        assert mean_ess_apf >= mean_ess_bpf * 0.95, (
            f"APF ESS ({mean_ess_apf:.1f}) should be >= 0.95 * Bootstrap ESS ({mean_ess_bpf:.1f})"
        )

        # LocallyOptimalPF ESS >= Bootstrap ESS (optimal proposal; allow tolerance)
        assert mean_ess_lopf >= mean_ess_bpf * 0.90, (
            f"LOPF ESS ({mean_ess_lopf:.1f}) should be >= 0.90 * Bootstrap ESS ({mean_ess_bpf:.1f})"
        )


class TestAllFiltersImportable:
    """Verify all filters are importable from the filters module."""

    def test_all_imports(self) -> None:
        from particlefilterbox.filters import (
            AuxiliaryPF,
            BaseParticleFilter,
            BootstrapPF,
            EnsemblePF,
            GuidedPF,
            LocallyOptimalPF,
            RaoBlackwellizedPF,
            RegularizedPF,
            SIR,
            UnscentedPF,
        )

        assert BaseParticleFilter is not None
        assert BootstrapPF is not None
        assert SIR is BootstrapPF
        assert AuxiliaryPF is not None
        assert LocallyOptimalPF is not None
        assert RegularizedPF is not None
        assert RaoBlackwellizedPF is not None
        assert UnscentedPF is not None
        assert EnsemblePF is not None
        assert GuidedPF is not None

    def test_all_in_all(self) -> None:
        import particlefilterbox.filters as filters_mod

        expected = [
            "BaseParticleFilter",
            "BootstrapPF",
            "SIR",
            "AuxiliaryPF",
            "LocallyOptimalPF",
            "RegularizedPF",
            "RaoBlackwellizedPF",
            "UnscentedPF",
            "EnsemblePF",
            "GuidedPF",
        ]
        for name in expected:
            assert name in filters_mod.__all__, f"{name} not in __all__"
