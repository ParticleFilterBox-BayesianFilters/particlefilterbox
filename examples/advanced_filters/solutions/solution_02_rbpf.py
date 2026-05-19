"""Solution: Rao-Blackwellized PF vs Bootstrap PF on mixed linear/nonlinear model.

Runs RBPF and BPF, compares RMSE and variance reduction.
Saves results to CSV.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from kalmanbox.core import StateSpaceRepresentation
from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.bootstrap import BootstrapPF
from particlefilterbox.filters.rao_blackwellized import RaoBlackwellizedPF


class MixedLinearModel(ParticleFilterModel):
    """Mixed linear/nonlinear model for RBPF demonstration.

    Nonlinear component: stochastic scaling factor (random walk)
        s_t = s_{t-1} + sigma_s * eta_t

    Linear component: actual state conditioned on s_t
        x_t = phi * x_{t-1} + sigma_x * eps_t
        y_t = exp(s_t/10) * x_t + sigma_y * nu_t
    """

    k_states = 2
    k_obs = 1
    k_nonlinear = 1
    k_linear = 1

    def __init__(
        self,
        phi: float = 0.95,
        sigma_x: float = 0.5,
        sigma_y: float = 1.0,
        sigma_s: float = 0.02,
    ):
        self.phi = phi
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_s = sigma_s

    @property
    def params(self) -> dict[str, float]:
        return {
            "phi": self.phi,
            "sigma_x": self.sigma_x,
            "sigma_y": self.sigma_y,
            "sigma_s": self.sigma_s,
        }

    def has_linear_substate(self) -> bool:
        return True

    def initial_distribution(self, n_particles: int, rng: np.random.Generator) -> np.ndarray:
        std_x = self.sigma_x / np.sqrt(1.0 - self.phi**2)
        s0 = rng.normal(0.0, 0.1, size=(n_particles, 1))
        x0 = rng.normal(0.0, std_x, size=(n_particles, 1))
        return np.hstack([s0, x0])

    def initial_nonlinear_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> np.ndarray:
        return rng.normal(0.0, 0.1, size=(n_particles, 1))

    def initial_linear_mean(self) -> np.ndarray:
        return np.zeros(1)

    def initial_linear_cov(self) -> np.ndarray:
        var = self.sigma_x**2 / (1.0 - self.phi**2)
        return np.array([[var]])

    def transition(self, particles: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
        n = particles.shape[0]
        s = particles[:, 0]
        x = particles[:, 1] if particles.shape[1] > 1 else np.zeros(n)
        s_new = s + self.sigma_s * rng.standard_normal(n)
        x_new = self.phi * x + self.sigma_x * rng.standard_normal(n)
        return np.column_stack([s_new, x_new])

    def transition_nonlinear(
        self, particles_nl: np.ndarray, t: int, rng: np.random.Generator
    ) -> np.ndarray:
        n = particles_nl.shape[0]
        s = particles_nl[:, 0]
        s_new = s + self.sigma_s * rng.standard_normal(n)
        return s_new.reshape(-1, 1)

    def linear_ssm(self, x_nonlinear: np.ndarray) -> StateSpaceRepresentation:
        """Return SSR conditioned on nonlinear state."""
        s = float(x_nonlinear[0])
        scale = np.exp(s / 10.0)

        ssr = StateSpaceRepresentation(k_states=1, k_endog=1, k_posdef=1)
        ssr.T[:] = self.phi
        ssr.R[:] = 1.0
        ssr.Q[:] = self.sigma_x**2
        ssr.Z[:] = scale
        ssr.H[:] = self.sigma_y**2
        ssr.c[:] = 0.0
        ssr.d[:] = 0.0
        return ssr

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        if particles.shape[1] >= 2:
            s = particles[:, 0]
            x = particles[:, 1]
        else:
            s = particles[:, 0]
            x = np.zeros_like(s)
        scale = np.exp(s / 10.0)
        mean = scale * x
        var = self.sigma_y**2
        residual = float(y_t[0]) - mean
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * (residual**2) / var


class SimpleLinearGaussianModel(ParticleFilterModel):
    """Simple linear-Gaussian model for Bootstrap PF comparison."""

    k_states = 1
    k_obs = 1

    def __init__(self, phi: float = 0.95, sigma_x: float = 0.5, sigma_y: float = 1.0):
        self.phi = phi
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    @property
    def params(self) -> dict[str, float]:
        return {"phi": self.phi, "sigma_x": self.sigma_x, "sigma_y": self.sigma_y}

    def initial_distribution(self, n_particles: int, rng: np.random.Generator) -> np.ndarray:
        std = self.sigma_x / np.sqrt(1.0 - self.phi**2)
        return rng.normal(0.0, std, size=(n_particles, 1))

    def transition(self, particles: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
        return self.phi * particles + self.sigma_x * rng.standard_normal(particles.shape)

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        residual = float(y_t[0]) - particles[:, 0]
        return -0.5 * np.log(2 * np.pi * self.sigma_y**2) - 0.5 * (residual / self.sigma_y) ** 2


def kalman_filter(
    y: np.ndarray, phi: float, sigma_x: float, sigma_y: float
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical Kalman filter for the linear-Gaussian model."""
    T = len(y)
    Q = sigma_x**2
    R = sigma_y**2

    x_filt = np.zeros(T)
    P_filt = np.zeros(T)

    x_pred = 0.0
    P_pred = Q / (1.0 - phi**2)

    for t in range(T):
        K = P_pred / (P_pred + R)
        x_filt[t] = x_pred + K * (y[t] - x_pred)
        P_filt[t] = (1.0 - K) * P_pred
        x_pred = phi * x_filt[t]
        P_pred = phi**2 * P_filt[t] + Q

    return x_filt, P_filt


def main() -> None:
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "bootstrap_sir", "data")
    df = pd.read_csv(os.path.join(data_dir, "simulated_linear_gaussian.csv"))
    y_obs = df["y_obs"].values
    x_true = df["x_true"].values
    T = len(df)

    phi, sigma_x, sigma_y = 0.95, 0.5, 1.0
    print(f"Loaded {T} observations from linear-Gaussian model")

    # Kalman benchmark
    x_kalman, P_kalman = kalman_filter(y_obs, phi, sigma_x, sigma_y)
    rmse_kalman = float(np.sqrt(np.mean((x_kalman - x_true) ** 2)))
    print(f"Kalman Filter RMSE: {rmse_kalman:.4f}")

    results: list[dict[str, object]] = []
    seed = 42

    # RBPF with N=500
    model_mixed = MixedLinearModel(phi=phi, sigma_x=sigma_x, sigma_y=sigma_y, sigma_s=0.02)
    config_rbpf = PFConfig(n_particles=500, resampling="systematic", seed=seed)
    rbpf = RaoBlackwellizedPF(model=model_mixed, config=config_rbpf)

    t0 = time.time()
    res_rbpf = rbpf.filter(y_obs)
    time_rbpf = time.time() - t0
    h_rbpf = res_rbpf.filtered_means[:, 1]  # linear component
    rmse_rbpf = float(np.sqrt(np.mean((h_rbpf - x_true) ** 2)))

    results.append({
        "Filter": "RBPF",
        "N": 500,
        "RMSE": round(rmse_rbpf, 6),
        "ESS_mean": round(float(np.mean(res_rbpf.ess_history)), 2),
        "log_likelihood": round(float(res_rbpf.log_likelihood), 4),
        "time_s": round(time_rbpf, 4),
    })
    print(
        f"RBPF (N=500):  RMSE={rmse_rbpf:.4f}, ESS_mean={np.mean(res_rbpf.ess_history):.1f}, "
        f"time={time_rbpf:.2f}s"
    )

    # BPF with N=500 and N=2000
    model_simple = SimpleLinearGaussianModel(phi=phi, sigma_x=sigma_x, sigma_y=sigma_y)
    for N_bpf in [500, 2000]:
        config_bpf = PFConfig(n_particles=N_bpf, resampling="systematic", seed=seed)
        bpf = BootstrapPF(model=model_simple, config=config_bpf)

        t0 = time.time()
        res_bpf = bpf.filter(y_obs)
        time_bpf = time.time() - t0
        h_bpf = res_bpf.filtered_means[:, 0]
        rmse_bpf = float(np.sqrt(np.mean((h_bpf - x_true) ** 2)))

        results.append({
            "Filter": "BPF",
            "N": N_bpf,
            "RMSE": round(rmse_bpf, 6),
            "ESS_mean": round(float(np.mean(res_bpf.ess_history)), 2),
            "log_likelihood": round(float(res_bpf.log_likelihood), 4),
            "time_s": round(time_bpf, 4),
        })
        print(
            f"BPF (N={N_bpf}): RMSE={rmse_bpf:.4f}, ESS_mean={np.mean(res_bpf.ess_history):.1f}, "
            f"time={time_bpf:.2f}s"
        )

    # Add Kalman reference row
    results.append({
        "Filter": "Kalman",
        "N": "-",
        "RMSE": round(rmse_kalman, 6),
        "ESS_mean": "-",
        "log_likelihood": "-",
        "time_s": "-",
    })

    # Variance reduction analysis: run multiple seeds
    print("\n--- Variance Reduction Analysis (5 seeds) ---")
    n_runs = 5
    rmse_rbpf_runs: list[float] = []
    rmse_bpf500_runs: list[float] = []

    for s in range(n_runs):
        cfg_r = PFConfig(n_particles=500, resampling="systematic", seed=s + 100)

        rbpf_r = RaoBlackwellizedPF(model=model_mixed, config=cfg_r)
        res_r = rbpf_r.filter(y_obs)
        rmse_rbpf_runs.append(float(np.sqrt(np.mean((res_r.filtered_means[:, 1] - x_true) ** 2))))

        bpf_r = BootstrapPF(model=model_simple, config=cfg_r)
        res_r = bpf_r.filter(y_obs)
        rmse_bpf500_runs.append(float(np.sqrt(np.mean((res_r.filtered_means[:, 0] - x_true) ** 2))))

    var_rbpf = float(np.var(rmse_rbpf_runs))
    var_bpf = float(np.var(rmse_bpf500_runs))
    reduction_pct = (1.0 - var_rbpf / var_bpf) * 100 if var_bpf > 0 else 0.0

    print(f"  RBPF (N=500) RMSE: mean={np.mean(rmse_rbpf_runs):.4f}, std={np.std(rmse_rbpf_runs):.4f}")
    print(f"  BPF  (N=500) RMSE: mean={np.mean(rmse_bpf500_runs):.4f}, std={np.std(rmse_bpf500_runs):.4f}")
    print(f"  Variance reduction: {reduction_pct:.1f}%")

    # Save results
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "results_rbpf.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
