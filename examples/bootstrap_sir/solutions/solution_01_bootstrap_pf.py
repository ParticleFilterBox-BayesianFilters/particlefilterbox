"""Solution: Bootstrap Particle Filter on Linear-Gaussian model.

Runs Bootstrap PF with multiple particle counts and compares to Kalman filter.
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

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.bootstrap import BootstrapPF


class LinearGaussianModel(ParticleFilterModel):
    """Linear-Gaussian state-space model for particle filtering."""

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
        noise = rng.normal(0.0, self.sigma_x, size=particles.shape)
        return self.phi * particles + noise

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        residual = y_t[0] - particles[:, 0]
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

    # Initialize from stationary distribution
    x_pred = 0.0
    P_pred = Q / (1.0 - phi**2)

    for t in range(T):
        # Update
        K = P_pred / (P_pred + R)
        x_filt[t] = x_pred + K * (y[t] - x_pred)
        P_filt[t] = (1.0 - K) * P_pred

        # Predict next
        x_pred = phi * x_filt[t]
        P_pred = phi**2 * P_filt[t] + Q

    return x_filt, P_filt


def main() -> None:
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    df = pd.read_csv(os.path.join(data_dir, "simulated_linear_gaussian.csv"))
    y_obs = df["y_obs"].values
    x_true = df["x_true"].values

    # Model parameters
    phi = 0.95
    sigma_x = 0.5
    sigma_y = 1.0

    # Kalman benchmark
    x_kalman, P_kalman = kalman_filter(y_obs, phi, sigma_x, sigma_y)
    rmse_kalman = float(np.sqrt(np.mean((x_kalman - x_true) ** 2)))
    print(f"Kalman Filter RMSE: {rmse_kalman:.4f}")

    # Bootstrap PF with various N
    results: list[dict[str, float]] = []
    N_values = [100, 500, 1000, 5000]

    for N in N_values:
        config = PFConfig(n_particles=N, resampling="systematic", seed=42)
        pf = BootstrapPF(LinearGaussianModel(phi, sigma_x, sigma_y), config)

        t0 = time.time()
        res = pf.filter(y_obs)
        elapsed = time.time() - t0

        pf_mean = res.filtered_means[:, 0]
        rmse = float(np.sqrt(np.mean((pf_mean - x_true) ** 2)))
        mean_ess = float(np.mean(res.ess_history))
        log_lik = float(res.log_likelihood)

        results.append({
            "N": N,
            "RMSE": rmse,
            "RMSE_Kalman": rmse_kalman,
            "ESS_mean": mean_ess,
            "ESS_N_pct": mean_ess / N * 100,
            "log_likelihood": log_lik,
            "time_s": elapsed,
        })
        print(
            f"N={N:5d}: RMSE={rmse:.4f}, ESS_mean={mean_ess:.1f} "
            f"({mean_ess / N * 100:.1f}%), time={elapsed:.2f}s"
        )

    # Verify convergence: RMSE should decrease toward Kalman as N grows
    rmse_values = [r["RMSE"] for r in results]
    print(f"\nConvergence check: RMSE sequence = {[f'{v:.4f}' for v in rmse_values]}")
    print(f"Kalman RMSE = {rmse_kalman:.4f}")
    if rmse_values[-1] < rmse_values[0]:
        print("OK: RMSE decreases with N (converging toward Kalman)")
    else:
        print("WARNING: RMSE did not decrease with N")

    # Save results
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "results_bootstrap_pf.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
