"""Solution script for Notebook 1: Particle filter on a New Keynesian DSGE.

Runs a Bootstrap particle filter on synthetic treasury-yield data simulated
from a 3-equation New Keynesian DSGE, computes the filtered states and their
RMSE against the ground truth, and writes the summary to
``results_dsge.csv``.

Usage
-----
    python solution_01_dsge.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters import BootstrapPF

SEED: int = 20260422
N_PARTICLES: int = 2000
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = Path(__file__).resolve().parent / "results_dsge.csv"


class NewKeynesianDSGE(ParticleFilterModel):
    """Latent IS / Phillips / Taylor system with Gaussian observation noise."""

    k_states = 3
    k_obs = 3

    def __init__(self) -> None:
        self.rho_x = 0.9
        self.rho_pi = 0.5
        self.kappa = 0.3
        self.rho_r = 0.8
        self.phi_pi = 0.5
        self.phi_x = 0.2
        self.sigma_x = 0.1
        self.sigma_pi = 0.1
        self.sigma_r = 0.05
        self.R_std: NDArray[np.float64] = np.array([0.20, 0.15, 0.10])

    @property
    def params(self) -> dict[str, float]:
        return {
            "rho_x": self.rho_x,
            "rho_pi": self.rho_pi,
            "kappa": self.kappa,
            "rho_r": self.rho_r,
            "phi_pi": self.phi_pi,
            "phi_x": self.phi_x,
            "sigma_x": self.sigma_x,
            "sigma_pi": self.sigma_pi,
            "sigma_r": self.sigma_r,
        }

    def initial_distribution(
        self,
        n_particles: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        return rng.normal(0.0, 0.3, size=(n_particles, self.k_states))

    def transition(
        self,
        particles: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        x_prev = particles[:, 0]
        pi_prev = particles[:, 1]
        r_prev = particles[:, 2]
        n = particles.shape[0]
        eps_x = rng.standard_normal(n) * self.sigma_x
        eps_pi = rng.standard_normal(n) * self.sigma_pi
        eps_r = rng.standard_normal(n) * self.sigma_r
        x_new = self.rho_x * x_prev + eps_x
        pi_new = self.rho_pi * pi_prev + self.kappa * x_new + eps_pi
        r_new = self.rho_r * r_prev + self.phi_pi * pi_new + self.phi_x * x_new + eps_r
        return np.column_stack([x_new, pi_new, r_new])

    def log_observation_likelihood(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        diff = particles - y_t
        var = self.R_std ** 2
        ll = -0.5 * np.sum(diff ** 2 / var, axis=1)
        ll -= 0.5 * float(np.sum(np.log(2 * np.pi * var)))
        return ll


def main() -> None:
    df = pd.read_csv(DATA_DIR / "treasury_yields.csv")
    obs_cols = ["output_gap_obs", "inflation_obs", "interest_rate_obs"]
    true_cols = ["output_gap_true", "inflation_true", "interest_rate_true"]
    Y = df[obs_cols].to_numpy(dtype=float)
    X_true = df[true_cols].to_numpy(dtype=float)

    model = NewKeynesianDSGE()
    config = PFConfig(
        n_particles=N_PARTICLES,
        resampling="systematic",
        ess_threshold=0.5,
        seed=SEED,
    )
    pf = BootstrapPF(model=model, config=config)

    t_start = time.perf_counter()
    result = pf.filter(Y)
    elapsed = time.perf_counter() - t_start

    filtered = result.filtered_means
    filtered_var = np.clip(np.diagonal(result.filtered_covs, axis1=1, axis2=2), 0.0, None)
    filtered_std = np.sqrt(filtered_var)

    rmse_obs = np.sqrt(np.mean((Y - X_true) ** 2, axis=0))
    rmse_filt = np.sqrt(np.mean((filtered - X_true) ** 2, axis=0))
    names = ["output_gap", "inflation", "interest_rate"]

    out = pd.DataFrame({
        "t": df["t"].to_numpy(),
        "output_gap_obs": Y[:, 0],
        "inflation_obs": Y[:, 1],
        "interest_rate_obs": Y[:, 2],
        "output_gap_true": X_true[:, 0],
        "inflation_true": X_true[:, 1],
        "interest_rate_true": X_true[:, 2],
        "output_gap_filt": filtered[:, 0],
        "inflation_filt": filtered[:, 1],
        "interest_rate_filt": filtered[:, 2],
        "output_gap_filt_std": filtered_std[:, 0],
        "inflation_filt_std": filtered_std[:, 1],
        "interest_rate_filt_std": filtered_std[:, 2],
        "ess": result.ess_history,
        "resampled": result.resampled.astype(int),
    })
    out.attrs["log_likelihood"] = float(result.log_likelihood)
    out.to_csv(OUT_PATH, index=False)

    print("=" * 64)
    print("DSGE particle filter - solution_01_dsge.py")
    print("=" * 64)
    print(f"observations        : {len(df)}")
    print(f"particles           : {N_PARTICLES}")
    print(f"log-likelihood      : {result.log_likelihood:.2f}")
    print(f"mean ESS            : {result.ess_history.mean():.1f}")
    print(f"resample share      : {result.resampled.mean():.1%}")
    print(f"elapsed             : {elapsed:.2f} s")
    print("-" * 64)
    for name, r_obs, r_filt in zip(names, rmse_obs, rmse_filt, strict=True):
        red = 100.0 * (1.0 - r_filt / r_obs)
        print(f"{name:>16s}: RMSE obs={r_obs:.4f}  RMSE filt={r_filt:.4f}  red={red:+.1f}%")
    print("-" * 64)
    print(f"CSV written to      : {OUT_PATH}")


if __name__ == "__main__":
    main()
