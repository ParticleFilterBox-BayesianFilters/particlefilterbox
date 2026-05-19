"""
Solution 01: Forward Filtering Backward Smoothing (FFBS)

Runs FFBSm and FFBSi on the Stochastic Volatility model,
compares with filtering, and saves results to results_ffbs.csv.

Usage:
    python solution_01_ffbs.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from particlefilterbox.smoothers import FFBSm, FFBSi

# ── Configuration ────────────────────────────────────────────────────────
SEED = 42
N_PARTICLES = 2000
T_USE = 100
M_TRAJECTORIES = 500

# SV model parameters
MU = -1.0
PHI = 0.97
SIGMA = 0.15

OUTPUT_DIR = Path(__file__).parent
DATA_PATH = OUTPUT_DIR.parent / "data" / "simulated_sv.csv"


# ── SV Model for smoothers ──────────────────────────────────────────────
@dataclass
class SVModelForSmoothing:
    """SV model with log_transition_density for smoothers."""
    mu: float = -1.0
    phi: float = 0.97
    sigma: float = 0.15

    def log_transition_density(
        self, x_new: np.ndarray, x_old: np.ndarray, t: int
    ) -> np.ndarray:
        if x_new.ndim == 1:
            x_new = x_new.reshape(1, -1)
        if x_old.ndim == 1:
            x_old = x_old.reshape(1, -1)
        mean = self.mu + self.phi * (x_old[:, 0] - self.mu)
        return (
            -0.5 * np.log(2 * np.pi * self.sigma**2)
            - 0.5 * ((x_new[:, 0] - mean) / self.sigma) ** 2
        )

    def log_observation_density(
        self, y: np.ndarray, x: np.ndarray, t: int
    ) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h = x[:, 0]
        vol = np.exp(h / 2.0)
        y_val = float(y) if np.isscalar(y) else float(y[0])
        return -0.5 * np.log(2.0 * np.pi) - np.log(vol) - 0.5 * (y_val / vol) ** 2


# ── Filter results container ────────────────────────────────────────────
@dataclass
class FilterResultsForSmoother:
    particles_history: list = field(default_factory=list)
    weights_history: list = field(default_factory=list)
    filtered_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    filtered_cov: np.ndarray = field(default_factory=lambda: np.array([]))
    observations: np.ndarray = field(default_factory=lambda: np.array([]))
    ancestor_indices: list = field(default_factory=list)


# ── Bootstrap Particle Filter ───────────────────────────────────────────
def run_bootstrap_pf(
    y_obs: np.ndarray,
    model: SVModelForSmoothing,
    n_particles: int,
    seed: int,
) -> tuple[FilterResultsForSmoother, float]:
    """Run bootstrap PF and return filter results + elapsed time."""
    T = len(y_obs)
    rng = np.random.default_rng(seed)
    var_stat = model.sigma**2 / (1.0 - model.phi**2)
    particles = rng.normal(model.mu, np.sqrt(var_stat), size=(n_particles, 1))

    particles_history: list[np.ndarray] = []
    weights_history: list[np.ndarray] = []
    ancestor_indices_history: list[np.ndarray] = []
    filtered_mean = np.zeros((T, 1))
    filtered_cov = np.zeros((T, 1, 1))

    t0 = time.perf_counter()
    for t in range(T):
        if t > 0:
            eta = rng.standard_normal(size=(n_particles, 1))
            particles = model.mu + model.phi * (particles - model.mu) + model.sigma * eta

        h = particles[:, 0]
        vol = np.exp(h / 2.0)
        log_w = -0.5 * np.log(2.0 * np.pi) - np.log(vol) - 0.5 * (y_obs[t] / vol) ** 2
        w = np.exp(log_w - np.max(log_w))
        w_normalized = w / np.sum(w)

        # Store BEFORE resampling
        particles_history.append(particles.copy())
        weights_history.append(w_normalized.copy())

        filtered_mean[t, 0] = np.sum(w_normalized * particles[:, 0])
        diff = particles[:, 0] - filtered_mean[t, 0]
        filtered_cov[t, 0, 0] = np.sum(w_normalized * diff**2)

        # Systematic resampling
        ess = 1.0 / np.sum(w_normalized**2)
        if ess < n_particles / 2:
            cumsum = np.cumsum(w_normalized)
            u = (rng.random() + np.arange(n_particles)) / n_particles
            indices = np.searchsorted(cumsum, u)
            indices = np.clip(indices, 0, n_particles - 1)
            ancestor_indices_history.append(indices.copy())
            particles = particles[indices].copy()
        else:
            ancestor_indices_history.append(np.arange(n_particles))

    elapsed = time.perf_counter() - t0

    results = FilterResultsForSmoother(
        particles_history=particles_history,
        weights_history=weights_history,
        filtered_mean=filtered_mean,
        filtered_cov=filtered_cov,
        observations=y_obs.reshape(-1, 1),
        ancestor_indices=ancestor_indices_history,
    )
    return results, elapsed


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Solution 01: FFBS Smoothing on Stochastic Volatility Model")
    print("=" * 60)

    # Load data
    df_full = pd.read_csv(DATA_PATH)
    df = df_full.iloc[:T_USE].reset_index(drop=True)
    y_obs = df["y_obs"].values
    h_true = df["h_true"].values
    T = len(y_obs)
    print(f"\nData: T={T} observations from {DATA_PATH.name}")

    # Setup model
    model = SVModelForSmoothing(mu=MU, phi=PHI, sigma=SIGMA)
    print(f"SV parameters: mu={MU}, phi={PHI}, sigma={SIGMA}")

    # ── Step 1: Run Bootstrap PF ─────────────────────────────────────
    print(f"\n--- Bootstrap PF (N={N_PARTICLES}, seed={SEED}) ---")
    filter_results, elapsed_filter = run_bootstrap_pf(y_obs, model, N_PARTICLES, SEED)
    rmse_filter = float(np.sqrt(np.mean((filter_results.filtered_mean[:, 0] - h_true) ** 2)))
    mae_filter = float(np.mean(np.abs(filter_results.filtered_mean[:, 0] - h_true)))
    mean_var_filter = float(np.mean(filter_results.filtered_cov[:, 0, 0]))
    print(f"  Time: {elapsed_filter:.2f}s")
    print(f"  RMSE: {rmse_filter:.4f}")
    print(f"  MAE:  {mae_filter:.4f}")
    print(f"  Mean Var: {mean_var_filter:.4f}")

    # ── Step 2: Run FFBSm ────────────────────────────────────────────
    print(f"\n--- FFBSm Smoother ---")
    ffbsm = FFBSm()
    t0 = time.perf_counter()
    results_ffbsm = ffbsm.smooth(filter_results, model)
    elapsed_ffbsm = time.perf_counter() - t0
    rmse_ffbsm = float(np.sqrt(np.mean((results_ffbsm.smoothed_mean[:, 0] - h_true) ** 2)))
    mae_ffbsm = float(np.mean(np.abs(results_ffbsm.smoothed_mean[:, 0] - h_true)))
    mean_var_ffbsm = float(np.mean(results_ffbsm.smoothed_cov[:, 0, 0]))
    print(f"  Time: {elapsed_ffbsm:.2f}s")
    print(f"  RMSE: {rmse_ffbsm:.4f}")
    print(f"  MAE:  {mae_ffbsm:.4f}")
    print(f"  Mean Var: {mean_var_ffbsm:.4f}")
    print(f"  Var reduction vs filter: {(1 - mean_var_ffbsm / mean_var_filter) * 100:.1f}%")

    # ── Step 3: Run FFBSi ────────────────────────────────────────────
    print(f"\n--- FFBSi Smoother (M={M_TRAJECTORIES}) ---")
    ffbsi = FFBSi(seed=SEED)
    t0 = time.perf_counter()
    results_ffbsi = ffbsi.smooth(filter_results, model, n_trajectories=M_TRAJECTORIES)
    elapsed_ffbsi = time.perf_counter() - t0
    rmse_ffbsi = float(np.sqrt(np.mean((results_ffbsi.smoothed_mean[:, 0] - h_true) ** 2)))
    mae_ffbsi = float(np.mean(np.abs(results_ffbsi.smoothed_mean[:, 0] - h_true)))
    mean_var_ffbsi = float(np.mean(results_ffbsi.smoothed_cov[:, 0, 0]))
    print(f"  Time: {elapsed_ffbsi:.2f}s")
    print(f"  Trajectories shape: {results_ffbsi.trajectories.shape}")
    print(f"  RMSE: {rmse_ffbsi:.4f}")
    print(f"  MAE:  {mae_ffbsi:.4f}")
    print(f"  Mean Var: {mean_var_ffbsi:.4f}")
    print(f"  Var reduction vs filter: {(1 - mean_var_ffbsi / mean_var_filter) * 100:.1f}%")

    # ── Step 4: Consistency check FFBSm vs FFBSi ─────────────────────
    mean_diff = float(np.mean(np.abs(
        results_ffbsm.smoothed_mean[:, 0] - results_ffbsi.smoothed_mean[:, 0]
    )))
    print(f"\n--- Consistency Check ---")
    print(f"  Mean |FFBSm - FFBSi|: {mean_diff:.4f}")
    print(f"  FFBSm and FFBSi are {'consistent' if mean_diff < 0.5 else 'INCONSISTENT'}")

    # ── Step 5: Verify smoother variance < filter variance ───────────
    print(f"\n--- Variance Reduction Verification ---")
    assert mean_var_ffbsm < mean_var_filter, (
        f"FFBSm variance ({mean_var_ffbsm:.4f}) should be < filter ({mean_var_filter:.4f})"
    )
    assert mean_var_ffbsi < mean_var_filter, (
        f"FFBSi variance ({mean_var_ffbsi:.4f}) should be < filter ({mean_var_filter:.4f})"
    )
    print("  PASSED: Both smoother variances < filter variance")

    # ── Step 6: Save results CSV ─────────────────────────────────────
    output_path = OUTPUT_DIR / "results_ffbs.csv"

    # Per-timestep results
    results_df = pd.DataFrame({
        "t": np.arange(T),
        "h_true": h_true,
        "filtered_mean": filter_results.filtered_mean[:, 0],
        "filtered_var": filter_results.filtered_cov[:, 0, 0],
        "ffbsm_smoothed_mean": results_ffbsm.smoothed_mean[:, 0],
        "ffbsm_smoothed_var": results_ffbsm.smoothed_cov[:, 0, 0],
        "ffbsi_smoothed_mean": results_ffbsi.smoothed_mean[:, 0],
        "ffbsi_smoothed_var": results_ffbsi.smoothed_cov[:, 0, 0],
    })
    results_df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"\nResults saved to {output_path}")
    print(f"  Rows: {len(results_df)}, Columns: {list(results_df.columns)}")

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame({
        "Method": ["Filter", "FFBSm", "FFBSi"],
        "RMSE": [rmse_filter, rmse_ffbsm, rmse_ffbsi],
        "MAE": [mae_filter, mae_ffbsm, mae_ffbsi],
        "Mean_Var": [mean_var_filter, mean_var_ffbsm, mean_var_ffbsi],
        "Var_Reduction_%": [
            0.0,
            (1 - mean_var_ffbsm / mean_var_filter) * 100,
            (1 - mean_var_ffbsi / mean_var_filter) * 100,
        ],
        "Time_s": [elapsed_filter, elapsed_ffbsm, elapsed_ffbsi],
    })
    print(summary.to_string(index=False, float_format="%.4f"))
    print(f"\nSeed: {SEED} | N: {N_PARTICLES} | T: {T} | M: {M_TRAJECTORIES}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
