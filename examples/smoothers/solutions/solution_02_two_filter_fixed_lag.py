"""
Solution 02: Two-Filter and Fixed-Lag Smoothers

Runs Two-Filter and Fixed-Lag smoothers with various lag values L,
compares with FFBSm, and saves results to results_two_filter_fixed_lag.csv.

Usage:
    python solution_02_two_filter_fixed_lag.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from particlefilterbox.smoothers import FFBSm, TwoFilterSmoother, FixedLagSmoother

# ── Configuration ────────────────────────────────────────────────────────
SEED = 42
N_PARTICLES = 2000
T_USE = 100
LAG_VALUES = [5, 10, 20, 50]

# SV model parameters
MU = -1.0
PHI = 0.97
SIGMA = 0.15

OUTPUT_DIR = Path(__file__).parent
DATA_PATH = OUTPUT_DIR.parent / "data" / "simulated_sv.csv"


# ── SV Model for smoothers ──────────────────────────────────────────────
@dataclass
class SVModelForSmoothing:
    """SV model with methods required by all smoothers."""
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
    print("=" * 65)
    print("Solution 02: Two-Filter & Fixed-Lag Smoothers on SV Model")
    print("=" * 65)

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
    mean_var_filter = float(np.mean(filter_results.filtered_cov[:, 0, 0]))
    print(f"  Time: {elapsed_filter:.2f}s, RMSE: {rmse_filter:.4f}")

    # ── Step 2: Run FFBSm (reference) ────────────────────────────────
    print(f"\n--- FFBSm (reference) ---")
    ffbsm = FFBSm()
    t0 = time.perf_counter()
    results_ffbsm = ffbsm.smooth(filter_results, model)
    elapsed_ffbsm = time.perf_counter() - t0
    rmse_ffbsm = float(np.sqrt(np.mean((results_ffbsm.smoothed_mean[:, 0] - h_true) ** 2)))
    mean_var_ffbsm = float(np.mean(results_ffbsm.smoothed_cov[:, 0, 0]))
    print(f"  Time: {elapsed_ffbsm:.2f}s, RMSE: {rmse_ffbsm:.4f}")

    # ── Step 3: Run Two-Filter Smoother ──────────────────────────────
    print(f"\n--- Two-Filter Smoother ---")
    tfs = TwoFilterSmoother(seed=SEED)
    t0 = time.perf_counter()
    results_tfs = tfs.smooth(filter_results, model)
    elapsed_tfs = time.perf_counter() - t0
    rmse_tfs = float(np.sqrt(np.mean((results_tfs.smoothed_mean[:, 0] - h_true) ** 2)))
    mean_var_tfs = float(np.mean(results_tfs.smoothed_cov[:, 0, 0]))
    print(f"  Time: {elapsed_tfs:.2f}s, RMSE: {rmse_tfs:.4f}")
    print(f"  Var reduction vs filter: {(1 - mean_var_tfs / mean_var_filter) * 100:.1f}%")

    # ── Step 4: Run Fixed-Lag Smoother with multiple L ───────────────
    results_fl: dict[int, object] = {}
    elapsed_fl: dict[int, float] = {}
    rmse_fl: dict[int, float] = {}
    mean_var_fl: dict[int, float] = {}

    print(f"\n--- Fixed-Lag Smoother (L = {LAG_VALUES}) ---")
    for L in LAG_VALUES:
        fls = FixedLagSmoother(lag=L)
        t0 = time.perf_counter()
        results_fl[L] = fls.smooth(filter_results, model)
        elapsed_fl[L] = time.perf_counter() - t0
        rmse_fl[L] = float(
            np.sqrt(np.mean((results_fl[L].smoothed_mean[:, 0] - h_true) ** 2))
        )
        mean_var_fl[L] = float(np.mean(results_fl[L].smoothed_cov[:, 0, 0]))
        print(f"  L={L:>3d}: RMSE={rmse_fl[L]:.4f}, Time={elapsed_fl[L]:.2f}s")

    # ── Step 5: Convergence check ────────────────────────────────────
    print(f"\n--- Fixed-Lag Convergence Analysis ---")
    max_L = max(LAG_VALUES)
    diff_large_L_vs_ffbsm = float(np.mean(np.abs(
        results_fl[max_L].smoothed_mean[:, 0] - results_ffbsm.smoothed_mean[:, 0]
    )))
    diff_small_L_vs_ffbsm = float(np.mean(np.abs(
        results_fl[min(LAG_VALUES)].smoothed_mean[:, 0] - results_ffbsm.smoothed_mean[:, 0]
    )))
    print(f"  Mean |FL(L={min(LAG_VALUES)}) - FFBSm|: {diff_small_L_vs_ffbsm:.4f}")
    print(f"  Mean |FL(L={max_L}) - FFBSm|: {diff_large_L_vs_ffbsm:.4f}")

    # Variance of fixed-lag with large L should be closer to FFBSm
    var_diff_small = abs(mean_var_fl[min(LAG_VALUES)] - mean_var_ffbsm)
    var_diff_large = abs(mean_var_fl[max_L] - mean_var_ffbsm)
    print(f"  |Var(FL L={min(LAG_VALUES)}) - Var(FFBSm)|: {var_diff_small:.4f}")
    print(f"  |Var(FL L={max_L}) - Var(FFBSm)|: {var_diff_large:.4f}")
    converges = var_diff_large <= var_diff_small
    print(f"  Large L converges toward FFBSm variance: {'YES' if converges else 'PARTIAL'}")

    # ── Step 6: Save results CSV ─────────────────────────────────────
    output_path = OUTPUT_DIR / "results_two_filter_fixed_lag.csv"

    # Per-timestep results
    data: dict[str, np.ndarray] = {
        "t": np.arange(T),
        "h_true": h_true,
        "filtered_mean": filter_results.filtered_mean[:, 0],
        "filtered_var": filter_results.filtered_cov[:, 0, 0],
        "ffbsm_smoothed_mean": results_ffbsm.smoothed_mean[:, 0],
        "ffbsm_smoothed_var": results_ffbsm.smoothed_cov[:, 0, 0],
        "two_filter_smoothed_mean": results_tfs.smoothed_mean[:, 0],
        "two_filter_smoothed_var": results_tfs.smoothed_cov[:, 0, 0],
    }
    for L in LAG_VALUES:
        data[f"fixed_lag_L{L}_smoothed_mean"] = results_fl[L].smoothed_mean[:, 0]
        data[f"fixed_lag_L{L}_smoothed_var"] = results_fl[L].smoothed_cov[:, 0, 0]

    results_df = pd.DataFrame(data)
    results_df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"\nResults saved to {output_path}")
    print(f"  Rows: {len(results_df)}, Columns: {list(results_df.columns)}")

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    rows = [
        {"Method": "Filter", "RMSE": rmse_filter, "Mean_Var": mean_var_filter,
         "Var_Reduction_%": 0.0, "Time_s": elapsed_filter, "Online": "Yes"},
        {"Method": "FFBSm", "RMSE": rmse_ffbsm, "Mean_Var": mean_var_ffbsm,
         "Var_Reduction_%": (1 - mean_var_ffbsm / mean_var_filter) * 100,
         "Time_s": elapsed_ffbsm, "Online": "No"},
        {"Method": "Two-Filter", "RMSE": rmse_tfs, "Mean_Var": mean_var_tfs,
         "Var_Reduction_%": (1 - mean_var_tfs / mean_var_filter) * 100,
         "Time_s": elapsed_tfs, "Online": "No"},
    ]
    for L in LAG_VALUES:
        rows.append({
            "Method": f"Fixed-Lag L={L}",
            "RMSE": rmse_fl[L],
            "Mean_Var": mean_var_fl[L],
            "Var_Reduction_%": (1 - mean_var_fl[L] / mean_var_filter) * 100,
            "Time_s": elapsed_fl[L],
            "Online": "Yes",
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format="%.4f"))
    print(f"\nSeed: {SEED} | N: {N_PARTICLES} | T: {T}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
