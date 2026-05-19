"""Solution: Unscented PF, Regularized PF, and Bootstrap PF on SV model.

Compares three particle filter variants on the Stochastic Volatility model.
Saves RMSE, ESS, and timing results to CSV.
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
from particlefilterbox.filters.regularized import RegularizedPF
from particlefilterbox.filters.unscented import UnscentedPF


class SVModelFull(ParticleFilterModel):
    """SV model with full interface for UPF/RPF/BPF.

    Provides transition_function, observation_function, Q, R_obs
    for UnscentedPF, plus standard interface for Bootstrap/Regularized.
    """

    k_states = 1
    k_obs = 1

    def __init__(self, mu: float = -1.0, phi: float = 0.97, sigma: float = 0.15):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.sigma_x = sigma  # alias for UPF

    @property
    def params(self) -> dict[str, float]:
        return {"mu": self.mu, "phi": self.phi, "sigma": self.sigma}

    def initial_distribution(self, n_particles: int, rng: np.random.Generator) -> np.ndarray:
        var_stat = self.sigma**2 / (1.0 - self.phi**2)
        return rng.normal(self.mu, np.sqrt(var_stat), size=(n_particles, 1))

    def transition(self, particles: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
        eta = rng.standard_normal(size=particles.shape)
        return self.mu + self.phi * (particles - self.mu) + self.sigma * eta

    def transition_function(self, x: np.ndarray, t: int) -> np.ndarray:
        """Deterministic transition (for UPF sigma points)."""
        return self.mu + self.phi * (np.atleast_1d(x) - self.mu)

    def transition_mean(self, particles: np.ndarray, t: int) -> np.ndarray:
        return self.mu + self.phi * (particles - self.mu)

    def observation_function(self, x: np.ndarray, t: int) -> np.ndarray:
        """Deterministic observation mapping: E[y|h] = 0."""
        return np.zeros(1)

    def Q(self, t: int) -> np.ndarray:
        """Process noise covariance."""
        return np.array([[self.sigma**2]])

    def R_obs(self, t: int) -> np.ndarray:
        """Observation noise covariance (approximate for SV)."""
        return np.array([[np.exp(self.mu)]])

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        h = particles[:, 0]
        vol = np.exp(h / 2.0)
        return -0.5 * np.log(2.0 * np.pi) - np.log(vol) - 0.5 * (float(y_t[0]) / vol) ** 2


def main() -> None:
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "bootstrap_sir", "data")
    df = pd.read_csv(os.path.join(data_dir, "simulated_sv.csv"))
    y_obs = df["y_obs"].values
    h_true = df["h_true"].values
    T = len(df)

    model = SVModelFull(mu=-1.0, phi=0.97, sigma=0.15)
    print(f"SV Model parameters: {model.params}")
    print(f"Loaded {T} observations")

    N = 500
    seed = 42
    results: list[dict[str, object]] = []

    # Define filters to compare
    filters_to_run: list[tuple[str, type, dict]] = [
        ("UnscentedPF", UnscentedPF, {"alpha": 1.0, "beta": 2.0, "kappa": 0.0}),
        ("RegularizedPF", RegularizedPF, {"bandwidth": "silverman", "kernel": "gaussian"}),
        ("BootstrapPF", BootstrapPF, {}),
    ]

    for filter_name, filter_cls, filter_kwargs in filters_to_run:
        config = PFConfig(n_particles=N, resampling="systematic", seed=seed)
        pf = filter_cls(model, config, **filter_kwargs)

        t0 = time.time()
        res = pf.filter(y_obs)
        elapsed = time.time() - t0

        h_filt = res.filtered_means[:, 0]
        rmse = float(np.sqrt(np.mean((h_filt - h_true) ** 2)))
        mean_ess = float(np.mean(res.ess_history))
        min_ess = float(np.min(res.ess_history))
        median_ess = float(np.median(res.ess_history))
        log_lik = float(res.log_likelihood)

        results.append({
            "Filter": filter_name,
            "N": N,
            "RMSE": round(rmse, 6),
            "ESS_mean": round(mean_ess, 2),
            "ESS_min": round(min_ess, 2),
            "ESS_median": round(median_ess, 2),
            "log_likelihood": round(log_lik, 4),
            "time_s": round(elapsed, 4),
        })
        print(
            f"{filter_name:16s}: RMSE={rmse:.4f}, ESS_mean={mean_ess:.1f}, "
            f"ESS_min={min_ess:.1f}, log_lik={log_lik:.2f}, time={elapsed:.2f}s"
        )

    # Verify: all metrics are finite
    print("\n--- Consistency Checks ---")
    all_ok = True
    for r in results:
        rmse_val = r["RMSE"]
        ess_val = r["ESS_mean"]
        ll_val = r["log_likelihood"]
        if not np.isfinite(rmse_val):
            print(f"  WARNING: {r['Filter']} RMSE is not finite: {rmse_val}")
            all_ok = False
        if not np.isfinite(ess_val):
            print(f"  WARNING: {r['Filter']} ESS_mean is not finite: {ess_val}")
            all_ok = False
        if not np.isfinite(ll_val):
            print(f"  WARNING: {r['Filter']} log_likelihood is not finite: {ll_val}")
            all_ok = False

    if all_ok:
        print("  All metrics are finite and valid.")

    # Compare RMSE
    rmse_vals = {r["Filter"]: r["RMSE"] for r in results}
    best = min(rmse_vals, key=rmse_vals.get)
    print(f"\n  Best RMSE: {best} ({rmse_vals[best]:.4f})")

    # Save results
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "results_unscented_regularized.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
